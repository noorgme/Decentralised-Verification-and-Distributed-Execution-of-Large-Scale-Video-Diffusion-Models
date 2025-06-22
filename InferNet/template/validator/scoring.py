import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import bittensor as bt
from typing import Tuple, List
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
import torchvision.transforms as transforms
import lpips

def verify_video_authenticity_common(video_path: str) -> bool:
    """Common video authenticity verification using entropy and frame differences."""
    try:
        cap = cv2.VideoCapture(video_path)
        prev_frame = None
        frame_diffs = []
        entropies = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Frame entropy
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropies.append(entropy)
            
            if prev_frame is not None:
                # Frame difference
                diff = cv2.absdiff(frame, prev_frame)
                frame_diffs.append(np.mean(diff))
            
            prev_frame = frame.copy()
        
        cap.release()
        
        if not frame_diffs or not entropies:
            return False
            
        # Check entropy distribution
        entropy_mean = np.mean(entropies)
        entropy_std = np.std(entropies)
        # Log entropy_mean and entropy_std
        bt.logging.info(f"Entropy mean: {entropy_mean}, Entropy std: {entropy_std}")

        if entropy_std < 0.01 or entropy_mean < 0.01:
            return False
            
        # Check frame differences
        diff_mean = np.mean(frame_diffs)
        diff_std = np.std(frame_diffs)
        # Log diff_mean and diff_std
        bt.logging.info(f"Diff mean: {diff_mean}, Diff std: {diff_std}")

        if diff_std < 0.01 or diff_mean < 0.01:
            return False
            
        return True
        
    except Exception as e:
        bt.logging.error(f"Error verifying video authenticity: {str(e)}")
        return False

class CLIPScorer:
    def __init__(self):
        """Video quality scorer using CLIP for frame-wise prompt alignment."""
        # Load CLIP model with standard processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def compute_quality_score(self, video_path: str, prompt: str) -> float:
        """
        Computes video quality score using frame-wise CLIP similarity.
        
        From paper Section 3.3.3:
        Q = 1/F * Σ(cosine_sim(E_text, E_frame_i))
        where E_frame_i = f_vision(frame_i) / ||f_vision(frame_i)||_2
        """
        try:
            # Validate inputs
            if not prompt or prompt.strip() == "":
                bt.logging.warning(f"Empty or None prompt provided, using default")
                prompt = "a video"
            
            bt.logging.info(f"Computing CLIP score for prompt: '{prompt}'")
            
            # Get prompt features
            text_inputs = self.clip_processor(text=prompt, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)
            
            # Get frame features
            cap = cv2.VideoCapture(video_path)
            frame_features = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(frame)
                    image_features = F.normalize(image_features, dim=-1)
                    frame_features.append(image_features)
                    frame_count += 1
            
            cap.release()
            
            bt.logging.info(f"Processed {frame_count} frames from video")
            
            if not frame_features:
                bt.logging.warning("No frames extracted from video")
                return 0.0
            
            # Average similarity across frames
            frame_features = torch.cat(frame_features, dim=0)
            similarities = torch.matmul(frame_features, text_features.T).squeeze()
            score = float(torch.mean(similarities).cpu())
            
            bt.logging.info(f"CLIP score computed: {score:.4f}")
            return score
            
        except Exception as e:
            bt.logging.error(f"Error computing quality score: {str(e)}")
            return 0.0

    def verify_video_authenticity(self, video_path: str) -> bool:
        """Checks if video is authentic using entropy and frame differences."""
        return verify_video_authenticity_common(video_path)


class MDVQS:
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        # Video quality scorer using CLIP, LPIPS and optical flow
        self.alpha = alpha  # Prompt fidelity weight
        self.beta = beta    # Video quality weight
        self.gamma = gamma  # Temporal consistency weight
        # Load models with standard processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.lpips_model = lpips.LPIPS(net='alex')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model.to(self.device)
        self.lpips_model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def compute_md_vqs(self, video_path: str, prompt: str) -> Tuple[float, float, float, float]:
        # Computes overall video quality score from all metrics
        try:
            # Get individual scores
            pf = self.compute_prompt_fidelity(video_path, prompt)
            vq = self.compute_video_quality(video_path)
            tc = self.compute_temporal_consistency(video_path)
            # Log pf, vq, tc
            bt.logging.info(f"PF: {pf}, VQ: {vq}, TC: {tc}")
            
            # Weighted sum
            total_score = (
                self.alpha * pf +
                self.beta * vq +
                self.gamma * tc
            )
            
            return pf, vq, tc, total_score
            
        except Exception as e:
            bt.logging.error(f"Error computing MD-VQS: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0

    def compute_prompt_fidelity(self, video_path: str, prompt: str) -> float:
        """Compute prompt fidelity score using CLIP similarity."""
        try:
            # Validate inputs
            if not prompt or prompt.strip() == "":
                bt.logging.warning(f"Empty or None prompt provided, using default")
                prompt = "a video"
            
            bt.logging.info(f"Computing prompt fidelity for prompt: '{prompt}'")
            
            # Get prompt features
            text_inputs = self.clip_processor(text=prompt, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)
            
            # Get frame features
            cap = cv2.VideoCapture(video_path)
            frame_features = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(frame)
                    image_features = F.normalize(image_features, dim=-1)
                    frame_features.append(image_features)
                    frame_count += 1
            
            cap.release()
            
            bt.logging.info(f"Processed {frame_count} frames for prompt fidelity")
            
            if not frame_features:
                bt.logging.warning("No frames extracted from video for prompt fidelity")
                return 0.0
            
            # Average similarity across frames
            frame_features = torch.cat(frame_features, dim=0)
            similarities = torch.matmul(frame_features, text_features.T).squeeze()
            score = float(torch.mean(similarities).cpu())
            
            bt.logging.info(f"Prompt fidelity score computed: {score:.4f}")
            return score
            
        except Exception as e:
            bt.logging.error(f"Error computing prompt fidelity: {str(e)}")
            return 0.0

    def compute_video_quality(self, video_path: str) -> float:
        """Compute video quality score using LPIPS."""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_qualities = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    frame_qualities.append(self.lpips_model(frame, frame).item())

            cap.release()
            
            if not frame_qualities:
                return 0.0
                
            return float(np.mean(frame_qualities))
            
        except Exception as e:
            bt.logging.error(f"Error computing video quality: {str(e)}")
            return 0.0

    def compute_temporal_consistency(self, video_path: str) -> float:
        """Compute temporal consistency score using optical flow."""
        try:
            cap = cv2.VideoCapture(video_path)
            prev_frame = None
            consistency_scores = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if prev_frame is not None:
                    # Compute optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    # Compute consistency score
                    consistency = np.mean(np.abs(flow))
                    consistency_scores.append(consistency)
                
                prev_frame = frame.copy()
            
            cap.release()
            
            if not consistency_scores:
                return 0.0
                
            return float(np.mean(consistency_scores))
            
        except Exception as e:
            bt.logging.error(f"Error computing temporal consistency: {str(e)}")
            return 0.0

    def verify_video_authenticity(self, video_path: str) -> bool:
        """Checks if video is authentic using entropy and frame differences."""
        return verify_video_authenticity_common(video_path)



MDVQS_scorer = MDVQS()
CLIP_scorer = CLIPScorer()

def compute_quality_score_mdvqs(video_path: str, prompt: str) -> float:
    """
    Wrapper around VideoScorer.compute_quality_score.
    """
    return MDVQS_scorer.compute_quality_score(video_path, prompt)

def verify_video_authenticity_mdvqs(video_path: str) -> bool:
    """
    Wrapper around VideoScorer.verify_video_authenticity.
    """
    return MDVQS_scorer.verify_video_authenticity(video_path)

def compute_quality_score_clip(video_path: str, prompt: str) -> float:
    return CLIP_scorer.compute_quality_score(video_path, prompt)

def verify_video_authenticity_clip(video_path: str) -> bool:
    return CLIP_scorer.verify_video_authenticity(video_path)





