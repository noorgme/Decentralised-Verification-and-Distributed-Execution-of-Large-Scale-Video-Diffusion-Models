import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import bittensor as bt
from typing import Tuple, List
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms

class VideoScorer:
    def __init__(self):
        """Video quality scorer using CLIP for frame-wise prompt alignment."""
        # Load CLIP model
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
            # Get prompt features
            text_inputs = self.clip_processor(text=prompt, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)
            
            # Get frame features
            cap = cv2.VideoCapture(video_path)
            frame_features = []
            
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
            
            cap.release()
            
            if not frame_features:
                return 0.0
            
            # Average similarity across frames
            frame_features = torch.cat(frame_features, dim=0)
            similarities = torch.matmul(frame_features, text_features.T).squeeze()
            return float(torch.mean(similarities).cpu())
            
        except Exception as e:
            bt.logging.error(f"Error computing quality score: {str(e)}")
            return 0.0

    def verify_video_authenticity(self, video_path: str) -> bool:
        """Checks if video is authentic using entropy and frame differences."""
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
            if entropy_std < 0.1 or entropy_mean < 3.0:
                return False
                
            # Check frame differences
            diff_mean = np.mean(frame_diffs)
            diff_std = np.std(frame_diffs)
            if diff_std < 0.1 or diff_mean < 1.0:
                return False
                
            return True
            
        except Exception as e:
            bt.logging.error(f"Error verifying video authenticity: {str(e)}")
            return False 

# Original MDVQS formula (commented out for reference):
"""
class MDVQS:
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        # Video quality scorer using CLIP, LPIPS and optical flow
        self.alpha = alpha  # Prompt fidelity weight
        self.beta = beta    # Video quality weight
        self.gamma = gamma  # Temporal consistency weight
        
        # Load models
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
"""

            # ------------------------------------------------------------------
# Convenience module-level API so validator.py and tests can import
# ------------------------------------------------------------------
_scorer = VideoScorer()

def compute_quality_score(video_path: str, prompt: str) -> float:
    """
    Wrapper around VideoScorer.compute_quality_score.
    """
    return _scorer.compute_quality_score(video_path, prompt)

def verify_video_authenticity(video_path: str) -> bool:
    """
    Wrapper around VideoScorer.verify_video_authenticity.
    """
    return _scorer.verify_video_authenticity(video_path)
