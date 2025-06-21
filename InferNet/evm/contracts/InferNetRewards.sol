// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract InferNetRewards {
    struct Submission {
        address miner;
        uint256 mdvqsScore;
        bytes proof;
        bool rewarded;
        bytes32 merkleRoot; // Decoded Merkle root from proof
        bytes signature;    // Decoded signature from proof
    }

    // User submits a prompt and deposit for a requestId
    mapping(uint256 => address) public userRequests; // requestId => user
    mapping(uint256 => uint256) public rewardPools;  // requestId => TAO amount
    mapping(uint256 => Submission[]) public submissions; // requestId => miner submissions
    mapping(address => uint256) public minerStakes; // miner => staked amount
    mapping(address => uint256) public minerRewards; // miner => claimable rewards
    // User submits a prompt and deposit for a requestId (commit-reveal)
    mapping(uint256 => bytes32) public promptHashes; // requestId => promptHash

    address public validator; // Only validator can record/distribute
    IERC20 public taoToken;

    modifier onlyValidator() {
        require(msg.sender == validator, "Only validator");
        _;
    }

    event Deposit(address indexed user, uint256 requestId, uint256 amount, bytes32 promptHash);
    event SubmissionRecorded(uint256 requestId, address miner, uint256 score);
    event RewardsDistributed(uint256 requestId, uint256 totalReward);
    event RewardClaimed(address miner, uint256 amount);
    event StakeSlashed(address miner, uint256 amount);
    event RefundIssued(address indexed user, uint256 requestId, uint256 amount);

    constructor(address _validator, address _taoToken) {
        validator = _validator;
        taoToken = IERC20(_taoToken);
    }

    // User deposits for a prompt (TAO) with commit-reveal
    function depositAndCommit(uint256 requestId, bytes32 promptHash, uint256 amount) external {
        require(amount > 0, "Deposit required");
        require(userRequests[requestId] == address(0), "Request already exists");
        require(taoToken.transferFrom(msg.sender, address(this), amount), "TAO transfer failed");
        userRequests[requestId] = msg.sender;
        rewardPools[requestId] += amount;
        promptHashes[requestId] = promptHash;
        emit Deposit(msg.sender, requestId, amount, promptHash);
    }

    // Validator records a miner's submission
    function recordSubmission(
        uint256 requestId,
        address miner,
        uint256 mdvqsScore,
        bytes calldata proof
    ) external onlyValidator {
        require(minerStakes[miner] > 0, "no stake");
        // Decode the proof bytes (bytes32 merkleRoot, bytes signature)
        (bytes32 merkleRoot, bytes memory signature) = abi.decode(proof, (bytes32, bytes));
        submissions[requestId].push(Submission(miner, mdvqsScore, proof, false, merkleRoot, signature));
        // Optionally emit for auditability
        emit SubmissionRecorded(requestId, miner, mdvqsScore);
        // You can also emit the decoded values if desired:
        // emit ProofDecoded(requestId, miner, merkleRoot, signature);
    }

    // Validator distributes rewards for a request
    function distributeRewards(uint256 requestId) external onlyValidator {
        uint256 pool = rewardPools[requestId];
        require(pool > 0, "already settled");
        Submission[] storage subs = submissions[requestId];
        uint256 totalScore = 0;
        for (uint i = 0; i < subs.length; i++) {
            totalScore += subs[i].mdvqsScore;
        }
        require(totalScore > 0, "No valid submissions");

        for (uint i = 0; i < subs.length; i++) {
            if (!subs[i].rewarded) {
                uint256 reward = (pool * subs[i].mdvqsScore) / totalScore;
                minerRewards[subs[i].miner] += reward;
                subs[i].rewarded = true;
            }
        }
        rewardPools[requestId] = 0;
        emit RewardsDistributed(requestId, pool);
    }

    // Miners claim their rewards
    function claimReward() external {
        uint256 amount = minerRewards[msg.sender];
        require(amount > 0, "No rewards");
        minerRewards[msg.sender] = 0;
        require(taoToken.transfer(msg.sender, amount), "TAO transfer failed");
        emit RewardClaimed(msg.sender, amount);
    }

    // Validator can slash a miner's stake (for cheating)
    function slashStake(address miner, uint256 amount) external onlyValidator {
        require(minerStakes[miner] >= amount, "Not enough stake");
        minerStakes[miner] -= amount;
        // Optionally: add slashed amount to a global pool or burn
        emit StakeSlashed(miner, amount);
    }

    // Miners can stake (optional, for slashing)
    function stake(uint256 amount) external {
        require(amount > 0, "Stake required");
        require(taoToken.transferFrom(msg.sender, address(this), amount), "TAO transfer failed");
        minerStakes[msg.sender] += amount;
    }

    // Reclaim deposit if not used
    function refundUnused(uint256 requestId) external {
        address user = userRequests[requestId];
        uint256 pool = rewardPools[requestId];
        require(pool > 0, "No funds to refund");
        require(submissions[requestId].length == 0, "Submissions exist");
        
        require(msg.sender == user, "Only original user can refund (or add timeout logic)");
        rewardPools[requestId] = 0;
        userRequests[requestId] = address(0);
        require(taoToken.transfer(user, pool), "TAO refund failed");
        emit RefundIssued(user, requestId, pool);
    }

    // No-op handlers
    function supportsInterface(bytes4) public pure returns (bool) {
    return false;
    }

    function name() public pure returns (string memory) {
        return "InferNet";
    }

    function decimals() public pure returns (uint8) {
        return 18;
    }

} 