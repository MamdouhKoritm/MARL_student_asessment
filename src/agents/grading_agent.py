import torch
import torch.nn as nn
import torch.nn.functional as F # Import F
from typing import Dict, Any, Tuple
import numpy as np
import gym # Added gym import
from .base_agent import BaseAgent

# Define the Dueling Network
class DuelingNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(DuelingNetwork, self).__init__()
        # Shared feature layers with residual connections
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        
        # Value stream with layer normalization
        self.value_stream_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_stream_ln1 = nn.LayerNorm(hidden_dim // 2)
        self.value_stream_relu = nn.ReLU()
        self.value_stream_dropout = nn.Dropout(p=0.2)
        self.value_stream_fc2 = nn.Linear(hidden_dim // 2, 1)
        
        # Advantage stream with layer normalization
        self.adv_stream_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.adv_stream_ln1 = nn.LayerNorm(hidden_dim // 2)
        self.adv_stream_relu = nn.ReLU()
        self.adv_stream_dropout = nn.Dropout(p=0.2)
        self.adv_stream_fc2 = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature layer processing with residual connection
        x1 = self.dropout1(self.relu1(self.ln1(self.fc1(x))))
        x2 = self.fc2(x1)
        x2 = self.ln2(x2)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)
        features = x2 + x1  # Residual connection

        # Value stream with layer norm
        val = self.value_stream_fc1(features)
        val = self.value_stream_ln1(val)
        val = self.value_stream_relu(val)
        val = self.value_stream_dropout(val)
        value = self.value_stream_fc2(val)

        # Advantage stream with layer norm
        adv = self.adv_stream_fc1(features)
        adv = self.adv_stream_ln1(adv)
        adv = self.adv_stream_relu(adv)
        adv = self.adv_stream_dropout(adv)
        advantage = self.adv_stream_fc2(adv)

        # Combine streams
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # --- DEBUG PRINTS for DuelingNetwork streams ---
        # print(f"DEBUG_DUELING: value_stream out - Min: {value.min().item()}, Max: {value.max().item()}, Mean: {value.mean().item()}, Shape: {value.shape}")
        # print(f"DEBUG_DUELING: advantage_stream out - Min: {advantage.min().item()}, Max: {advantage.max().item()}, Mean: {advantage.mean().item()}, Shape: {advantage.shape}")
        # --- END DEBUG PRINTS ---

        return qvals

class GradingAgent(BaseAgent):
    def __init__(self,
                 hidden_dim: int = 256,
                 learning_rate: float = 1e-5,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 50,
                 weight_decay: float = 1e-5):
        
        # State dimension: 3 embeddings (768 each) + context features (10)
        self.embedding_dim = 768
        self.context_dim = 10
        state_dim = self.embedding_dim * 3 + self.context_dim  # 2314
        
        # Define individual action spaces
        self.score_action_space = gym.spaces.Discrete(6)  # Scores 0-5
        self.feedback_action_space = gym.spaces.Discrete(11)  # Feedback IDs 0-10

        # Action dimension: scores (6) * feedback IDs (11)
        action_dim = self.score_action_space.n * self.feedback_action_space.n  # 66 total actions
        
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            weight_decay=weight_decay
        )
        
        # Use Dueling Network
        self.network = DuelingNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_network = DuelingNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval() # Target network is not trained directly

        self._init_optimizer() # Initialize optimizer after networks are defined
    
    def _preprocess_observation(self, observation: Dict[str, Any]) -> torch.Tensor:
        """
        Convert observation dictionary to tensor.
        
        Args:
            observation: Dictionary containing embeddings and context features
            
        Returns:
            Preprocessed observation tensor of shape (batch_size, state_dim)
            where state_dim = 3 * embedding_dim + context_dim = 2314
        """
        # Convert numpy arrays to tensors
        question_embedding = torch.from_numpy(observation['question_embedding']).float()  # (768,)
        response_embedding = torch.from_numpy(observation['response_embedding']).float()  # (768,)
        answer_embedding = torch.from_numpy(observation['answer_embedding']).float()    # (768,)
        context_features = torch.from_numpy(observation['context_features']).float()    # (10,)
        
        # Ensure all embeddings are flattened and have correct shape
        question_embedding = question_embedding.view(-1)  # Flatten to (768,)
        response_embedding = response_embedding.view(-1)  # Flatten to (768,)
        answer_embedding = answer_embedding.view(-1)      # Flatten to (768,)
        context_features = context_features.view(-1)      # Flatten to (10,)
        
        # Concatenate all features
        state = torch.cat([
            question_embedding,  # (768,)
            response_embedding,  # (768,)
            answer_embedding,    # (768,)
            context_features    # (10,)
        ])  # Result: (2314,)
        
        # Add batch dimension
        state = state.unsqueeze(0)  # Shape: (1, 2314)
        
        # Move to device
        state = state.to(self.device)
            
        return state
    
    def decode_action(self, action_idx: int) -> Dict[str, int]:
        """
        Decode flat action index into score and feedback ID.
        
        Args:
            action_idx: Integer index of the action
            
        Returns:
            Dictionary containing score and feedback_id
        """
        score = action_idx // 11  # Integer division
        feedback_id = action_idx % 11
        
        return {
            'score': score,
            'feedback_id': feedback_id
        }
    
    def encode_action(self, score: int, feedback_id: int) -> int:
        """
        Encode score and feedback ID into flat action index.
        
        Args:
            score: Integer score (0-5)
            feedback_id: Integer feedback ID (0-10)
            
        Returns:
            Flat action index
        """
        return score * 11 + feedback_id
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> Dict[str, int]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor of shape (batch_size, state_dim)
            training: Whether to use epsilon-greedy exploration
            
        Returns:
            Dictionary containing score and feedback_id
        """
        action_idx = super().select_action(state, training)
        return self.decode_action(action_idx)
    
    def train_step(self) -> float:
        """
        Perform one training step.
        
        Returns:
            Loss value
        """
        loss = super().train_step()
        
        # Add reward shaping or additional training logic here if needed
        
        return loss 