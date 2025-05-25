import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np
import gym

from .base_agent import BaseAgent

class BiasDetectorNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        # First block
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Second block with residual connection
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        
        # Third block with residual connection
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First block
        x1 = self.fc1(x)
        x1 = self.ln1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)
        
        # Second block with residual
        x2 = self.fc2(x1)
        x2 = self.ln2(x2)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)
        x2 = x2 + x1  # Residual connection
        
        # Third block with residual
        x3 = self.fc3(x2)
        x3 = self.ln3(x3)
        x3 = self.relu3(x3)
        x3 = self.dropout3(x3)
        x3 = x3 + x2  # Residual connection
        
        # Output
        return self.fc_out(x3)

class BiasDetectorAgent(BaseAgent):
    def __init__(self,
                 context_dim: int = 10,
                 agent1_score_actions: int = 6, # 0-5
                 agent1_feedback_actions: int = 11, # 0-10
                 hidden_dim: int = 128, # Smaller default hidden_dim for a simpler agent
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 10,
                 weight_decay: float = 0.0):

        self.context_dim = context_dim
        self.agent1_score_actions = agent1_score_actions
        self.agent1_feedback_actions = agent1_feedback_actions
        
        # State dimension: context_features + one_hot_encoded_agent1_score + one_hot_encoded_agent1_feedback
        state_dim = self.context_dim + self.agent1_score_actions + self.agent1_feedback_actions # 10 + 6 + 11 = 27
        
        # Action dimension: binary (e.g., 0: no bias, 1: bias detected)
        action_dim = 2
        
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
        
        self.network = BiasDetectorNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_network = BiasDetectorNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self._init_optimizer()

    def _preprocess_observation(self, observation: Dict[str, Any]) -> torch.Tensor:
        """
        Convert observation dictionary to tensor for Agent 2.
        Observation is expected to contain:
        - 'context_features': np.ndarray of shape (context_dim,)
        - 'agent1_score': int, score from Agent 1 (can be current or previous depending on training setup)
        - 'agent1_feedback_id': int, feedback_id from Agent 1 (can be current or previous)
        """
        context_features = torch.from_numpy(observation['context_features']).float().to(self.device) # (context_dim,)
        
        agent1_score = observation['agent1_score']
        agent1_feedback_id = observation['agent1_feedback_id']
        
        # One-hot encode Agent 1's action
        score_one_hot = F.one_hot(torch.tensor(agent1_score), num_classes=self.agent1_score_actions).float().to(self.device)
        feedback_one_hot = F.one_hot(torch.tensor(agent1_feedback_id), num_classes=self.agent1_feedback_actions).float().to(self.device)
        
        # Ensure tensors are flat
        context_features = context_features.view(-1)
        score_one_hot = score_one_hot.view(-1)
        feedback_one_hot = feedback_one_hot.view(-1)

        state = torch.cat([
            context_features,
            score_one_hot,
            feedback_one_hot
        ]) # Shape: (state_dim,)
        
        # Add batch dimension
        state = state.unsqueeze(0) # Shape: (1, state_dim)
        
        return state.to(self.device)

    # select_action is inherited from BaseAgent and should work as is for discrete actions.
    # train_step is inherited from BaseAgent and should work as is.
    # save/load are inherited from BaseAgent. 