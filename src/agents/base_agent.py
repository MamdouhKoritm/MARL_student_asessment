import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import random
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.cat(states)
        actions = torch.tensor(actions).unsqueeze(1) # Ensure actions are (batch_size, 1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)

class BaseAgent(nn.Module):
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 10,
                 weight_decay: float = 0.0):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.weight_decay_val = weight_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network and optimizer are initialized by subclasses
        self.network = None
        self.target_network = None
        self.optimizer = None
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.train_step_counter = 0
    
    def _init_optimizer(self):
        if self.network is None:
            raise ValueError("self.network must be defined by the subclass before calling _init_optimizer.")
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay_val)
        print(f"Optimizer initialized with lr={self.learning_rate} and weight_decay={self.weight_decay_val}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Pass state through network to get Q-values."""
        if self.network is None:
            raise NotImplementedError("Network not defined in BaseAgent subclass.")
        return self.network(state) # Shape: (batch_size, action_dim)
    
    def _preprocess_observation(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Convert observation dictionary to tensor. To be implemented by subclasses."""
        raise NotImplementedError
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)  # Explore
        else:
            with torch.no_grad():
                q_values = self.forward(state) # Shape: (batch_size, action_dim)
                return q_values.argmax(dim=1).item()  # Exploit
    
    def update_epsilon(self) -> None:
        """Update epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train_step(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        if self.optimizer is None:
            self._init_optimizer()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).float()
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).float()

        curr_q = self.network(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))
        
        loss = nn.SmoothL1Loss()(curr_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Proper gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        # Always apply gradients unless loss is NaN
        if not torch.isnan(loss):
            self.optimizer.step()
        
        # Update target network
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        if self.network is None or self.optimizer is None:
            print("Warning: Network or optimizer not initialized. Cannot save agent state.")
            return
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        if self.network is None:
            # If network is not defined by subclass, we can't proceed
            raise ValueError("Network must be defined by subclass before loading agent state.")

        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
        # Crucially, initialize optimizer *after* loading network and *before* loading optimizer state
        # This ensures optimizer is created with the correct parameters (including weight_decay)
        # that were set during the agent's __init__.
        self._init_optimizer() 
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        
        if self.target_network is not None:
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_network.eval()
        else:
            print("Warning: Target network not defined. Cannot load its state.")
            
        print(f"Agent loaded from {path}") 