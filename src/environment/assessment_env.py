import gym
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
import json
import pandas as pd

class AssessmentEnv(gym.Env):
    """
    Gym environment for educational assessment using MARL.
    """
    
    def __init__(self, train_data_path: str, embedding_data_path: str):
        super().__init__()
        
        # Load training data
        with open(train_data_path, 'r') as f:
            self.train_data = json.load(f)
            
        # Load embedding data
        self.embedding_df = pd.read_pickle(embedding_data_path)
        
        # Define action spaces for Agent 1
        self.score_space = gym.spaces.Discrete(6)  # Scores 0-5
        self.feedback_space = gym.spaces.Discrete(11)  # Feedback IDs 0-10
        
        # Define observation space (384-dim embeddings for Q, R, A + context features)
        self.embedding_dim = 768
        self.context_dim = 10
        self.observation_space = gym.spaces.Dict({
            'question_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),
            'response_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),
            'answer_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),
            'context_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.context_dim,), dtype=np.float32)
        })
        
        # Current episode tracking
        self.current_episode = 0
        self.max_episodes = len(self.train_data)
        
        # Initialize episode data
        self.reset()
    
    def _parse_embedding(self, embedding_str: Any) -> np.ndarray:
        """
        Parse embedding string to numpy array.
        
        Args:
            embedding_str: String representation of embedding or numpy array
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        try:
            if isinstance(embedding_str, str):
                # Remove brackets and split by comma
                embedding_str = embedding_str.strip('[]')
                values = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
            elif isinstance(embedding_str, np.ndarray):
                # If already a numpy array, use it directly
                values = embedding_str
            else:
                raise ValueError("Unsupported embedding format")
            
            # Convert to numpy array
            embedding = np.array(values, dtype=np.float32)
            
            # Ensure correct shape
            if embedding.shape != (self.embedding_dim,):
                print(f"Warning: Expected embedding dimension {self.embedding_dim}, got {embedding.shape}")
                # Pad or truncate to correct size
                if len(embedding) < self.embedding_dim:
                    embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
                else:
                    embedding = embedding[:self.embedding_dim]
            
            return embedding
            
        except (ValueError, AttributeError) as e:
            print(f"Warning: Error parsing embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation based on episode data.
        
        Returns:
            Dictionary containing embeddings and context features,
            where each embedding has shape (384,) and context features has shape (10,)
        """
        current_data = self.train_data[self.current_episode]
        event_id = current_data['event_id']
        
        # Get embeddings for the current event
        embeddings = self.embedding_df[self.embedding_df['event_id'] == event_id]
        
        if len(embeddings) == 0:
            # If no embeddings available, use zeros
            return {
                'question_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
                'response_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
                'answer_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
                'context_features': self._encode_context_features(current_data)
            }
        
        # Get the first (and should be only) row
        embedding_row = embeddings.iloc[0]
        
        # Parse embeddings
        question_embedding = self._parse_embedding(embedding_row['embedding_question'])
        rubric_embedding = self._parse_embedding(embedding_row['embedding_rubric'])
        answer_embedding = self._parse_embedding(embedding_row['embedding_answer'])
        
        return {
            'question_embedding': question_embedding,
            'response_embedding': answer_embedding,  # Using answer embedding for response
            'answer_embedding': rubric_embedding,  # Using rubric embedding for answer
            'context_features': self._encode_context_features(current_data)
        }
    
    def _encode_context_features(self, data: Dict) -> np.ndarray:
        """
        Encode context features from the data.
        
        Args:
            data: Dictionary containing event data
            
        Returns:
            Context features array of shape (10,)
        """
        features = []
        
        # Subject encoding (one-hot)
        subjects = ['4th Grade Mathematics', '4th Grade English', '4th Grade Science']
        subject_encoding = [1 if data['subject'] == s else 0 for s in subjects]
        features.extend(subject_encoding)
        
        # Question type encoding (one-hot)
        question_types = ['Calculation', 'Short Answer', 'Essay', 'Definition', 'Diagram Labeling']
        question_type_encoding = [1 if data['question_type'] == t else 0 for t in question_types]
        features.extend(question_type_encoding)
        
        # Normalized historical averages
        features.append(data['student_historical_avg'] / 5.0)  # Normalize to [0,1]
        features.append(data['cohort_avg_score'] / 5.0)  # Normalize to [0,1]
            
        features = np.array(features, dtype=np.float32)
        assert features.shape == (self.context_dim,), f"Context features shape mismatch: {features.shape}"
        return features
    
    def check_consistency(self, score: int, feedback_id: int) -> float:
        """ Placeholder for checking consistency between score and feedback.
            Returns a bonus reward. Needs to be implemented based on domain knowledge.
        """
        # TODO: Implement actual consistency logic based on score-feedback pairings
        # Example: if score is 5, feedback_id 1 might be consistent (bonus 1.0)
        #          if score is 1, feedback_id 7 might be consistent (bonus 1.0)
        #          otherwise, bonus 0.0
        # This requires defining what combinations are considered consistent.
        # For now, returning 0.0 as a placeholder.
        return 0.0

    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Dictionary with 'score' and 'feedback_id' keys
            
        Returns:
            observation: Next state observation
            reward: Reward for the action
            done: Whether the episode is complete
            info: Additional information
        """
        # Validate action
        assert 0 <= action['score'] <= 5, "Invalid score"
        assert 0 <= action['feedback_id'] <= 10, "Invalid feedback ID"
        
        # Get current episode data
        current_data = self.train_data[self.current_episode]
        
        # Calculate reward based on expert labels
        pred_score = action['score']
        true_score = current_data['expert_score']
        pred_feedback = action['feedback_id']
        true_feedback = current_data['expert_feedback_comment_id']

        # Graduated score reward (less penalty for close predictions, steeper for larger errors)
        score_error = abs(pred_score - true_score)
        if score_error <= 1:
            score_reward = -score_error  # Penalty of 0 or -1
        else:
            score_reward = -2 * score_error # Penalty of -4, -6, -8, -10 for errors of 2, 3, 4, 5
        
        # Feedback reward with more granularity and impact
        if pred_feedback == true_feedback:
            feedback_reward = 3.0  # Reward for correct feedback
        else:
            feedback_reward = -3.0 # Strong penalty for incorrect feedback (kept from previous enhancement)
        
        # Consistency bonus (placeholder)
        consistency_bonus = self.check_consistency(pred_score, pred_feedback)
        
        reward = score_reward + feedback_reward + consistency_bonus
        
        # Move to next episode
        self.current_episode += 1
        done = self.current_episode >= self.max_episodes
        
        # Get next observation
        observation = self._get_observation() if not done else None
        
        info = {
            'episode': self.current_episode,
            'score_error': score_error,
            'feedback_correct': pred_feedback == true_feedback
        }
        
        return observation, reward, done, info
    
    def reset(self, episode_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset the environment to start a new episode.
        
        Args:
            episode_idx: Optional specific episode index to reset to. 
                         If None, resets to the beginning (episode 0).
        """
        if episode_idx is not None:
            if 0 <= episode_idx < self.max_episodes:
                self.current_episode = episode_idx
            else:
                print(f"Warning: Provided episode_idx {episode_idx} is out of bounds (0-{self.max_episodes-1}). Defaulting to 0.")
                self.current_episode = 0
        else:
            self.current_episode = 0 # Default behavior for training
            
        if self.current_episode >= self.max_episodes: # Should not happen if data loaded
            print(f"Warning: current_episode {self.current_episode} is >= max_episodes {self.max_episodes} after reset. Clamping.")
            self.current_episode = self.max_episodes -1 if self.max_episodes > 0 else 0
            if self.max_episodes == 0:
                 raise ValueError("Cannot reset environment, max_episodes is 0 (no data loaded).")

        return self._get_observation()
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the current state of the environment."""
        if mode == 'human':
            current_data = self.train_data[self.current_episode]
            print(f"\nEpisode {self.current_episode + 1}/{self.max_episodes}")
            print(f"Subject: {current_data['subject']}")
            print(f"Question: {current_data['question_text']}")
            print(f"Student Answer: {current_data['student_answer_text']}")
            print(f"Expert Score: {current_data['expert_score']}")
            print(f"Expert Feedback ID: {current_data['expert_feedback_comment_id']}")
        return None 