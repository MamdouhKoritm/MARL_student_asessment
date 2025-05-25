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
        print(f"Loading training data from {train_data_path}")
        with open(train_data_path, 'r') as f:
            self.train_data = json.load(f)
        
        # Verify training data structure
        if not isinstance(self.train_data, list) or len(self.train_data) == 0:
            raise ValueError("Training data must be a non-empty list")
        
        required_fields = ['event_id', 'subject', 'question_type', 'student_historical_avg', 
                         'cohort_avg_score', 'expert_score', 'expert_feedback_comment_id']
        
        for item in self.train_data[:5]:  # Check first 5 items
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                raise ValueError(f"Training data items missing required fields: {missing_fields}")
            
        # Load embedding data
        print(f"Loading embedding data from {embedding_data_path}")
        self.embedding_df = pd.read_pickle(embedding_data_path)
        
        # Verify embedding data structure
        required_columns = ['event_id', 'embedding_question', 'embedding_rubric', 'embedding_answer']
        missing_columns = [col for col in required_columns if col not in self.embedding_df.columns]
        if missing_columns:
            raise ValueError(f"Embedding data missing required columns: {missing_columns}")
            
        # Verify embeddings match training data
        train_event_ids = set(item['event_id'] for item in self.train_data)
        embedding_event_ids = set(self.embedding_df['event_id'])
        missing_embeddings = train_event_ids - embedding_event_ids
        if missing_embeddings:
            print(f"Warning: {len(missing_embeddings)} training items missing embeddings")
            
        # Define action spaces for Agent 1
        self.score_space = gym.spaces.Discrete(6)  # Scores 0-5
        self.feedback_space = gym.spaces.Discrete(11)  # Feedback IDs 0-10
        
        # Define observation space
        self.embedding_dim = 768
        self.context_dim = 10
        self.observation_space = gym.spaces.Dict({
            'question_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),
            'response_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),
            'answer_embedding': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),
            'context_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.context_dim,), dtype=np.float32)
        })
        
        self._idx_to_serve_on_next_reset = 0
        self._active_idx = -1 
        self.current_lap = 0
        self.max_data_items = len(self.train_data)
        
        print(f"Environment initialized with {self.max_data_items} training items")
    
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
    
    def _get_observation(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.train_data):
            return None
        
        data_item = self.train_data[idx]

        # Construct context features
        subject_encoding = [1 if data_item['subject'] == s else 0 for s in ['4th Grade Mathematics', '4th Grade English', '4th Grade Science']]
        question_type_encoding = [1 if data_item['question_type'] == t else 0 for t in ['Calculation', 'Short Answer', 'Essay', 'Definition', 'Diagram Labeling']]
        historical_features = [
            data_item['student_historical_avg'] / 5.0,
            data_item['cohort_avg_score'] / 5.0
        ]
        
        # Concatenate all features into a single flat array
        context_features = np.concatenate([
            np.array(subject_encoding, dtype=np.float32),
            np.array(question_type_encoding, dtype=np.float32),
            np.array(historical_features, dtype=np.float32)
        ])
        
        assert context_features.shape == (self.context_dim,), f"Context features shape mismatch: {context_features.shape}"

        # Get embeddings for the current event
        embeddings = self.embedding_df[self.embedding_df['event_id'] == data_item['event_id']]
        
        if len(embeddings) == 0:
            # If no embeddings available, use zeros
            return {
                'question_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
                'response_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
                'answer_embedding': np.zeros(self.embedding_dim, dtype=np.float32),
                'context_features': context_features
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
            'context_features': context_features
        }
    
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

    def get_active_idx(self) -> int:
        return self._active_idx

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
        if not (0 <= self._active_idx < self.max_data_items):
            raise ValueError(f"Step called with invalid _active_idx: {self._active_idx}")

        current_data = self.train_data[self._active_idx]
        
        # Validate action
        assert 0 <= action['score'] <= 5, "Invalid score"
        assert 0 <= action['feedback_id'] <= 10, "Invalid feedback ID"
        
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
        feedback_correct = (pred_feedback == true_feedback)
        if feedback_correct:
            feedback_reward = 3.0  # Reward for correct feedback
        else:
            feedback_reward = -3.0 # Strong penalty for incorrect feedback
        
        # Consistency bonus (placeholder)
        consistency_bonus = self.check_consistency(pred_score, pred_feedback)
        
        reward = score_reward + feedback_reward + consistency_bonus
        
        done = True # Episode is always one step for this environment structure

        # Get the next observation for the replay buffer
        next_obs_for_buffer = self._get_observation(self._idx_to_serve_on_next_reset)
        
        info = {
            'active_episode_idx': self._active_idx,
            'score_error': score_error,
            'feedback_correct': feedback_correct,
            'true_score': true_score,
            'pred_score': pred_score,
            'true_feedback': true_feedback,
            'pred_feedback': pred_feedback,
            'score_reward': score_reward,
            'feedback_reward': feedback_reward,
            'consistency_bonus': consistency_bonus
        }
        
        return next_obs_for_buffer, reward, done, info
    
    def reset(self) -> Dict[str, np.ndarray]:
        self._active_idx = self._idx_to_serve_on_next_reset
        observation = self._get_observation(self._active_idx)
        
        if self._idx_to_serve_on_next_reset == self.max_data_items - 1:
            self._idx_to_serve_on_next_reset = 0
            self.current_lap += 1
        else:
            self._idx_to_serve_on_next_reset += 1
            
        return observation
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the current state of the environment."""
        if not (0 <= self._active_idx < self.max_data_items):
            print("Cannot render, _active_idx is invalid.")
            return None
        if mode == 'human':
            current_data = self.train_data[self._active_idx]
            print(f"\nEpisode (Data Index) {self._active_idx + 1}/{self.max_data_items}")
            print(f"Subject: {current_data['subject']}")
            print(f"Question: {current_data['question_text']}")
            print(f"Student Answer: {current_data['student_answer_text']}")
            print(f"Expert Score: {current_data['expert_score']}")
            print(f"Expert Feedback ID: {current_data['expert_feedback_comment_id']}")
        return None 