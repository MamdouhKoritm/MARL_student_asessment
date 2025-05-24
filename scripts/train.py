import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple
import sys
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.assessment_env import AssessmentEnv
from src.agents.grading_agent import GradingAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Train the assessment agent')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data JSON')
    parser.add_argument('--train_embeddings', type=str, required=True, help='Path to training embeddings pickle')
    parser.add_argument('--eval_data', type=str, required=True, help='Path to evaluation data JSON')
    parser.add_argument('--eval_embeddings', type=str, required=True, help='Path to evaluation embeddings pickle')
    parser.add_argument('--num_episodes', type=int, default=500, help='Total number of episodes to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--target_update', type=int, default=20, help='Target network update frequency')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--eval_frequency', type=int, default=25, help='Evaluate every N episodes')
    return parser.parse_args()

def evaluate_agent(agent: GradingAgent, env: AssessmentEnv, num_episodes: int = 100) -> Dict:
    """Evaluate the agent's performance."""
    agent.eval()
    total_reward = 0
    score_errors = []
    feedback_correct_count = 0
    total_eval_steps = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Convert state to tensor and ensure proper shape
            state_tensor = agent._preprocess_observation(state)  # Shape: (1, state_dim)
            
            # Get action
            action = agent.select_action(state_tensor, training=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Track metrics
            score_errors.append(info['score_error'])
            if info['feedback_correct']:
                feedback_correct_count += 1
            total_eval_steps += 1
                
            if done:
                break
                
            state = next_state
            
        total_reward += episode_reward
    
    agent.train()
    return {
        'avg_reward': total_reward / num_episodes,
        'avg_score_error': np.mean(score_errors),
        'feedback_accuracy': feedback_correct_count / total_eval_steps if total_eval_steps > 0 else 0
    }

def main():
    # Default configuration using distinct synthetic datasets
    config = {
        'train_data': '../synthetic_data_train_v2_checkpoint.json',       
        'train_embeddings': '../embedded_dataset_balanced_v2.pkl',      
        'num_episodes': 500,  
        'batch_size': 32,
        'learning_rate': 5e-6,  
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,  
        'target_update': 20,
        'checkpoint_dir': '../checkpoints',
        'hidden_dim': 512,  
        'weight_decay': 1e-5 
    }
    
    # Create training environment
    print("Loading training environment...")
    train_env = AssessmentEnv(config['train_data'], config['train_embeddings'])
    print(f"Training environment loaded. Max episodes in data: {train_env.max_episodes}")
    
    # Create agent
    agent = GradingAgent(
        hidden_dim=config['hidden_dim'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        buffer_size=10000,
        batch_size=config['batch_size'],
        target_update_freq=config['target_update'],
        weight_decay=config['weight_decay']
    )
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    print("\nStarting training...")
    print(f"Training for {config['num_episodes']} episodes")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epsilon: {config['epsilon_start']} -> {config['epsilon_end']} (decay: {config['epsilon_decay']})")
    
    start_time = time.time()

    for episode in range(config['num_episodes']):
        agent.train() 
        state = train_env.reset()
        episode_reward = 0
        episode_losses = [] # To store losses for the episode
        steps_in_episode = 0
        training_steps_in_episode = 0 # To count steps where training actually happened
        done = False
        
        while not done:
            state_tensor = agent._preprocess_observation(state)
            action = agent.select_action(state_tensor) 
            next_state, reward, done, info = train_env.step(action)
            
            if next_state is not None:
                next_state_tensor = agent._preprocess_observation(next_state)
            else:
                next_state_tensor = torch.zeros((1, agent.state_dim), device=agent.device)
            
            action_idx = agent.encode_action(action['score'], action['feedback_id'])
            agent.replay_buffer.push(
                state_tensor, action_idx, reward, next_state_tensor, done
            )
            
            current_loss = 0 # Default to 0 if no training step happens
            if len(agent.replay_buffer) > config['batch_size']:
                current_loss = agent.train_step() 
                episode_losses.append(current_loss)
                training_steps_in_episode += 1
                agent.update_epsilon()
            
            episode_reward += reward
            
            state = next_state
            steps_in_episode += 1
        
        # Calculate average loss for the episode
        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0.0

        # Timing and basic episode summary
        elapsed_time = time.time() - start_time
        average_time_per_episode = elapsed_time / (episode + 1)
        estimated_remaining_time = average_time_per_episode * (config['num_episodes'] - episode - 1)
        remaining_minutes = int(estimated_remaining_time // 60)
        remaining_seconds = int(estimated_remaining_time % 60)

        print(f"Episode {episode + 1}/{config['num_episodes']} completed. Reward: {episode_reward:.3f}. Est. time rem: {remaining_minutes}m {remaining_seconds}s")
        
        # Save checkpoint every 25 episodes
        if (episode + 1) % 25 == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'model_episode_{episode + 1}.pt')
            agent.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(config['checkpoint_dir'], 'trained_model.pt')
    agent.save(final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}")

if __name__ == '__main__':
    main() 