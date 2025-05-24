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
    
    # Determine project root assuming train.py is in a subdirectory e.g., scripts/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    default_train_data = os.path.join(project_root, "synthetic_data_train_v2_checkpoint.json")
    default_train_embeddings = os.path.join(project_root, "embedded_dataset_balanced_v2.pkl")
    default_eval_data = os.path.join(project_root, "synthetic_data_eval_v2_checkpoint.json")
    default_eval_embeddings = os.path.join(project_root, "embedded_evaluation_dataset_balanced_v2.pkl")
    default_checkpoint_dir = os.path.join(project_root, "checkpoints_v2")

    parser.add_argument('--train_data', type=str, default=default_train_data, help='Path to training data JSON')
    parser.add_argument('--train_embeddings', type=str, default=default_train_embeddings, help='Path to training embeddings pickle')
    parser.add_argument('--eval_data', type=str, default=default_eval_data, help='Path to evaluation data JSON')
    parser.add_argument('--eval_embeddings', type=str, default=default_eval_embeddings, help='Path to evaluation embeddings pickle')
    parser.add_argument('--num_episodes', type=int, default=700, help='Total number of episodes to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--target_update', type=int, default=20, help='Target network update frequency')
    parser.add_argument('--checkpoint_dir', type=str, default=default_checkpoint_dir, help='Directory to save checkpoints')
    parser.add_argument('--eval_frequency', type=int, default=25, help='Evaluate every N episodes')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of the agent network')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')
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
    args = parse_args() # Parse arguments from command line

    # Use parsed arguments to set up configuration
    # Paths from args are assumed to be correct as provided by user (e.g., absolute or correctly relative)
    
    # Create training environment
    print("Loading training environment...")
    # Make train_data and train_embeddings paths absolute from the script or make them CLI args too
    # For now, assuming they are correctly passed if overridden by CLI, or defaults are fine relative to execution
    # Let's make them command-line arguments for full control
    train_env = AssessmentEnv(args.train_data, args.train_embeddings)
    print(f"Training environment loaded. Max episodes in data: {train_env.max_episodes}")
    
    # Create agent
    agent = GradingAgent(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=10000, # Or make this an arg
        batch_size=args.batch_size,
        target_update_freq=args.target_update, # Renamed from target_update in parse_args to match agent
        weight_decay=args.weight_decay
    )
    
    # Determine checkpoint directory
    checkpoint_directory = args.checkpoint_dir # Directly use the required argument
    
    # Create checkpoint directory
    # Important: Ensure this path is what you intend. If it's relative, it's relative to CWD.
    os.makedirs(checkpoint_directory, exist_ok=True)
    print(f"Using checkpoint directory: {os.path.abspath(checkpoint_directory)}")
    
    print("\nStarting training...")
    print(f"Training for {args.num_episodes} episodes")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})")
    
    start_time = time.time()

    for episode in range(args.num_episodes):
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
            if len(agent.replay_buffer) > args.batch_size:
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
        estimated_remaining_time = average_time_per_episode * (args.num_episodes - episode - 1)
        remaining_minutes = int(estimated_remaining_time // 60)
        remaining_seconds = int(estimated_remaining_time % 60)

        print(f"Episode {episode + 1}/{args.num_episodes} completed. Reward: {episode_reward:.3f}. Est. time rem: {remaining_minutes}m {remaining_seconds}s")
        
        # Save checkpoint every X episodes (use args.eval_frequency or a dedicated save_frequency arg)
        save_frequency = args.eval_frequency # Assuming eval_frequency is also save frequency for checkpoints
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_directory, f'model_episode_{episode + 1}.pt')
            agent.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(checkpoint_directory, 'trained_model.pt')
    agent.save(final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}")

if __name__ == '__main__':
    main() 