import argparse
import json
import os
import torch
import numpy as np
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.assessment_env import AssessmentEnv # Might need modification or a new env
from src.agents.grading_agent import GradingAgent
from src.agents.bias_detector_agent import BiasDetectorAgent
from src.agents.base_agent import ReplayBuffer # Agent 2 will have its own buffer

def parse_args():
    parser = argparse.ArgumentParser(description='Train Bias Detector Agent (Agent 2)')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    default_train_data = os.path.join(project_root, "synthetic_data_train_v2_checkpoint.json")
    default_train_embeddings = os.path.join(project_root, "embedded_dataset_balanced_v2.pkl")
    default_checkpoint_dir_agent1 = os.path.join(project_root, "checkpoints_v2") # Where Agent 1 models are
    default_checkpoint_dir_agent2 = os.path.join(project_root, "checkpoints_agent2") # For Agent 2
    
    parser.add_argument('--train_data', type=str, default=default_train_data)
    parser.add_argument('--train_embeddings', type=str, default=default_train_embeddings)
    parser.add_argument('--agent1_model_path', type=str, required=True, help='Path to the trained Agent 1 model (.pt file)')
    parser.add_argument('--agent1_hidden_dim', type=int, default=512, help='Hidden dim for Agent 1 (must match loaded model)')
    
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate_agent2', type=float, default=1e-4)
    parser.add_argument('--gamma_agent2', type=float, default=0.99)
    parser.add_argument('--epsilon_start_agent2', type=float, default=1.0)
    parser.add_argument('--epsilon_end_agent2', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_agent2', type=float, default=0.995)
    parser.add_argument('--target_update_agent2', type=int, default=10)
    parser.add_argument('--hidden_dim_agent2', type=int, default=128)
    parser.add_argument('--weight_decay_agent2', type=float, default=0.0)
    parser.add_argument('--buffer_size_agent2', type=int, default=10000)

    parser.add_argument('--checkpoint_dir_agent2', type=str, default=default_checkpoint_dir_agent2)
    parser.add_argument('--save_frequency', type=int, default=25)
    
    parser.add_argument('--bias_label_key', type=str, default='bias_label', help='The key in the training data JSON to find the true bias label (0 or 1).')
    # TODO: Add arguments for bias labels if they come from a separate file or have specific keys in main data

    return parser.parse_args()

# We might need a simplified evaluation for Agent 2 or a new one
# def evaluate_agent2(agent2: BiasDetectorAgent, env, agent1: GradingAgent, num_episodes: int = 50):
#     pass 

def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Environment
    # NOTE: The environment might need to be adapted or a wrapper created if Agent 2's interaction
    # changes how episodes terminate or how rewards are structured fundamentally.
    # For now, assume Agent 2's actions don't alter the primary flow of an "assessment event" from Agent 1's perspective.
    print("Loading environment...")
    env = AssessmentEnv(args.train_data, args.train_embeddings)
    print(f"Environment loaded. Max episodes in data: {env.max_episodes}")

    # Load pretrained Agent 1 (GradingAgent)
    print(f"Loading Agent 1 from: {args.agent1_model_path}")
    agent1 = GradingAgent(
        hidden_dim=args.agent1_hidden_dim, 
        # Other params like LR, gamma etc. are not needed for a frozen eval model,
        # but state_dim and action_dim are set internally by GradingAgent
    ).to(device)
    agent1.load(args.agent1_model_path)
    agent1.eval() # Set Agent 1 to evaluation mode (frozen, no exploration/training)
    print("Agent 1 loaded and set to eval mode.")

    # Initialize Agent 2 (BiasDetectorAgent)
    agent2 = BiasDetectorAgent(
        context_dim=env.context_dim, # Get from env
        agent1_score_actions=env.score_space.n,
        agent1_feedback_actions=env.feedback_space.n,
        hidden_dim=args.hidden_dim_agent2,
        learning_rate=args.learning_rate_agent2,
        gamma=args.gamma_agent2,
        epsilon_start=args.epsilon_start_agent2,
        epsilon_end=args.epsilon_end_agent2,
        epsilon_decay=args.epsilon_decay_agent2,
        buffer_size=args.buffer_size_agent2,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_agent2,
        weight_decay=args.weight_decay_agent2
    ).to(device)
    print("Agent 2 initialized.")

    os.makedirs(args.checkpoint_dir_agent2, exist_ok=True)
    print(f"Agent 2 checkpoints will be saved to: {args.checkpoint_dir_agent2}")

    print("\\nStarting Agent 2 training...")
    start_time = time.time()

    for episode in range(args.num_episodes):
        agent1_obs_dict = env.reset() # Observation for Agent 1
        
        # Store previous Agent 1 action for Agent 2's observation (1-step delay)
        # For the very first step, there's no "previous" Agent 1 action.
        # We need a sensible default, e.g., a neutral or common action.
        # Let's assume score 0, feedback_id 0 as a starting point.
        # These are raw integer values, not one-hot encoded yet.
        prev_agent1_action = {'score': 0, 'feedback_id': 0} 
        
        done = False
        episode_reward_agent2 = 0
        episode_losses_agent2 = []
        
        while not done:
            # Agent 1 processes the current state (frozen)
            agent1_obs_tensor = agent1._preprocess_observation(agent1_obs_dict).to(device)
            with torch.no_grad(): # Ensure no gradients for Agent 1
                # Agent 1 selects action (score, feedback_id dict)
                current_agent1_action_dict = agent1.select_action(agent1_obs_tensor, training=False) 

            # Agent 2's turn
            # Construct Agent 2's observation using current context_features and *previous* Agent 1's action
            agent2_obs_dict = {
                'context_features': agent1_obs_dict['context_features'], # Current context
                'agent1_score': prev_agent1_action['score'],
                'agent1_feedback_id': prev_agent1_action['feedback_id']
            }
            agent2_obs_tensor = agent2._preprocess_observation(agent2_obs_dict).to(device)
            
            # Agent 2 selects an action (e.g., 0 for no bias, 1 for bias)
            agent2_action_idx = agent2.select_action(agent2_obs_tensor, training=True) # Agent 2 is training

            # Environment step is based on Agent 1's action primarily
            # The environment returns next_state for Agent 1, and reward for Agent 1
            next_agent1_obs_dict, reward_agent1, done, info_agent1 = env.step(current_agent1_action_dict)

            # --- Reward for Agent 2 ---
            true_bias_label = 0 # Default to no bias
            current_data_item = env.train_data[env.current_episode] # env.current_episode is the index for the *current* step before advancing
            if args.bias_label_key in current_data_item:
                true_bias_label = int(current_data_item[args.bias_label_key])
            else:
                print(f"Warning: Bias label key '{args.bias_label_key}' not found in data for episode {env.current_episode}. Defaulting to 0.")

            if agent2_action_idx == 1 and true_bias_label == 1: # True Positive (Correct Detection)
                reward_agent2 = 1.0
            elif agent2_action_idx == 0 and true_bias_label == 0: # True Negative (Correct Rejection)
                reward_agent2 = 0.5 
            elif agent2_action_idx == 1 and true_bias_label == 0: # False Positive
                reward_agent2 = -0.75
            elif agent2_action_idx == 0 and true_bias_label == 1: # False Negative
                reward_agent2 = -1.0
            else: # Should not happen for binary actions/labels
                reward_agent2 = 0.0
                print(f"Warning: Unexpected agent2_action_idx ({agent2_action_idx}) or true_bias_label ({true_bias_label})")
            
            episode_reward_agent2 += reward_agent2

            # Store Agent 2's experience
            # Next state for Agent 2 would be based on next_agent1_obs_dict['context_features']
            # and current_agent1_action (which becomes the *previous* for the next step)
            if next_agent1_obs_dict is not None:
                next_agent2_obs_for_buffer = {
                    'context_features': next_agent1_obs_dict['context_features'],
                    'agent1_score': current_agent1_action_dict['score'],
                    'agent1_feedback_id': current_agent1_action_dict['feedback_id']
                }
                next_agent2_state_tensor = agent2._preprocess_observation(next_agent2_obs_for_buffer).to(device)
            else: # Terminal state
                # For terminal states, the next state for Q-learning is often zeros.
                # The actual content doesn't matter as much if `done` is true.
                next_agent2_state_tensor = torch.zeros((1, agent2.state_dim), device=agent2.device)

            agent2.replay_buffer.push(
                agent2_obs_tensor, 
                agent2_action_idx, 
                reward_agent2, 
                next_agent2_state_tensor, 
                done 
            )

            # Train Agent 2
            if len(agent2.replay_buffer) > args.batch_size:
                loss_agent2 = agent2.train_step()
                if loss_agent2 is not None: # train_step might return None if not enough samples
                    episode_losses_agent2.append(loss_agent2)
                agent2.update_epsilon() # Decay epsilon for Agent 2

            # Update state for Agent 1's next iteration
            agent1_obs_dict = next_agent1_obs_dict
            # Update prev_agent1_action for Agent 2's next iteration's observation
            prev_agent1_action = current_agent1_action_dict
            
            if done:
                break
        
        avg_loss_agent2 = np.mean(episode_losses_agent2) if episode_losses_agent2 else 0.0
        elapsed_time = time.time() - start_time
        print(f"Episode {episode + 1}/{args.num_episodes} | A2 Reward: {episode_reward_agent2:.2f} | A2 Avg Loss: {avg_loss_agent2:.4f} | Epsilon A2: {agent2.epsilon:.3f} | Time: {elapsed_time:.2f}s")

        if (episode + 1) % args.save_frequency == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir_agent2, f'agent2_episode_{episode + 1}.pt')
            agent2.save(checkpoint_path)

    final_model_path = os.path.join(args.checkpoint_dir_agent2, 'agent2_trained_final.pt')
    agent2.save(final_model_path)
    print(f"\\nAgent 2 training complete. Final model saved to {final_model_path}")

if __name__ == '__main__':
    main() 