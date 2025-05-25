import argparse
import json
import os
import torch
import numpy as np
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.assessment_env import AssessmentEnv
from src.agents.grading_agent import GradingAgent
from src.agents.bias_detector_agent import BiasDetectorAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Jointly Train Grading Agent (Agent 1) and Bias Detector Agent (Agent 2)')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Common args
    default_train_data = os.path.join(project_root, "enhanced_synthetic_data_train_v2_checkpoint.json")
    default_train_embeddings = os.path.join(project_root, "enhanced_train_embedded_dataset_balanced_v2.pkl")
    parser.add_argument('--train_data', type=str, default=default_train_data)
    parser.add_argument('--train_embeddings', type=str, default=default_train_embeddings)
    parser.add_argument('--num_episodes', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--bias_label_key', type=str, default='bias_label', help='Key for true bias label in data')
    parser.add_argument('--bias_penalty_agent1', type=float, default=2.0, help='Penalty for Agent 1 if its action is confirmed biased')
    parser.add_argument('--save_frequency', type=int, default=100, help="Frequency to save models")
    default_checkpoint_dir = os.path.join(project_root, "checkpoints_joint")
    parser.add_argument('--checkpoint_dir', type=str, default=default_checkpoint_dir)

    # Agent 1 (GradingAgent) specific args
    parser.add_argument('--hidden_dim_agent1', type=int, default=512)
    parser.add_argument('--learning_rate_agent1', type=float, default=5e-5)
    parser.add_argument('--gamma_agent1', type=float, default=0.99)
    parser.add_argument('--epsilon_start_agent1', type=float, default=1.0)
    parser.add_argument('--epsilon_end_agent1', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_agent1', type=float, default=0.995)
    parser.add_argument('--target_update_agent1', type=int, default=10)
    parser.add_argument('--buffer_size_agent1', type=int, default=50000)
    parser.add_argument('--weight_decay_agent1', type=float, default=1e-5)

    # Agent 2 (BiasDetectorAgent) specific args
    parser.add_argument('--hidden_dim_agent2', type=int, default=256)
    parser.add_argument('--learning_rate_agent2', type=float, default=1e-4)
    parser.add_argument('--gamma_agent2', type=float, default=0.99)
    parser.add_argument('--epsilon_start_agent2', type=float, default=1.0)
    parser.add_argument('--epsilon_end_agent2', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_agent2', type=float, default=0.995)
    parser.add_argument('--target_update_agent2', type=int, default=10)
    parser.add_argument('--buffer_size_agent2', type=int, default=50000)
    parser.add_argument('--weight_decay_agent2', type=float, default=1e-5)
    
    return parser.parse_args()

def calculate_agent2_reward(agent2_action_idx: int, true_bias_label: int) -> float:
    if agent2_action_idx == 1 and true_bias_label == 1: # True Positive
        return 1.0
    elif agent2_action_idx == 0 and true_bias_label == 0: # True Negative
        return 0.5
    elif agent2_action_idx == 1 and true_bias_label == 0: # False Positive
        return -0.75
    elif agent2_action_idx == 0 and true_bias_label == 1: # False Negative
        return -1.0
    return 0.0 # Should not happen

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and verify data
    print(f"\nLoading data from:")
    print(f"Training data: {args.train_data}")
    print(f"Embeddings: {args.train_embeddings}")
    
    env = AssessmentEnv(args.train_data, args.train_embeddings)
    
    # Print data statistics
    total_data = len(env.train_data)
    bias_count = sum(1 for item in env.train_data if item.get('bias_label', 0) == 1)
    print(f"\nData Statistics:")
    print(f"Total examples: {total_data}")
    print(f"Biased examples: {bias_count} ({bias_count/total_data*100:.2f}%)")
    print(f"Non-biased examples: {total_data-bias_count} ({(total_data-bias_count)/total_data*100:.2f}%)")
    
    # Verify embeddings
    print("\nVerifying embeddings...")
    sample_obs = env.reset()
    for key in ['question_embedding', 'response_embedding', 'answer_embedding']:
        if key in sample_obs:
            embedding = sample_obs[key]
            print(f"{key}: shape={embedding.shape}, dtype={embedding.dtype}")
            print(f"    range: [{embedding.min():.3f}, {embedding.max():.3f}], mean={embedding.mean():.3f}")
    
    # Initialize agents with proper batch processing
    print("\nInitializing agents...")
    agent1 = GradingAgent(
        hidden_dim=args.hidden_dim_agent1,
        learning_rate=args.learning_rate_agent1,
        gamma=args.gamma_agent1,
        epsilon_start=args.epsilon_start_agent1,
        epsilon_end=args.epsilon_end_agent1,
        epsilon_decay=args.epsilon_decay_agent1,
        buffer_size=args.buffer_size_agent1,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_agent1,
        weight_decay=args.weight_decay_agent1
    ).to(device)

    agent2 = BiasDetectorAgent(
        context_dim=env.context_dim,
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

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize metrics tracking
    metrics = {
        'agent1_rewards': [],
        'agent2_rewards': [],
        'score_errors': [],
        'feedback_accuracy': [],
        'bias_true_positives': 0,
        'bias_false_positives': 0,
        'bias_true_negatives': 0,
        'bias_false_negatives': 0,
        'training_steps': 0  # Add training steps counter
    }

    def normalize_reward(reward, scale_factor=5.0):
        return reward / scale_factor

    def update_metrics(metrics, score_error, feedback_correct, true_bias, pred_bias, a1_reward, a2_reward):
        metrics['score_errors'].append(score_error)
        metrics['feedback_accuracy'].append(1.0 if feedback_correct else 0.0)
        metrics['agent1_rewards'].append(a1_reward)
        metrics['agent2_rewards'].append(a2_reward)
        
        if true_bias == 1 and pred_bias == 1:
            metrics['bias_true_positives'] += 1
        elif true_bias == 0 and pred_bias == 1:
            metrics['bias_false_positives'] += 1
        elif true_bias == 0 and pred_bias == 0:
            metrics['bias_true_negatives'] += 1
        elif true_bias == 1 and pred_bias == 0:
            metrics['bias_false_negatives'] += 1

    def print_metrics(metrics, episode):
        window = 100  # Moving average window
        if len(metrics['agent1_rewards']) > 0:
            recent_a1_rewards = metrics['agent1_rewards'][-window:]
            recent_a2_rewards = metrics['agent2_rewards'][-window:]
            recent_score_errors = metrics['score_errors'][-window:]
            recent_feedback_acc = metrics['feedback_accuracy'][-window:]
            
            # Calculate bias detection metrics
            total_predictions = (metrics['bias_true_positives'] + metrics['bias_false_positives'] + 
                               metrics['bias_true_negatives'] + metrics['bias_false_negatives'])
            if total_predictions > 0:
                accuracy = (metrics['bias_true_positives'] + metrics['bias_true_negatives']) / total_predictions
                precision = (metrics['bias_true_positives'] / (metrics['bias_true_positives'] + metrics['bias_false_positives']) 
                           if (metrics['bias_true_positives'] + metrics['bias_false_positives']) > 0 else 0)
                recall = (metrics['bias_true_positives'] / (metrics['bias_true_positives'] + metrics['bias_false_negatives'])
                         if (metrics['bias_true_positives'] + metrics['bias_false_negatives']) > 0 else 0)
                
                print(f"\nEpisode {episode} Metrics (last {window} episodes):")
                print(f"A1 Avg Reward: {np.mean(recent_a1_rewards):.2f}")
                print(f"A2 Avg Reward: {np.mean(recent_a2_rewards):.2f}")
                print(f"Avg Score Error: {np.mean(recent_score_errors):.2f}")
                print(f"Feedback Accuracy: {np.mean(recent_feedback_acc):.2f}")
                print(f"Bias Detection - Acc: {accuracy:.2f}, Prec: {precision:.2f}, Rec: {recall:.2f}")

    print("\nStarting Joint Training...")
    start_time = time.time()
    
    # Training loop with proper batch processing
    for episode in range(args.num_episodes):
        agent1_obs_dict = env.reset()
        done = False
        total_reward_agent1 = 0
        episode_reward_agent2_total = 0
        
        while not done:
            # Process a batch of experiences
            metrics['training_steps'] += 1
            
            # --- Agent 1 Action --- 
            agent1_obs_tensor = agent1._preprocess_observation(agent1_obs_dict).to(device)
            current_agent1_action_dict = agent1.select_action(agent1_obs_tensor, training=True)
            agent1_action_idx_encoded = agent1.encode_action(current_agent1_action_dict['score'], current_agent1_action_dict['feedback_id'])

            # --- Agent 2 Bias Detection ---
            agent2_obs_dict = {
                'context_features': agent1_obs_dict['context_features'],
                'agent1_score': current_agent1_action_dict['score'],
                'agent1_feedback_id': current_agent1_action_dict['feedback_id']
            }
            agent2_obs_tensor = agent2._preprocess_observation(agent2_obs_dict).to(device)
            agent2_bias_prediction_idx = agent2.select_action(agent2_obs_tensor, training=True)

            # --- Environment Step ---
            next_agent1_obs_for_buffer, base_reward_agent1, done, info = env.step(current_agent1_action_dict)

            # --- Get True Bias Label ---
            current_data_item = env.train_data[env.get_active_idx()]
            true_bias_label = int(current_data_item.get(args.bias_label_key, 0))
            
            # --- Calculate Rewards ---
            reward_agent2 = calculate_agent2_reward(agent2_bias_prediction_idx, true_bias_label)
            
            bias_penalty_for_agent1 = 0.0
            if agent2_bias_prediction_idx == 1 and true_bias_label == 1:
                bias_penalty_for_agent1 = args.bias_penalty_agent1
            
            final_reward_agent1 = normalize_reward(base_reward_agent1 - bias_penalty_for_agent1)
            total_reward_agent1 += final_reward_agent1
            episode_reward_agent2_total += reward_agent2

            # Update metrics
            update_metrics(
                metrics,
                info['score_error'],
                info['feedback_correct'],
                true_bias_label,
                agent2_bias_prediction_idx,
                final_reward_agent1,
                reward_agent2
            )

            # --- Store and Train ---
            if next_agent1_obs_for_buffer is not None:
                next_agent1_obs_tensor = agent1._preprocess_observation(next_agent1_obs_for_buffer).to(device)
                with torch.no_grad():
                    next_step_agent1_action_dict = agent1.select_action(next_agent1_obs_tensor, training=False)
                next_agent2_obs_dict = {
                    'context_features': next_agent1_obs_for_buffer['context_features'],
                    'agent1_score': next_step_agent1_action_dict['score'],
                    'agent1_feedback_id': next_step_agent1_action_dict['feedback_id']
                }
                next_agent2_obs_tensor = agent2._preprocess_observation(next_agent2_obs_dict).to(device)
            else:
                next_agent1_obs_tensor = torch.zeros((1, agent1.state_dim), device=agent1.device)
                next_agent2_obs_tensor = torch.zeros((1, agent2.state_dim), device=agent2.device)

            agent1.replay_buffer.push(agent1_obs_tensor, agent1_action_idx_encoded, final_reward_agent1, next_agent1_obs_tensor, done)
            agent2.replay_buffer.push(agent2_obs_tensor, agent2_bias_prediction_idx, reward_agent2, next_agent2_obs_tensor, done)

            # Train on multiple batches
            if len(agent1.replay_buffer) > args.batch_size:
                for _ in range(4):  # Multiple training steps per environment step
                    agent1.train_step()
            if len(agent2.replay_buffer) > args.batch_size:
                for _ in range(4):  # Multiple training steps per environment step
                    agent2.train_step()

            agent1.update_epsilon()
            agent2.update_epsilon()
            
            agent1_obs_dict = next_agent1_obs_for_buffer

        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (episode + 1)) * (args.num_episodes - episode - 1)

        if (episode + 1) % 10 == 0:
            print_metrics(metrics, episode + 1)
            print(f"Time: {elapsed_time/60:.1f}m | Est. Remaining: {remaining_time/60:.1f}m")
            print(f"Training steps: {metrics['training_steps']}, Epsilon: {agent1.epsilon:.3f}")

        if (episode + 1) % args.save_frequency == 0:
            agent1_path = os.path.join(args.checkpoint_dir, f'agent1_episode_{episode+1}.pt')
            agent2_path = os.path.join(args.checkpoint_dir, f'agent2_episode_{episode+1}.pt')
            agent1.save(agent1_path)
            agent2.save(agent2_path)

    print("\nJoint Training Complete.")
    agent1.save(os.path.join(args.checkpoint_dir, 'agent1_final.pt'))
    agent2.save(os.path.join(args.checkpoint_dir, 'agent2_final.pt'))

if __name__ == '__main__':
    main() 