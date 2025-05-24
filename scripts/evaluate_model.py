import argparse
import json
import os
import torch
import numpy as np
from typing import Dict, List, Tuple
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.assessment_env import AssessmentEnv
from src.agents.grading_agent import GradingAgent

def evaluate_agent_on_env(agent: GradingAgent, env: AssessmentEnv, num_episodes: int) -> Dict:
    """Evaluate the agent's performance on a given environment."""
    agent.eval()  # Set the agent to evaluation mode
    total_reward = 0
    score_errors = []
    total_eval_steps = 0
    
    all_true_scores: List[int] = []
    all_pred_scores: List[int] = []
    all_true_feedback_ids: List[int] = []
    all_pred_feedback_ids: List[int] = []
    
    print(f"Evaluating for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset(episode_idx=episode) # Allow specific episode reset for full coverage
        done = False
        episode_reward = 0
        
        # It's important that env.reset(episode_idx) correctly sets the environment
        # to the specified episode to ensure we iterate through the entire eval set once.
        # If env.reset() always goes to 0, and step increments current_episode,
        # we need to ensure we are getting the correct true labels.

        if (episode + 1) % 10 == 0 or episode == num_episodes -1 : # Print progress
            print(f"  Evaluating episode {episode + 1}/{num_episodes}")

        while not done:
            if state is None: 
                print(f"Warning: current state is None for episode {episode}, breaking inner loop.")
                # This might happen if an episode in eval_data is malformed or env has issues
                # If it's the first step, env.reset() might be the issue or the data itself.
                if total_eval_steps == 0 and episode_reward == 0: # Stuck at the start
                     # Try to get data for current episode to log, if possible
                    if episode < len(env.train_data):
                        current_data_for_true_label = env.train_data[episode]
                        all_true_scores.append(current_data_for_true_label['expert_score'])
                        all_pred_scores.append(-1) # Placeholder for error
                        all_true_feedback_ids.append(current_data_for_true_label['expert_feedback_comment_id'])
                        all_pred_feedback_ids.append(-1) # Placeholder for error
                        total_eval_steps +=1 # Count this as a step attempted
                break 
                
            state_tensor = agent._preprocess_observation(state)
            action = agent.select_action(state_tensor, training=False) # Greedy action
            
            # Get true labels for the *current* state before step advances the episode
            # Assuming env.current_episode points to the data for the *current* state
            # before env.step() is called and potentially increments it.
            # If env.step() loads the next item, then this current_episode is already for next.
            # The original AssessmentEnv increments current_episode in step *after* processing current.
            # So, env.train_data[env.current_episode] should be the data for the state we just got an action for.
            
            # Let's fetch true data directly based on the episode index for safety
            # This assumes AssessmentEnv loads data in order and reset(episode_idx) works.
            current_data_idx = env.current_episode # This should be the index for the current item
            if current_data_idx >= len(env.train_data):
                print(f"Warning: current_data_idx {current_data_idx} out of bounds for train_data len {len(env.train_data)}")
                break # Avoid error

            current_true_data = env.train_data[current_data_idx]
            
            all_true_scores.append(current_true_data['expert_score'])
            all_pred_scores.append(action['score'])
            all_true_feedback_ids.append(current_true_data['expert_feedback_comment_id'])
            all_pred_feedback_ids.append(action['feedback_id'])
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            score_errors.append(info['score_error'])
            total_eval_steps += 1
                
            if done:
                break
            state = next_state
            
        total_reward += episode_reward
    
    avg_reward = total_reward / num_episodes if num_episodes > 0 else 0
    avg_score_error = np.mean(score_errors) if score_errors else 0
    
    # Calculate accuracies
    score_accuracy = accuracy_score(all_true_scores, all_pred_scores) if all_true_scores else 0
    feedback_id_accuracy = accuracy_score(all_true_feedback_ids, all_pred_feedback_ids) if all_true_feedback_ids else 0

    return {
        'avg_reward': avg_reward,
        'avg_score_error': avg_score_error,
        'score_accuracy': score_accuracy,
        'feedback_id_accuracy': feedback_id_accuracy,
        'all_true_scores': all_true_scores,
        'all_pred_scores': all_pred_scores,
        'all_true_feedback_ids': all_true_feedback_ids,
        'all_pred_feedback_ids': all_pred_feedback_ids,
        'total_eval_steps': total_eval_steps
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained assessment agent.')
    parser.add_argument('--eval_data', type=str, 
                        default='../synthetic_data_eval_v2_checkpoint.json', 
                        help='Path to evaluation data JSON file. Defaults to ../synthetic_data_eval_v2_checkpoint.json')
    parser.add_argument('--eval_embeddings', type=str, 
                        default='../embedded_evaluation_dataset_balanced_v2.pkl', 
                        help='Path to evaluation embeddings pickle file. Defaults to ../embedded_evaluation_dataset_balanced_v2.pkl')
    parser.add_argument('--model_path', type=str, 
                        default='../checkpoints/trained_model.pt',
                        help='Path to the saved model checkpoint (.pt file). Defaults to ../checkpoints/trained_model.pt')
    parser.add_argument('--hidden_dim', type=int, default=512, 
                        help='Hidden dimension of the agent network (must match trained model).')
    # Add other agent-specific parameters if they are crucial for model loading and structure,
    # though ideally, most are handled by the agent's load method or are not critical for eval.
    
    args = parser.parse_args()

    print("--- Evaluation Script ---")
    print(f"Evaluation Data: {args.eval_data}")
    print(f"Evaluation Embeddings: {args.eval_embeddings}")
    print(f"Model Path: {args.model_path}")
    print(f"Agent Hidden Dimension: {args.hidden_dim}")

    print("\nLoading evaluation environment...")
    eval_env = AssessmentEnv(args.eval_data, args.eval_embeddings)
    print(f"Evaluation environment loaded. Number of available evaluation items: {eval_env.max_episodes}")

    print("\nInitializing agent...")
    agent = GradingAgent(
        hidden_dim=args.hidden_dim,
        learning_rate=0, # Not used for eval
        gamma=0, # Not used for eval
        epsilon_start=0, # Not used for eval
        epsilon_end=0, # Not used for eval
        epsilon_decay=0, # Not used for eval
        buffer_size=1, # Not used for eval
        batch_size=1, # Not used for eval
        target_update_freq=100000, # Not used for eval
        weight_decay=0 # Not used for eval, but optimizer init needs it
    )
    
    print(f"Loading model weights from {args.model_path}...")
    try:
        agent.load(args.model_path)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the --hidden_dim matches the one used for training the loaded model,")
        print("and the model file is a valid checkpoint.")
        return

    # Determine number of episodes for evaluation (e.g., all available in the eval set)
    num_eval_episodes = eval_env.max_episodes
    if num_eval_episodes == 0:
        print("Warning: No episodes found in the evaluation data. Exiting.")
        return

    print(f"\nStarting evaluation on {num_eval_episodes} items...")
    eval_metrics = evaluate_agent_on_env(agent, eval_env, num_episodes=num_eval_episodes)
    
    print("\n--- Evaluation Results ---")
    print(f"  Average Reward: {eval_metrics['avg_reward']:.3f}")
    print(f"  Average Score Error: {eval_metrics['avg_score_error']:.3f}")
    print(f"  Score Prediction Accuracy: {eval_metrics['score_accuracy']:.3f}")
    print(f"  Feedback ID Prediction Accuracy: {eval_metrics['feedback_id_accuracy']:.3f}")
    print(f"  Total Evaluation Steps: {eval_metrics['total_eval_steps']}")

    if eval_metrics['all_true_scores'] and eval_metrics['all_pred_scores']:
        print("\n--- Score Prediction Details ---")
        score_labels = sorted(list(set(eval_metrics['all_true_scores'] + eval_metrics['all_pred_scores'])))
        if -1 in score_labels: score_labels.remove(-1) # Remove placeholder if present
        
        # Ensure all expected labels (0-5) are present for confusion matrix, handle if some are missing
        expected_score_labels = list(range(agent.score_action_space.n)) # Assumes agent has this
        # If any predicted/true scores are outside this, confusion_matrix might behave unexpectedly
        # or classification_report might miss some.
        # For now, use labels derived from data, can be refined if agent has fixed score_space properties.

        print("Score Confusion Matrix:")
        # Ensure labels are present, otherwise confusion_matrix might have different dimensions
        unique_true_scores = sorted(list(set(eval_metrics['all_true_scores'])))
        unique_pred_scores = sorted(list(set(eval_metrics['all_pred_scores'])))
        all_present_score_labels = sorted(list(set(unique_true_scores + unique_pred_scores)))
        
        # Filter out placeholder -1 if it exists due to errors
        filtered_true_scores = [s for s in eval_metrics['all_true_scores'] if s != -1]
        filtered_pred_scores = [s for s in eval_metrics['all_pred_scores'] if s != -1]
        
        if filtered_true_scores and filtered_pred_scores: # Only print if we have valid data
            # Define labels for confusion matrix to ensure all classes 0-5 are shown
            cm_score_labels = list(range(eval_env.score_space.n)) # from AssessmentEnv definition
            
            score_conf_matrix = confusion_matrix(filtered_true_scores, filtered_pred_scores, labels=cm_score_labels)
            print(score_conf_matrix)
            print("\nScore Classification Report:")
            print(classification_report(filtered_true_scores, filtered_pred_scores, labels=cm_score_labels, zero_division=0))
        else:
            print("Not enough valid data for score confusion matrix or classification report.")

    if eval_metrics['all_true_feedback_ids'] and eval_metrics['all_pred_feedback_ids']:
        print("\n--- Feedback ID Prediction Details ---")
        # Similar label handling for feedback IDs
        expected_feedback_labels = list(range(agent.feedback_action_space.n))

        filtered_true_feedback = [f for f in eval_metrics['all_true_feedback_ids'] if f != -1]
        filtered_pred_feedback = [f for f in eval_metrics['all_pred_feedback_ids'] if f != -1]

        if filtered_true_feedback and filtered_pred_feedback:
            cm_feedback_labels = list(range(eval_env.feedback_space.n)) # from AssessmentEnv

            feedback_conf_matrix = confusion_matrix(filtered_true_feedback, filtered_pred_feedback, labels=cm_feedback_labels)
            print("Feedback ID Confusion Matrix:")
            print(feedback_conf_matrix)
            print("\nFeedback ID Classification Report:")
            print(classification_report(filtered_true_feedback, filtered_pred_feedback, labels=cm_feedback_labels, zero_division=0))
        else:
            print("Not enough valid data for feedback ID confusion matrix or classification report.")
            
    print("--- End of Evaluation ---")

if __name__ == '__main__':
    main() 