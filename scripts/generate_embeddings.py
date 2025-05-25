import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_data(data_path: str) -> list:
    """Load the training data from JSON file."""
    print(f"Loading data from {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def create_embeddings(model, texts: list, batch_size: int = 32) -> np.ndarray:
    """Create embeddings for a list of texts using batched processing."""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        # Remove empty strings and replace with a space to avoid model errors
        batch = [text if text.strip() else " " for text in batch]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings = torch.cat(embeddings)
    return embeddings.cpu().numpy()

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    input_file = os.path.join(project_root, "enhanced_synthetic_data_train_v2_checkpoint.json")
    output_file = os.path.join(project_root, "embedded_dataset_balanced_v2.pkl")
    
    # Load the model
    print("Loading BERT model...")
    model = SentenceTransformer('all-mpnet-base-v2')  # Using a powerful model for high-quality embeddings
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data = load_data(input_file)
    print(f"Loaded {len(data)} items")
    
    # Extract texts to embed
    questions = [item['question_text'] for item in data]
    rubrics = [item['rubric_section'] for item in data]
    answers = [item['student_answer_text'] for item in data]
    
    # Generate embeddings
    print("\nGenerating question embeddings...")
    question_embeddings = create_embeddings(model, questions)
    
    print("\nGenerating rubric embeddings...")
    rubric_embeddings = create_embeddings(model, rubrics)
    
    print("\nGenerating answer embeddings...")
    answer_embeddings = create_embeddings(model, answers)
    
    # Create DataFrame with embeddings and event_ids
    print("\nCreating DataFrame...")
    df = pd.DataFrame({
        'event_id': [item['event_id'] for item in data],
        'embedding_question': list(question_embeddings),
        'embedding_rubric': list(rubric_embeddings),
        'embedding_answer': list(answer_embeddings)
    })
    
    # Save to pickle file
    print(f"\nSaving embeddings to {output_file}")
    df.to_pickle(output_file)
    
    # Print some statistics
    print("\nEmbedding Statistics:")
    print(f"Number of items: {len(df)}")
    print(f"Embedding dimensions: {question_embeddings.shape[1]}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")

if __name__ == '__main__':
    main() 