# MARL Educational Assessment System

A Multi-Agent Reinforcement Learning system for educational assessment using PyTorch.

## Project Structure
```
project/
├── src/
│   ├── agents/        # RL agents implementation
│   ├── environment/   # Gym environment
│   ├── utils/         # Utility functions
│   └── training/      # Training scripts
├── data/             # Dataset files
└── scripts/          # Utility scripts
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Training Agent 1 (Grading & Feedback Agent):
```bash
python src/training/train_agent1.py
```

## Data Files
- `synthetic_data_train_v2_checkpoint.json`: Training data
- `embedded_dataset_balanced_v2.pkl`: Embedded dataset for training 