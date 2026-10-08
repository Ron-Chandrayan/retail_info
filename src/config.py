from pathlib import Path
import random
import numpy as np

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Ensure output directories exist
for d in [OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configuration parameters
MAX_USERS = 5000  # limit for development; set to None to use all users
MAX_SEQUENCE_LENGTH = 25  # truncate/pad sequences to this length
EMBEDDING_DIM = 32
N_CLUSTERS = 4
TOP_K = 5
RANDOM_SEED = 42

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# PyTorch seed will be set inside model modules

# Recommendation weighting constants
USER_ITEM_WEIGHT = 0.65
POPULARITY_WEIGHT = 0.20
SEGMENT_WEIGHT = 0.10
RECENCY_WEIGHT = 0.05

# Recency decay (in days) for exponential decay of interaction freshness
RECENCY_DECAY_DAYS = 7

# Item embedding training parameters
ITEM_EMBEDDING_DIM = 32
ITEM_CONTEXT_WINDOW = 3
MAX_ITEM_VOCAB = 50000
ITEM_EPOCHS = 2
