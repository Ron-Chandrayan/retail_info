import pandas as pd
import numpy as np
from pathlib import Path
from .config import (
    DATA_DIR,
    OUTPUT_DIR,
    ITEM_EMBEDDING_DIM,
    ITEM_CONTEXT_WINDOW,
    MAX_ITEM_VOCAB,
    ITEM_EPOCHS,
    RANDOM_SEED,
)


def _build_interaction_sequences(events_path: Path) -> dict:
    """Build per-user ordered list of item ids from events.

    Returns:
        dict: visitorid -> list of itemid (as strings) ordered by timestamp.
    """
    df = pd.read_csv(events_path)
    # Ensure required columns exist
    required = {"timestamp", "visitorid", "itemid", "event"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"events.csv missing columns: {missing}")
    # Convert timestamp to int for sorting
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "visitorid", "itemid"])
    df["visitorid"] = df["visitorid"].astype(str)
    df["itemid"] = df["itemid"].astype(str)
    # Sort chronologically
    df = df.sort_values("timestamp")
    # Build sequences
    sequences = {}
    for visitor, group in df.groupby("visitorid"):
        sequences[visitor] = group["itemid"].tolist()
    return sequences


def _generate_skipgram_pairs(sequences: dict):
    """Yield (target, context) pairs for skip‑gram training.

    Uses a sliding window of size ITEM_CONTEXT_WINDOW.
    """
    for items in sequences.values():
        n = len(items)
        for idx, target in enumerate(items):
            start = max(0, idx - ITEM_CONTEXT_WINDOW)
            end = min(n, idx + ITEM_CONTEXT_WINDOW + 1)
            for ctx_idx in range(start, end):
                if ctx_idx == idx:
                    continue
                yield target, items[ctx_idx]


def train_item_embeddings():
    """Train a lightweight skip‑gram style item embedding.

    The model is implemented using NumPy for simplicity and runs for a few epochs.
    The resulting embedding matrix is saved to ``outputs/item_embeddings.csv`` with
    columns ``itemid, emb_0, ..., emb_{DIM-1}``.
    """
    events_path = DATA_DIR / "events.csv"
    sequences = _build_interaction_sequences(events_path)

    # Build vocabulary limited by MAX_ITEM_VOCAB
    all_items = set(item for seq in sequences.values() for item in seq)
    vocab_items = list(all_items)[:MAX_ITEM_VOCAB]
    item_to_idx = {item: i for i, item in enumerate(vocab_items)}
    vocab_size = len(vocab_items)

    # Initialize embeddings randomly
    rng = np.random.default_rng(RANDOM_SEED)
    embed = rng.normal(scale=0.01, size=(vocab_size, ITEM_EMBEDDING_DIM))
    # Simple SGD parameters
    lr = 0.05
    for epoch in range(ITEM_EPOCHS):
        total_loss = 0.0
        for target, context in _generate_skipgram_pairs(sequences):
            if target not in item_to_idx or context not in item_to_idx:
                continue
            t_idx = item_to_idx[target]
            c_idx = item_to_idx[context]
            # Positive pair dot product
            score = embed[t_idx].dot(embed[c_idx])
            # Sigmoid
            prob = 1 / (1 + np.exp(-score))
            # Binary cross‑entropy loss (positive pair)
            loss = -np.log(prob + 1e-8)
            total_loss += loss
            # Gradient ascent on positive pair
            grad = (1 - prob)
            embed[t_idx] += lr * grad * embed[c_idx]
            embed[c_idx] += lr * grad * embed[t_idx]
        print(f"Epoch {epoch+1}/{ITEM_EPOCHS}, loss: {total_loss:.4f}")

    # Save to CSV
    df_out = pd.DataFrame(embed, index=vocab_items)
    df_out.index.name = "itemid"
    df_out.columns = [f"emb_{i}" for i in range(ITEM_EMBEDDING_DIM)]
    output_path = OUTPUT_DIR / "item_embeddings.csv"
    df_out.to_csv(output_path)
    print(f"Saved item embeddings to {output_path}")

if __name__ == "__main__":
    train_item_embeddings()
