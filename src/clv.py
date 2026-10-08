import pandas as pd
import numpy as np
from pathlib import Path
from .config import OUTPUT_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def compute_clv_proxy(user_features_path: Path = OUTPUT_DIR / "user_features.csv",
                     output_path: Path = OUTPUT_DIR / "customer_value.csv") -> None:
    """Compute a deterministic CLV proxy for each user.

    Since monetary values are unavailable, the proxy combines:
    * transaction count (frequency of purchases)
    * recency (how recent the last transaction was)
    * interaction frequency (overall activity)
    The resulting score is normalized to [0, 1] and saved.
    """
    df = pd.read_csv(user_features_path)
    # Basic components
    trans_count = df["transaction"].astype(float)
    recency = df["recency_days"].astype(float)
    freq = df["interaction_frequency"].astype(float)
    # Normalize components
    def min_max(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)
    # Recency: more recent -> higher score, invert normalized recency
    recency_score = 1 - min_max(recency)
    clv_score = 0.5 * min_max(trans_count) + 0.3 * recency_score + 0.2 * min_max(freq)
    df_clv = pd.DataFrame({
        "visitorid": df["visitorid"],
        "CLV_proxy": clv_score,
        "clv_source": "synthetic_behavioral_proxy",
    })
    df_clv.to_csv(output_path, index=False)

if __name__ == "__main__":
    compute_clv_proxy()
