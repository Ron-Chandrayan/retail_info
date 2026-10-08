import pandas as pd
import numpy as np
from pathlib import Path
from .config import OUTPUT_DIR, RANDOM_SEED

SENTIMENT_SEED = RANDOM_SEED
np.random.seed(SENTIMENT_SEED)

def generate_synthetic_sentiment(user_features_path: Path = OUTPUT_DIR / "user_features.csv",
                                 output_path: Path = OUTPUT_DIR / "synthetic_sentiment.csv") -> None:
    """Generate a deterministic synthetic sentiment proxy for each user.

    The sentiment score is based on behavioral signals such as purchase
    frequency, cart conversion, and recency.  Higher engagement yields a
    higher (more positive) sentiment.
    """
    df = pd.read_csv(user_features_path)
    # Basic scoring – weighted sum of normalized signals
    # Normalize each component to [0, 1]
    def min_max_norm(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    score = (
        0.4 * min_max_norm(df["purchase_frequency"])
        + 0.3 * min_max_norm(df["cart_conversion_rate"])
        + 0.2 * (1 - min_max_norm(df["recency_days"]))  # recent -> higher
        + 0.1 * min_max_norm(df["interaction_frequency"])
    )
    # Scale to [-1, 1]
    score = 2 * score - 1
    df_sent = pd.DataFrame({
        "visitorid": df["visitorid"],
        "sentiment_score": score,
        "sentiment_label": pd.cut(score, bins=[-float('inf'), -0.33, 0.33, float('inf')],
                                    labels=["negative", "neutral", "positive"]),
        "sentiment_source": "synthetic_behavioral_proxy",
    })
    df_sent.to_csv(output_path, index=False)

if __name__ == "__main__":
    generate_synthetic_sentiment()
