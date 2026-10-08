import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime, timezone

from .config import OUTPUT_DIR, RANDOM_SEED

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def generate_recommendation_logs():
    """Generate synthetic recommendation logs based on the generated recommendations.

    Each recommendation is assigned to either a treatment (personalized) or control (baseline)
    group. Outcomes (clicked, add_to_cart, subsequent_transaction) are simulated with
    deterministic random seeds for reproducibility.
    """
    rec_path = OUTPUT_DIR / "recommendations.csv"
    if not rec_path.is_file():
        print("Recommendations not found. Run dynamic recommendation first.")
        return
    recs = pd.read_csv(rec_path)
    logs = []
    for _, row in recs.iterrows():
        # Randomly assign treatment (50% chance)
        treatment = random.random() < 0.5
        # Simulate outcome probabilities based on rank and treatment
        rank = row["recommendation_rank"]
        base_click_prob = 0.3 / rank  # higher rank more likely
        click_prob = base_click_prob * (1.2 if treatment else 0.9)
        clicked = random.random() < min(click_prob, 1.0)
        # Add-to-cart probability only if clicked
        addtocart = False
        transaction = False
        if clicked:
            addtocart = random.random() < 0.4
            if addtocart:
                transaction = random.random() < 0.3
        log = {
            "visitorid": row["visitorid"],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "recommended_item": row["itemid"],
            "recommendation_rank": rank,
            "recommendation_score": row["score"],
            "segment": row["segment"],
            "treatment_control": "treatment" if treatment else "control",
            "clicked": int(clicked),
            "add_to_cart": int(addtocart),
            "subsequent_transaction": int(transaction),
            "engagement_outcome": int(transaction or addtocart or clicked),
            "log_source": "generated_prototype_recommendation_logs",
        }
        logs.append(log)
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(OUTPUT_DIR / "recommendation_logs.csv", index=False)
    print(f"Generated {len(logs_df)} recommendation log entries.")

if __name__ == "__main__":
    generate_recommendation_logs()
