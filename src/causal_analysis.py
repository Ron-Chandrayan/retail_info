import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .config import OUTPUT_DIR

def run_causal_analysis():
    """Perform a simple causal analysis using the generated recommendation logs.

    The analysis treats the randomly assigned "treatment" (personalized recommendation)
    versus "control" (baseline) as the experimental condition.  The outcome is the
    binary ``engagement_outcome`` flag (clicked, add‑to‑cart or transaction).
    """
    logs_path = OUTPUT_DIR / "recommendation_logs.csv"
    if not logs_path.is_file():
        print("Recommendation logs not found. Run the recommendation log generation step first.")
        return
    df = pd.read_csv(logs_path)
    # Ensure the treatment_control column exists
    if "treatment_control" not in df.columns:
        print("Column 'treatment_control' missing in logs. Cannot perform causal analysis.")
        return
    # Compute means per group
    treatment = df[df["treatment_control"] == "treatment"]
    control = df[df["treatment_control"] == "control"]
    treat_rate = treatment["engagement_outcome"].mean()
    control_rate = control["engagement_outcome"].mean()
    ate = treat_rate - control_rate
    uplift = (ate / control_rate) if control_rate != 0 else np.nan
    # Sample sizes
    n_treat = len(treatment)
    n_control = len(control)
    # Save results
    results = pd.DataFrame({
        "metric": ["treatment_rate", "control_rate", "ATE", "uplift", "n_treatment", "n_control"],
        "value": [treat_rate, control_rate, ate, uplift, n_treat, n_control]
    })
    results_path = OUTPUT_DIR / "causal_results.csv"
    results.to_csv(results_path, index=False)
    # Visualisation
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Treatment", "Control"], [treat_rate, control_rate], color=["steelblue", "orange"])
    ax.set_ylabel("Engagement Outcome Rate")
    ax.set_title("Treatment vs Control Engagement Rates")
    plt.tight_layout()
    viz_path = OUTPUT_DIR / "causal_effect.png"
    plt.savefig(viz_path)
    plt.close()
    print(f"Causal analysis results saved to {results_path}")
    print(f"Causal effect visualization saved to {viz_path}")

if __name__ == "__main__":
    run_causal_analysis()
