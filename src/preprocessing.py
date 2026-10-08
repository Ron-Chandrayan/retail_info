import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from .config import DATA_DIR, OUTPUT_DIR, MAX_USERS, RANDOM_SEED

def load_events(path: Path) -> pd.DataFrame:
    """Load events CSV and parse timestamps.

    Parameters
    ----------
    path: Path
        Path to ``events.csv``.
    Returns
    -------
    DataFrame
        Normalised event dataframe.
    """
    df = pd.read_csv(path)
    # Validate required columns
    required = {"timestamp", "visitorid", "event", "itemid", "transactionid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Convert timestamp (Unix ms) to UTC datetime
    df["event_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Sort chronologically
    df = df.sort_values("event_time").reset_index(drop=True)
    # Normalise event names (lowercase)
    df["event"] = df["event"].str.lower()
    # Fill missing transaction IDs with NaN
    df["transactionid"] = df["transactionid"].replace({"": np.nan})
    return df

def dataset_summary(df: pd.DataFrame) -> None:
    print("Dataset dimensions:", df.shape)
    print("Missing values per column:")
    print(df.isna().sum())

def create_user_features(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate per visitor
    agg = {
        "view": lambda x: (x == "view").sum(),
        "addtocart": lambda x: (x == "addtocart").sum(),
        "transaction": lambda x: (x == "transaction").sum(),
    }
    user_feat = df.groupby("visitorid")["event"].apply(lambda s: pd.Series({k: (s == k).sum() for k in ["view", "addtocart", "transaction"]})).reset_index()
    # Unique items viewed/purchased
    viewed = df[df["event"] == "view"].groupby("visitorid")["itemid"].nunique().rename("unique_items_viewed")
    purchased = df[df["event"] == "transaction"].groupby("visitorid")["itemid"].nunique().rename("unique_items_purchased")
    user_feat = user_feat.merge(viewed, on="visitorid", how="left").merge(purchased, on="visitorid", how="left")
    # Ensure required count columns exist (in case a user has no events of a certain type)
    for col in ["view", "addtocart", "transaction"]:
        if col not in user_feat.columns:
            user_feat[col] = 0
    # Derived rates
    user_feat["purchase_frequency"] = user_feat["transaction"] / (user_feat["view"] + 1e-6)
    user_feat["cart_conversion_rate"] = user_feat["addtocart"] / (user_feat["view"] + 1e-6)
    user_feat["purchase_conversion_rate"] = user_feat["transaction"] / (user_feat["addtocart"] + 1e-6)
    # Recency (days since last event)
    last_event = df.groupby("visitorid")["event_time"].max()
    now = pd.Timestamp.now(tz=timezone.utc)
    user_feat["recency_days"] = (now - last_event).dt.days
    # Interaction frequency (events per day)
    total_events = df.groupby("visitorid").size().rename("total_events")
    days_active = (df.groupby("visitorid")["event_time"].max() - df.groupby("visitorid")["event_time"].min()).dt.days.replace(0, 1)
    user_feat = user_feat.merge(total_events, on="visitorid", how="left")
    user_feat["interaction_frequency"] = user_feat["total_events"] / days_active
    return user_feat

def save_outputs(df_events: pd.DataFrame, df_user_feat: pd.DataFrame) -> None:
    # Processed events
    df_events.to_csv(OUTPUT_DIR / "processed_events.csv", index=False)
    # User features
    df_user_feat.to_csv(OUTPUT_DIR / "user_features.csv", index=False)
    # User sequences (list of events per user as strings)
    user_seq = df_events.groupby("visitorid").apply(lambda g: " ".join(g["event"] + "_item_" + g["itemid"].astype(str)))
    user_seq = user_seq.reset_index().rename(columns={0: "event_sequence"})
    user_seq.to_csv(OUTPUT_DIR / "user_sequences.csv", index=False)
    # Item statistics
    item_stats = df_events.groupby("itemid")["event"].value_counts().unstack(fill_value=0)
    item_stats.to_csv(OUTPUT_DIR / "item_statistics.csv")

def preprocess_dataset(events_path: Path = DATA_DIR / "events.csv") -> None:
    if not events_path.is_file():
        print("RetailRocket events.csv not found. Place the dataset inside data/events.csv before running the complete pipeline.")
        return
    df = load_events(events_path)
    dataset_summary(df)
    # Optionally limit users for quick dev runs
    if MAX_USERS is not None:
        sampled_visitors = df["visitorid"].drop_duplicates().sample(n=min(MAX_USERS, df["visitorid"].nunique()), random_state=RANDOM_SEED)
        df = df[df["visitorid"].isin(sampled_visitors)]
    user_feat = create_user_features(df)
    save_outputs(df, user_feat)

if __name__ == "__main__":
    preprocess_dataset()
