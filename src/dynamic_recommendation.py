
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    OUTPUT_DIR,
    TOP_K,
    RANDOM_SEED,
    EMBEDDING_DIM,
    USER_ITEM_WEIGHT,
    POPULARITY_WEIGHT,
    SEGMENT_WEIGHT,
    RECENCY_WEIGHT,
    ITEM_EMBEDDING_DIM,
    RECENCY_DECAY_DAYS,
)

np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_user_embeddings() -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_DIR / "user_embeddings.csv"
    )


def load_segments() -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_DIR / "customer_segments.csv"
    )


def load_item_stats() -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_DIR / "item_statistics.csv"
    )


def load_user_features() -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_DIR / "user_features.csv"
    )


def load_clv() -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT_DIR / "customer_value.csv"
    )


# ============================================================
# ID CLEANING
# ============================================================

def clean_visitor_ids(series: pd.Series) -> pd.Series:
    """
    Convert visitor IDs to strings and clean values such as:

        tensor(478) -> 478
    """
    return (
        series
        .astype(str)
        .str.replace(
            r"tensor\((\d+)\)",
            r"\1",
            regex=True
        )
    )


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations() -> None:

    total_start = time.time()

    print("\n--- Generating dynamic recommendations ---")

    # ========================================================
    # LOAD DATA
    # ========================================================

    load_start = time.time()

    print("Loading recommendation data...")

    emb_df = load_user_embeddings()
    seg_df = load_segments()
    item_stats = load_item_stats()
    user_feat = load_user_features()
    clv = load_clv()

    print(
        f"Recommendation data loaded in "
        f"{time.time() - load_start:.2f}s"
    )

    # ========================================================
    # MAKE VISITOR IDs CONSISTENT
    # ========================================================

    emb_df["visitorid"] = clean_visitor_ids(
        emb_df["visitorid"]
    )

    seg_df["visitorid"] = clean_visitor_ids(
        seg_df["visitorid"]
    )

    user_feat["visitorid"] = clean_visitor_ids(
        user_feat["visitorid"]
    )

    clv["visitorid"] = clean_visitor_ids(
        clv["visitorid"]
    )

    # ========================================================
    # IMPORTANT:
    # ENSURE USER-LEVEL FILES HAVE ONE ROW PER USER
    # ========================================================

    print("Checking user-level data...")

    print(
        f"Embedding users: {len(emb_df):,} "
        f"(unique: {emb_df['visitorid'].nunique():,})"
    )

    print(
        f"Segment rows: {len(seg_df):,} "
        f"(unique users: {seg_df['visitorid'].nunique():,})"
    )

    print(
        f"User feature rows: {len(user_feat):,} "
        f"(unique users: {user_feat['visitorid'].nunique():,})"
    )

    print(
        f"CLV rows: {len(clv):,} "
        f"(unique users: {clv['visitorid'].nunique():,})"
    )

    # Keep exactly one row per visitor.
    #
    # This prevents many-to-many merges from turning
    # 5,000 users into 45,000 rows.

    seg_df = (
        seg_df
        .drop_duplicates(
            subset=["visitorid"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    user_feat = (
        user_feat
        .drop_duplicates(
            subset=["visitorid"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    clv = (
        clv
        .drop_duplicates(
            subset=["visitorid"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # Also protect the embeddings file itself.
    emb_df = (
        emb_df
        .drop_duplicates(
            subset=["visitorid"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # MERGE EMBEDDINGS + SEGMENTS
    # ========================================================

    df = emb_df.merge(
        seg_df,
        on="visitorid",
        how="left",
        validate="one_to_one"
    )

    # ========================================================
    # MERGE USER FEATURES
    # ========================================================

    df = df.merge(
        user_feat,
        on="visitorid",
        how="left",
        validate="one_to_one"
    )

    # ========================================================
    # ADD CLV
    # ========================================================

    if "CLV_proxy" not in df.columns:

        df = df.merge(
            clv[["visitorid", "CLV_proxy"]],
            on="visitorid",
            how="left",
            validate="one_to_one"
        )

    # ========================================================
    # VERIFY USER COUNT
    # ========================================================

    print(
        f"Users after merging recommendation data: "
        f"{len(df):,}"
    )

    print(
        f"Unique users after merging: "
        f"{df['visitorid'].nunique():,}"
    )

    # The recommender is based on user embeddings.
    # Therefore, the merged dataset must not contain
    # duplicate visitor IDs.

    if len(df) != df["visitorid"].nunique():
        raise ValueError(
            "Duplicate visitor IDs detected after merging "
            "recommendation data."
        )

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    embedding_cols = [
        f"emb_{i}"
        for i in range(EMBEDDING_DIM)
    ]

    required_columns = (
        embedding_cols
        + ["visitorid", "segment"]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # ========================================================
    # HANDLE MISSING VALUES
    # ========================================================

    df[embedding_cols] = (
        df[embedding_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    df["segment"] = (
        pd.to_numeric(
            df["segment"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    if "CLV_proxy" in df.columns:
        df["CLV_proxy"] = (
            pd.to_numeric(
                df["CLV_proxy"],
                errors="coerce"
            )
            .fillna(0)
        )

    # ========================================================
    # ITEM POPULARITY
    # ========================================================

    print("Preparing item popularity...")

    item_stats = item_stats.copy()

    item_stats["itemid"] = (
        item_stats["itemid"]
        .astype(str)
    )

    if "view" not in item_stats.columns:
        raise ValueError(
            "item_statistics.csv does not contain "
            "a 'view' column."
        )

    item_pop = (
        item_stats
        .set_index("itemid")["view"]
        .fillna(0)
    )

    item_pop.index = (
        item_pop.index.astype(str)
    )

    # --------------------------------------------------------
    # Top popular items
    # --------------------------------------------------------

    top_items_global = (
        item_pop
        .sort_values(
            ascending=False
        )
        .head(500)
        .index
        .tolist()
    )

    if not top_items_global:
        raise ValueError(
            "No items available for recommendation."
        )

    # ========================================================
    # ITEM EMBEDDINGS
    # ========================================================

    print("Loading item embeddings...")

    item_emb_path = (
        OUTPUT_DIR /
        "item_embeddings.csv"
    )

    if not item_emb_path.exists():
        raise FileNotFoundError(
            f"Item embeddings file not found at "
            f"{item_emb_path}. "
            f"Run item_embeddings.train_item_embeddings() first."
        )

    item_embeddings = pd.read_csv(
        item_emb_path,
        index_col="itemid"
    )

    item_embeddings.index = (
        item_embeddings.index.astype(str)
    )

    embedding_cols_item = [
        f"emb_{i}"
        for i in range(ITEM_EMBEDDING_DIM)
    ]

    if not set(embedding_cols_item).issubset(
        set(item_embeddings.columns)
    ):
        raise ValueError(
            "Item embeddings columns do not match "
            "ITEM_EMBEDDING_DIM."
        )

    item_embeddings = (
        item_embeddings[
            embedding_cols_item
        ]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # ========================================================
    # CANDIDATE ITEMS
    # ========================================================

    candidates = [
        item
        for item in top_items_global
        if item in item_embeddings.index
        and item in item_pop.index
    ]

    if not candidates:
        raise ValueError(
            "No valid candidate items exist in both "
            "item_statistics.csv and item_embeddings.csv."
        )

    print(
        f"Candidate items: {len(candidates):,}"
    )

    # --------------------------------------------------------
    # Convert candidate embeddings ONCE
    # --------------------------------------------------------

    candidate_embeddings = (
        item_embeddings
        .loc[candidates]
        .values
        .astype(np.float32)
    )

    # ========================================================
    # PRECOMPUTE POPULARITY SCORES
    # ========================================================

    popularity_values = (
        item_pop
        .reindex(candidates)
        .fillna(0)
        .values
        .astype(np.float32)
    )

    pop_scores = np.log1p(
        popularity_values
    )

    max_pop = pop_scores.max()

    if max_pop > 0:

        normalized_popularity = (
            pop_scores / max_pop
        )

    else:

        normalized_popularity = (
            np.zeros_like(pop_scores)
        )

    # ========================================================
    # SEGMENT-ITEM AFFINITY + RECENCY
    # ========================================================

    print("Processing historical events...")

    event_start = time.time()

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    events_path = (
        PROJECT_ROOT /
        "data" /
        "events.csv"
    )

    seg_item_score_lookup = {}
    last_interaction_lookup = {}

    if events_path.is_file():

        events_df = pd.read_csv(
            events_path,
            usecols=[
                "visitorid",
                "itemid",
                "event",
                "timestamp"
            ]
        )

        print(
            f"Events loaded: "
            f"{len(events_df):,}"
        )

        # ----------------------------------------------------
        # Clean IDs
        # ----------------------------------------------------

        events_df["visitorid"] = (
            clean_visitor_ids(
                events_df["visitorid"]
            )
        )

        events_df["itemid"] = (
            events_df["itemid"]
            .astype(str)
        )

        # ----------------------------------------------------
        # Event weights
        # ----------------------------------------------------

        event_weights = {
            "view": 1,
            "addtocart": 3,
            "transaction": 5
        }

        events_df["weight"] = (
            events_df["event"]
            .map(event_weights)
            .fillna(0)
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Add user segments
        #
        # Use map instead of merge here.
        # This avoids an expensive large merge on millions
        # of historical events.
        # ----------------------------------------------------

        visitor_segment_map = (
            df[
                [
                    "visitorid",
                    "segment"
                ]
            ]
            .drop_duplicates(
                subset=["visitorid"]
            )
            .set_index("visitorid")["segment"]
        )

        events_df["segment"] = (
            events_df["visitorid"]
            .map(visitor_segment_map)
        )

        # ----------------------------------------------------
        # Segment-item affinity
        # ----------------------------------------------------

        seg_item_agg = (
            events_df
            .dropna(subset=["segment"])
            .groupby(
                [
                    "segment",
                    "itemid"
                ],
                sort=False
            )["weight"]
            .sum()
            .reset_index()
        )

        if not seg_item_agg.empty:

            max_weight = (
                seg_item_agg["weight"]
                .max()
            )

            if max_weight > 0:

                seg_item_agg["norm_score"] = (
                    seg_item_agg["weight"]
                    / max_weight
                )

            else:

                seg_item_agg["norm_score"] = 0.0

            seg_item_score_lookup = {
                (
                    int(row.segment),
                    str(row.itemid)
                ): float(row.norm_score)
                for row in (
                    seg_item_agg
                    .itertuples(
                        index=False
                    )
                )
            }

        # ----------------------------------------------------
        # Last interaction timestamp
        # ----------------------------------------------------

        print(
            "Calculating last interaction timestamps..."
        )

        last_ts_df = (
            events_df
            .groupby(
                [
                    "visitorid",
                    "itemid"
                ],
                sort=False
            )["timestamp"]
            .max()
            .reset_index()
        )

        # ----------------------------------------------------
        # Create dictionary
        # ----------------------------------------------------

        for uid, item_id, timestamp in (
            last_ts_df.itertuples(
                index=False
            )
        ):

            try:

                timestamp_value = int(
                    timestamp
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            uid = str(uid)
            item_id = str(item_id)

            if uid not in last_interaction_lookup:
                last_interaction_lookup[uid] = {}

            last_interaction_lookup[
                uid
            ][item_id] = timestamp_value

        del last_ts_df
        del events_df

    else:

        print(
            f"Warning: events file not found at "
            f"{events_path}"
        )

    print(
        f"Historical event processing completed "
        f"in {time.time() - event_start:.2f}s"
    )

    # ========================================================
    # PREPARE USER EMBEDDINGS
    # ========================================================

    print(
        f"Generating recommendations for "
        f"{len(df):,} users..."
    )

    user_embeddings = (
        df[embedding_cols]
        .values
        .astype(np.float32)
    )

    user_segments = (
        df["segment"]
        .values
        .astype(int)
    )

    user_ids = (
        df["visitorid"]
        .astype(str)
        .values
    )

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    similarity_start = time.time()

    print(
        "Calculating user-item similarity matrix..."
    )

    similarities_matrix = cosine_similarity(
        user_embeddings,
        candidate_embeddings
    ).astype(np.float32)

    print(
        f"Similarity matrix calculated in "
        f"{time.time() - similarity_start:.2f}s"
    )

    # ========================================================
    # RECOMMENDATION GENERATION
    # ========================================================

    recommendation_start = time.time()

    recs = []

    current_ts_ms = int(
        pd.Timestamp.utcnow().timestamp()
        * 1000
    )

    decay_ms = (
        RECENCY_DECAY_DAYS
        * 86400000
    )

    # ========================================================
    # PROCESS USERS
    # ========================================================

    for user_idx in range(
        len(user_ids)
    ):

        user_id = user_ids[user_idx]

        segment = int(
            user_segments[user_idx]
        )

        similarities = (
            similarities_matrix[user_idx]
        )

        # ----------------------------------------------------
        # Segment scores
        # ----------------------------------------------------

        seg_scores = np.fromiter(
            (
                seg_item_score_lookup.get(
                    (
                        segment,
                        str(item)
                    ),
                    0.0
                )
                for item in candidates
            ),
            dtype=np.float32,
            count=len(candidates)
        )

        # ----------------------------------------------------
        # Recency scores
        # ----------------------------------------------------

        user_history = (
            last_interaction_lookup.get(
                user_id,
                {}
            )
        )

        if user_history:

            timestamps = np.array(
                [
                    user_history.get(
                        str(item),
                        0
                    )
                    for item in candidates
                ],
                dtype=np.float64
            )

            has_interaction = (
                timestamps > 0
            )

            recency_scores = np.zeros(
                len(candidates),
                dtype=np.float32
            )

            recency_scores[
                has_interaction
            ] = np.exp(
                -(
                    current_ts_ms
                    - timestamps[
                        has_interaction
                    ]
                )
                / decay_ms
            ).astype(np.float32)

        else:

            recency_scores = np.zeros(
                len(candidates),
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Dynamic recommendation score
        # ----------------------------------------------------

        scores = (
            USER_ITEM_WEIGHT
            * similarities

            + POPULARITY_WEIGHT
            * normalized_popularity

            + SEGMENT_WEIGHT
            * seg_scores

            + RECENCY_WEIGHT
            * recency_scores
        )

        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        top_k = min(
            TOP_K,
            len(candidates)
        )

        # argpartition is faster than sorting everything
        top_idx = np.argpartition(
            -scores,
            top_k - 1
        )[:top_k]

        # Sort only selected Top-K
        top_idx = top_idx[
            np.argsort(
                -scores[top_idx]
            )
        ]

        # ----------------------------------------------------
        # Build output
        # ----------------------------------------------------

        for rank, idx in enumerate(
            top_idx,
            start=1
        ):

            recs.append({
                "visitorid": user_id,
                "itemid": candidates[idx],
                "score": float(
                    scores[idx]
                ),
                "segment": segment,
                "recommendation_rank": rank,
                "similarity_score": float(
                    similarities[idx]
                ),
                "popularity_score": float(
                    normalized_popularity[idx]
                ),
                "segment_score": float(
                    seg_scores[idx]
                ),
                "recency_score": float(
                    recency_scores[idx]
                )
            })

        # ----------------------------------------------------
        # Progress indicator
        # ----------------------------------------------------

        if (
            user_idx + 1
        ) % 500 == 0:

            elapsed = (
                time.time()
                - recommendation_start
            )

            print(
                f"Processed "
                f"{user_idx + 1:,}/"
                f"{len(user_ids):,} users "
                f"({elapsed:.2f}s)"
            )

    # ========================================================
    # SAVE
    # ========================================================

    print("Saving recommendations...")

    recs_df = pd.DataFrame(
        recs
    )

    output_path = (
        OUTPUT_DIR /
        "recommendations.csv"
    )

    recs_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    recommendation_time = (
        time.time()
        - recommendation_start
    )

    total_time = (
        time.time()
        - total_start
    )

    print()
    print(
        f"Generated "
        f"{len(recs_df):,} recommendations."
    )

    print(
        f"Saved recommendations to "
        f"{output_path}"
    )

    print(
        f"Recommendation generation time: "
        f"{recommendation_time:.2f}s"
    )

    print(
        f"Total dynamic recommendation time: "
        f"{total_time:.2f}s"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_recommendations()