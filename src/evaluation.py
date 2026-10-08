"""
Evaluation of the existing recommendation system.

Evaluation design:
- Transformer is trained beforehand on the existing training users.
- Users present in user_embeddings.csv are treated as training users.
- Remaining users with >= 5 interactions are unseen test users.
- Each test user is split chronologically: 80% history / 20% future.
- Recommendations are generated only from the existing candidate pool.
- The primary quality metrics are calculated only for "candidate-reachable"
  users, i.e. users whose future contains at least one candidate item.
  This prevents impossible ground-truth items from artificially forcing misses.
- A separate all-user Hit@5/Precision@5/Recall@5/NDCG@5 is also reported
  for transparency.
- Segment-item affinity is built ONLY from training users, preventing future
  test-user information from leaking into evaluation.
- The existing recommendation score and weights are preserved.
- A popularity-only baseline is evaluated on the same candidate pool.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# PROJECT CONFIG
# ============================================================

from src.config import (
    OUTPUT_DIR,
    DATA_DIR,
    TOP_K,
    USER_ITEM_WEIGHT,
    POPULARITY_WEIGHT,
    SEGMENT_WEIGHT,
    RECENCY_WEIGHT,
    ITEM_EMBEDDING_DIM,
    RECENCY_DECAY_DAYS,
    MAX_SEQUENCE_LENGTH,
)


# ============================================================
# METRICS
# ============================================================

def hit_at_k(recs, ground_truth):
    return int(bool(set(recs) & set(ground_truth)))


def precision_at_k(recs, ground_truth):
    if not recs:
        return 0.0
    return len(set(recs) & set(ground_truth)) / len(recs)


def recall_at_k(recs, ground_truth):
    if not ground_truth:
        return 0.0
    return len(set(recs) & set(ground_truth)) / len(ground_truth)


def ndcg_at_k(recs, ground_truth, k=TOP_K):
    ground_truth = set(ground_truth)

    dcg = 0.0
    for i, item in enumerate(recs[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)

    ideal_len = min(k, len(ground_truth))
    idcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(ideal_len)
    )

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# HELPERS
# ============================================================

def normalize_ids(series):
    return series.astype(str).str.strip()


def safe_user_sort_key(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def encode_sequence(vocab, tokens):
    ids = [vocab.encode(token) for token in tokens]

    if len(ids) > MAX_SEQUENCE_LENGTH:
        ids = ids[-MAX_SEQUENCE_LENGTH:]

    elif len(ids) < MAX_SEQUENCE_LENGTH:
        pad_id = vocab.encode("<PAD>")
        ids = ids + [
            pad_id
            for _ in range(MAX_SEQUENCE_LENGTH - len(ids))
        ]

    return torch.tensor(
        ids,
        dtype=torch.long,
    ).unsqueeze(0)


def top_k_from_scores(items, scores, k):
    k = min(k, len(items))

    if k <= 0:
        return [], np.array([], dtype=np.float32)

    idx = np.argpartition(
        -scores,
        k - 1,
    )[:k]

    idx = idx[
        np.argsort(-scores[idx])
    ]

    return (
        [items[i] for i in idx],
        scores[idx],
    )


# ============================================================
# MAIN
# ============================================================

def run_evaluation():

    total_start = time.perf_counter()

    print()
    print("=" * 70)
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 70)

    # ========================================================
    # 1. LOAD EVENTS
    # ========================================================

    events_path = DATA_DIR / "events.csv"

    if not events_path.is_file():
        raise FileNotFoundError(
            f"events.csv not found at {events_path}"
        )

    print("\nLoading events...")

    events_df = pd.read_csv(
        events_path,
        usecols=[
            "visitorid",
            "itemid",
            "event",
            "timestamp",
        ],
    )

    events_df["visitorid"] = normalize_ids(
        events_df["visitorid"]
    )

    events_df["itemid"] = normalize_ids(
        events_df["itemid"]
    )

    events_df["event"] = (
        events_df["event"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    events_df["timestamp"] = pd.to_numeric(
        events_df["timestamp"],
        errors="coerce",
    )

    events_df = events_df.dropna(
        subset=[
            "visitorid",
            "itemid",
            "timestamp",
        ]
    )

    events_df["timestamp"] = events_df["timestamp"].astype(np.int64)

    print(
        f"Events loaded: {len(events_df):,}"
    )

    # ========================================================
    # 2. LOAD TRAINED TRANSFORMER
    # ========================================================

    print("\nLoading trained Transformer...")

    from src import transformer_embeddings

    model, vocab = (
        transformer_embeddings.load_trained_model()
    )

    model.eval()
    model.to("cpu")

    print("Transformer loaded successfully.")

    # ========================================================
    # 3. LOAD ITEM EMBEDDINGS
    # ========================================================

    print("\nLoading item embeddings...")

    item_emb_path = (
        OUTPUT_DIR / "item_embeddings.csv"
    )

    if not item_emb_path.is_file():
        raise FileNotFoundError(
            f"item_embeddings.csv not found at {item_emb_path}"
        )

    item_embeddings = pd.read_csv(
        item_emb_path,
        index_col="itemid",
    )

    item_embeddings.index = normalize_ids(
        item_embeddings.index
    )

    embedding_cols = [
        f"emb_{i}"
        for i in range(ITEM_EMBEDDING_DIM)
    ]

    missing_cols = [
        col
        for col in embedding_cols
        if col not in item_embeddings.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing item embedding columns: "
            + str(missing_cols)
        )

    item_embeddings = (
        item_embeddings[embedding_cols]
        .astype(np.float32)
    )

    item_embeddings = item_embeddings[
        ~item_embeddings.index.duplicated(
            keep="first"
        )
    ]

    print(
        f"Unique item embeddings: "
        f"{len(item_embeddings):,}"
    )

    # ========================================================
    # 4. LOAD CUSTOMER SEGMENTS
    # ========================================================

    print("\nLoading customer segments...")

    seg_path = (
        OUTPUT_DIR / "customer_segments.csv"
    )

    seg_map = {}

    if seg_path.is_file():

        segments_df = pd.read_csv(
            seg_path
        )

        if (
            "visitorid" in segments_df.columns
            and "segment" in segments_df.columns
        ):

            segments_df["visitorid"] = (
                normalize_ids(
                    segments_df["visitorid"]
                )
            )

            segments_df = (
                segments_df
                .drop_duplicates(
                    "visitorid",
                    keep="last",
                )
            )

            seg_map = dict(
                zip(
                    segments_df["visitorid"],
                    segments_df["segment"].astype(int),
                )
            )

    print(
        f"Users with segment information: "
        f"{len(seg_map):,}"
    )

    # ========================================================
    # 5. LOAD USER EMBEDDINGS
    # ========================================================

    print("\nLoading user embeddings...")

    user_emb_path = (
        OUTPUT_DIR / "user_embeddings.csv"
    )

    if not user_emb_path.is_file():
        raise FileNotFoundError(
            f"user_embeddings.csv not found at "
            f"{user_emb_path}"
        )

    user_emb_df = pd.read_csv(
        user_emb_path
    )

    if "visitorid" not in user_emb_df.columns:
        raise ValueError(
            "user_embeddings.csv does not contain visitorid."
        )

    user_emb_df["visitorid"] = (
        normalize_ids(
            user_emb_df["visitorid"]
        )
    )

    # Users represented in user_embeddings.csv are the
    # users used by the existing Transformer training pipeline.
    training_user_ids = set(
        user_emb_df["visitorid"].unique()
    )

    # ========================================================
    # 6. USER SPLIT
    # ========================================================

    all_user_ids = set(
        events_df["visitorid"].unique()
    )

    test_user_ids = (
        all_user_ids - training_user_ids
    )

    print("\nUser split:")
    print(
        f"Total unique users: "
        f"{len(all_user_ids):,}"
    )
    print(
        f"Training users: "
        f"{len(training_user_ids):,}"
    )
    print(
        f"Remaining test users: "
        f"{len(test_user_ids):,}"
    )

    # ========================================================
    # 7. ELIGIBLE TEST USERS
    # ========================================================

    print(
        "\nFinding test users with enough interactions..."
    )

    user_event_counts = (
        events_df
        .groupby(
            "visitorid",
            sort=False,
        )
        .size()
    )

    eligible_test_users = set(
        user_event_counts[
            user_event_counts >= 5
        ].index
    )

    eligible_test_users -= training_user_ids

    print(
        f"Eligible test users (>=5 interactions): "
        f"{len(eligible_test_users):,}"
    )

    if not eligible_test_users:
        raise RuntimeError(
            "No eligible test users found."
        )

    # ========================================================
    # 8. FILTER TEST EVENTS
    # ========================================================

    print(
        "\nFiltering events to eligible test users..."
    )

    evaluation_events = events_df[
        events_df["visitorid"].isin(
            eligible_test_users
        )
    ].copy()

    evaluation_events = (
        evaluation_events
        .sort_values(
            [
                "visitorid",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Evaluation events: "
        f"{len(evaluation_events):,}"
    )

    # ========================================================
    # 9. ITEM STATISTICS
    # ========================================================

    print("\nLoading item statistics...")

    item_stats_path = (
        OUTPUT_DIR / "item_statistics.csv"
    )

    if not item_stats_path.is_file():
        raise FileNotFoundError(
            f"item_statistics.csv not found at "
            f"{item_stats_path}"
        )

    item_stats = pd.read_csv(
        item_stats_path,
        index_col="itemid",
    )

    item_stats.index = normalize_ids(
        item_stats.index
    )

    if "view" not in item_stats.columns:
        raise ValueError(
            "item_statistics.csv must contain 'view' column."
        )

    item_pop = (
        item_stats["view"]
        .fillna(0)
        .astype(np.float32)
    )

    # ========================================================
    # 10. CANDIDATE ITEMS
    # ========================================================

    print("\nBuilding candidate item pool...")

    # Preserve the existing recommendation philosophy:
    # top 500 popular items, then intersect with available
    # item embeddings.
    top_items_global = (
        item_pop
        .sort_values(ascending=False)
        .head(500)
        .index
        .astype(str)
        .tolist()
    )

    candidates = [
        item
        for item in top_items_global
        if item in item_embeddings.index
    ]

    if not candidates:

        common_items = (
            item_pop.index
            .intersection(
                item_embeddings.index
            )
        )

        if len(common_items) == 0:
            raise RuntimeError(
                "No common item IDs between "
                "item_statistics.csv and "
                "item_embeddings.csv."
            )

        candidates = (
            item_pop
            .loc[common_items]
            .sort_values(ascending=False)
            .head(500)
            .index
            .astype(str)
            .tolist()
        )

        print(
            "WARNING: Used common-item fallback."
        )

    print(
        f"Candidate items: "
        f"{len(candidates):,}"
    )

    candidate_set = set(candidates)

    candidate_embeddings = (
        item_embeddings
        .loc[candidates]
        .values
        .astype(np.float32)
    )

    # ========================================================
    # 11. NORMALIZED POPULARITY
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

    max_pop = (
        pop_scores.max()
        if len(pop_scores)
        else 0
    )

    if max_pop > 0:
        normalized_popularity = (
            pop_scores / max_pop
        )
    else:
        normalized_popularity = (
            np.zeros_like(pop_scores)
        )

    # ========================================================
    # 12. EVENT WEIGHTS
    # ========================================================

    event_weights = {
        "view": 1,
        "addtocart": 3,
        "transaction": 5,
    }

    evaluation_events["weight"] = (
        evaluation_events["event"]
        .map(event_weights)
        .fillna(0)
        .astype(np.float32)
    )

    # ========================================================
    # 13. LEAKAGE-FREE SEGMENT-ITEM AFFINITY
    # ========================================================
    #
    # IMPORTANT:
    # The previous evaluator calculated segment-item affinity
    # from all test-user events before their 80/20 split.
    # That allowed future test interactions to influence scores.
    #
    # We now calculate this ONLY from the training users.
    # Therefore no test-user future information enters the
    # segment component.
    # ========================================================

    print(
        "\nBuilding leakage-free segment-item affinity..."
    )

    training_events = events_df[
        events_df["visitorid"].isin(
            training_user_ids
        )
    ].copy()

    training_events["segment"] = (
        training_events["visitorid"]
        .map(seg_map)
    )

    seg_item_agg = (
        training_events
        .dropna(subset=["segment"])
        .assign(
            weight=lambda df: (
                df["event"]
                .map(event_weights)
                .fillna(0)
                .astype(np.float32)
            )
        )
        .groupby(
            [
                "segment",
                "itemid",
            ],
            as_index=False,
        )["weight"]
        .sum()
    )

    if not seg_item_agg.empty:

        max_weight = (
            seg_item_agg["weight"].max()
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
                str(row.itemid),
            ): float(row.norm_score)
            for row in seg_item_agg.itertuples(
                index=False
            )
        }

    else:
        seg_item_score_lookup = {}

    print(
        f"Training events used for segment affinity: "
        f"{len(training_events):,}"
    )

    print(
        f"Segment-item relationships: "
        f"{len(seg_item_score_lookup):,}"
    )

    # ========================================================
    # 14. GROUP TEST USERS
    # ========================================================

    print("\nGrouping eligible users...")

    user_groups = {
        uid: group
        for uid, group
        in evaluation_events.groupby(
            "visitorid",
            sort=False,
        )
    }

    print(
        f"User groups prepared: "
        f"{len(user_groups):,}"
    )

    # ========================================================
    # 15. EVALUATION CONTAINERS
    # ========================================================

    per_user_rows = []
    rec_rows = []

    excluded_users = 0

    exclusion_reasons = {
        "less_than_2_events": 0,
        "empty_future": 0,
        "no_valid_history": 0,
    }

    evaluation_start = time.perf_counter()

    test_user_list = sorted(
        eligible_test_users,
        key=safe_user_sort_key,
    )

    total_test_users = len(
        test_user_list
    )

    print()
    print(
        f"Starting evaluation for "
        f"{total_test_users:,} users..."
    )

    # ========================================================
    # 16. USER-BY-USER EVALUATION
    # ========================================================

    for idx, user_id in enumerate(
        test_user_list,
        start=1,
    ):

        if idx % 500 == 0:
            elapsed = (
                time.perf_counter()
                - evaluation_start
            )

            print(
                f"Processed "
                f"{idx:,}/{total_test_users:,} "
                f"users ({elapsed:.2f}s)"
            )

        user_events = user_groups.get(
            user_id
        )

        if user_events is None:
            excluded_users += 1
            continue

        user_events = (
            user_events
            .sort_values("timestamp")
        )

        # ----------------------------------------------------
        # Chronological 80/20 split
        # ----------------------------------------------------

        if len(user_events) < 2:
            excluded_users += 1
            exclusion_reasons[
                "less_than_2_events"
            ] += 1
            continue

        split_idx = int(
            0.8 * len(user_events)
        )

        split_idx = max(
            1,
            min(
                split_idx,
                len(user_events) - 1,
            ),
        )

        historical = (
            user_events.iloc[:split_idx]
        )

        future = (
            user_events.iloc[split_idx:]
        )

        if historical.empty:
            excluded_users += 1
            exclusion_reasons[
                "no_valid_history"
            ] += 1
            continue

        if future.empty:
            excluded_users += 1
            exclusion_reasons[
                "empty_future"
            ] += 1
            continue

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        ground_truth_all = set(
            future["itemid"]
            .astype(str)
            .unique()
        )

        # IMPORTANT:
        # Only items that the recommender can actually return
        # can be valid candidates for the candidate-restricted
        # ranking metrics.
        ground_truth_candidate = (
            ground_truth_all
            & candidate_set
        )

        candidate_reachable = int(
            bool(ground_truth_candidate)
        )

        # ----------------------------------------------------
        # Historical Transformer sequence
        # ----------------------------------------------------

        tokens = [
            f"{row.event}_item_{row.itemid}"
            for row in historical.itertuples(
                index=False
            )
        ]

        if not tokens:
            excluded_users += 1
            exclusion_reasons[
                "no_valid_history"
            ] += 1
            continue

        input_ids = encode_sequence(
            vocab,
            tokens,
        )

        # ----------------------------------------------------
        # Transformer inference
        # ----------------------------------------------------

        with torch.no_grad():
            user_emb = (
                model
                .get_user_embedding(
                    input_ids
                )
                .cpu()
                .numpy()
                .squeeze(0)
            )

        # ----------------------------------------------------
        # Segment
        # ----------------------------------------------------

        segment = int(
            seg_map.get(
                user_id,
                0,
            )
        )

        # ----------------------------------------------------
        # USER-ITEM SIMILARITY
        # ----------------------------------------------------

        similarities = (
            cosine_similarity(
                user_emb.reshape(
                    1,
                    -1,
                ),
                candidate_embeddings,
            )
            .astype(np.float32)
            .flatten()
        )

        # ----------------------------------------------------
        # SEGMENT SCORE
        # ----------------------------------------------------

        seg_scores = np.fromiter(
            (
                seg_item_score_lookup.get(
                    (
                        segment,
                        str(item),
                    ),
                    0.0,
                )
                for item in candidates
            ),
            dtype=np.float32,
            count=len(candidates),
        )

        # ----------------------------------------------------
        # RECENCY
        # ----------------------------------------------------
        # ONLY historical interactions are used.
        # ----------------------------------------------------

        historical_lookup = {}

        for row in historical.itertuples(
            index=False
        ):

            item_id = str(row.itemid)
            timestamp = int(row.timestamp)

            previous = historical_lookup.get(
                item_id
            )

            if (
                previous is None
                or timestamp > previous
            ):
                historical_lookup[
                    item_id
                ] = timestamp

        current_ts_ms = int(
            historical["timestamp"].max()
        )

        decay_ms = (
            RECENCY_DECAY_DAYS
            * 86400000
        )

        if decay_ms <= 0:
            decay_ms = 1

        recency_scores = np.fromiter(
            (
                np.exp(
                    -(
                        current_ts_ms
                        - historical_lookup[item]
                    )
                    / decay_ms
                )
                if item in historical_lookup
                else 0.0
                for item in candidates
            ),
            dtype=np.float32,
            count=len(candidates),
        )

        # ----------------------------------------------------
        # EXISTING RECOMMENDATION SCORE
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
        # MODEL TOP-K
        # ----------------------------------------------------

        top_items, top_scores = (
            top_k_from_scores(
                candidates,
                scores,
                TOP_K,
            )
        )

        # ----------------------------------------------------
        # POPULARITY BASELINE
        # ----------------------------------------------------

        popularity_top_items, _ = (
            top_k_from_scores(
                candidates,
                normalized_popularity,
                TOP_K,
            )
        )

        # ----------------------------------------------------
        # ALL-USER METRICS
        # ----------------------------------------------------

        all_hit = hit_at_k(
            top_items,
            ground_truth_all,
        )

        all_precision = precision_at_k(
            top_items,
            ground_truth_all,
        )

        all_recall = recall_at_k(
            top_items,
            ground_truth_all,
        )

        all_ndcg = ndcg_at_k(
            top_items,
            ground_truth_all,
        )

        # ----------------------------------------------------
        # CANDIDATE-RESTRICTED METRICS
        # ----------------------------------------------------

        if candidate_reachable:

            model_hit = hit_at_k(
                top_items,
                ground_truth_candidate,
            )

            model_precision = precision_at_k(
                top_items,
                ground_truth_candidate,
            )

            model_recall = recall_at_k(
                top_items,
                ground_truth_candidate,
            )

            model_ndcg = ndcg_at_k(
                top_items,
                ground_truth_candidate,
            )

            baseline_hit = hit_at_k(
                popularity_top_items,
                ground_truth_candidate,
            )

            baseline_precision = precision_at_k(
                popularity_top_items,
                ground_truth_candidate,
            )

            baseline_recall = recall_at_k(
                popularity_top_items,
                ground_truth_candidate,
            )

            baseline_ndcg = ndcg_at_k(
                popularity_top_items,
                ground_truth_candidate,
            )

        else:

            model_hit = np.nan
            model_precision = np.nan
            model_recall = np.nan
            model_ndcg = np.nan

            baseline_hit = np.nan
            baseline_precision = np.nan
            baseline_recall = np.nan
            baseline_ndcg = np.nan

        # ----------------------------------------------------
        # PER-USER METRICS
        # ----------------------------------------------------

        per_user_rows.append(
            {
                "visitorid": user_id,

                "candidate_reachable":
                    candidate_reachable,

                "num_ground_truth_items":
                    len(ground_truth_all),

                "num_candidate_ground_truth_items":
                    len(ground_truth_candidate),

                # Primary candidate-restricted metrics
                "hit@5":
                    model_hit,

                "precision@5":
                    model_precision,

                "recall@5":
                    model_recall,

                "ndcg@5":
                    model_ndcg,

                # Full future-set diagnostic metrics
                "all_hit@5":
                    all_hit,

                "all_precision@5":
                    all_precision,

                "all_recall@5":
                    all_recall,

                "all_ndcg@5":
                    all_ndcg,

                # Popularity baseline
                "popularity_hit@5":
                    baseline_hit,

                "popularity_precision@5":
                    baseline_precision,

                "popularity_recall@5":
                    baseline_recall,

                "popularity_ndcg@5":
                    baseline_ndcg,
            }
        )

        # ----------------------------------------------------
        # RECOMMENDATION OUTPUT
        # ----------------------------------------------------

        for rank, (
            item_id,
            score,
        ) in enumerate(
            zip(
                top_items,
                top_scores,
            ),
            start=1,
        ):

            is_relevant = int(
                item_id in ground_truth_candidate
            )

            rec_rows.append(
                {
                    "visitorid": user_id,
                    "itemid": item_id,
                    "recommendation_rank": rank,
                    "score": float(score),
                    "is_relevant": is_relevant,
                    "candidate_reachable":
                        candidate_reachable,
                }
            )

    # ========================================================
    # 17. AGGREGATE
    # ========================================================

    per_user_df = pd.DataFrame(
        per_user_rows
    )

    test_recs_df = pd.DataFrame(
        rec_rows
    )

    evaluated_users = len(
        per_user_df
    )

    reachable_df = (
        per_user_df[
            per_user_df[
                "candidate_reachable"
            ] == 1
        ]
        if evaluated_users > 0
        else pd.DataFrame()
    )

    reachable_users = len(
        reachable_df
    )

    unreachable_users = (
        evaluated_users
        - reachable_users
    )

    # Primary metrics: candidate-restricted
    if reachable_users > 0:

        model_metrics = {
            "hit_rate@5":
                reachable_df["hit@5"].mean(),

            "precision@5":
                reachable_df["precision@5"].mean(),

            "recall@5":
                reachable_df["recall@5"].mean(),

            "ndcg@5":
                reachable_df["ndcg@5"].mean(),
        }

        popularity_metrics = {
            "hit_rate@5":
                reachable_df[
                    "popularity_hit@5"
                ].mean(),

            "precision@5":
                reachable_df[
                    "popularity_precision@5"
                ].mean(),

            "recall@5":
                reachable_df[
                    "popularity_recall@5"
                ].mean(),

            "ndcg@5":
                reachable_df[
                    "popularity_ndcg@5"
                ].mean(),
        }

    else:

        model_metrics = {
            "hit_rate@5": np.nan,
            "precision@5": np.nan,
            "recall@5": np.nan,
            "ndcg@5": np.nan,
        }

        popularity_metrics = {
            "hit_rate@5": np.nan,
            "precision@5": np.nan,
            "recall@5": np.nan,
            "ndcg@5": np.nan,
        }

    # Full future-set diagnostic metrics
    if evaluated_users > 0:

        all_user_metrics = {
            "hit_rate@5":
                per_user_df["all_hit@5"].mean(),

            "precision@5":
                per_user_df["all_precision@5"].mean(),

            "recall@5":
                per_user_df["all_recall@5"].mean(),

            "ndcg@5":
                per_user_df["all_ndcg@5"].mean(),
        }

    else:

        all_user_metrics = {
            "hit_rate@5": np.nan,
            "precision@5": np.nan,
            "recall@5": np.nan,
            "ndcg@5": np.nan,
        }

    # ========================================================
    # 18. COVERAGE
    # ========================================================

    if not test_recs_df.empty:

        unique_recommended_items = (
            test_recs_df[
                "itemid"
            ].nunique()
        )

        total_recommendations = len(
            test_recs_df
        )

        duplicate_count = (
            test_recs_df
            .duplicated(
                subset=[
                    "visitorid",
                    "itemid",
                ]
            )
            .sum()
        )

    else:

        unique_recommended_items = 0
        total_recommendations = 0
        duplicate_count = 0

    catalog_coverage = (
        unique_recommended_items
        / len(candidates)
        if candidates
        else np.nan
    )

    candidate_reach_rate = (
        reachable_users / evaluated_users
        if evaluated_users > 0
        else np.nan
    )

    # ========================================================
    # 19. TIMING
    # ========================================================

    evaluation_time = (
        time.perf_counter()
        - evaluation_start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    avg_time_per_user = (
        evaluation_time
        / evaluated_users
        if evaluated_users > 0
        else np.nan
    )

    # ========================================================
    # 20. SAVE OUTPUTS
    # ========================================================

    eval_dir = (
        OUTPUT_DIR / "evaluation"
    )

    eval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_rows = [

        # Primary model metrics
        {
            "metric": "model_hit_rate@5",
            "value": model_metrics["hit_rate@5"],
        },
        {
            "metric": "model_precision@5",
            "value": model_metrics["precision@5"],
        },
        {
            "metric": "model_recall@5",
            "value": model_metrics["recall@5"],
        },
        {
            "metric": "model_ndcg@5",
            "value": model_metrics["ndcg@5"],
        },

        # Popularity baseline
        {
            "metric": "popularity_hit_rate@5",
            "value": popularity_metrics["hit_rate@5"],
        },
        {
            "metric": "popularity_precision@5",
            "value": popularity_metrics["precision@5"],
        },
        {
            "metric": "popularity_recall@5",
            "value": popularity_metrics["recall@5"],
        },
        {
            "metric": "popularity_ndcg@5",
            "value": popularity_metrics["ndcg@5"],
        },

        # All-user diagnostic
        {
            "metric": "all_user_hit_rate@5",
            "value": all_user_metrics["hit_rate@5"],
        },
        {
            "metric": "all_user_precision@5",
            "value": all_user_metrics["precision@5"],
        },
        {
            "metric": "all_user_recall@5",
            "value": all_user_metrics["recall@5"],
        },
        {
            "metric": "all_user_ndcg@5",
            "value": all_user_metrics["ndcg@5"],
        },

        # Dataset/evaluation information
        {
            "metric": "total_users",
            "value": len(all_user_ids),
        },
        {
            "metric": "training_users",
            "value": len(training_user_ids),
        },
        {
            "metric": "remaining_test_users",
            "value": len(test_user_ids),
        },
        {
            "metric": "eligible_test_users",
            "value": len(eligible_test_users),
        },
        {
            "metric": "evaluated_users",
            "value": evaluated_users,
        },
        {
            "metric": "candidate_reachable_users",
            "value": reachable_users,
        },
        {
            "metric": "candidate_unreachable_users",
            "value": unreachable_users,
        },
        {
            "metric": "candidate_reach_rate",
            "value": candidate_reach_rate,
        },
        {
            "metric": "candidate_items",
            "value": len(candidates),
        },
        {
            "metric": "total_recommendations",
            "value": total_recommendations,
        },
        {
            "metric": "catalog_coverage",
            "value": catalog_coverage,
        },
        {
            "metric": "unique_recommended_items",
            "value": unique_recommended_items,
        },
        {
            "metric": "duplicate_recommendations",
            "value": duplicate_count,
        },
        {
            "metric": "evaluation_time_seconds",
            "value": evaluation_time,
        },
        {
            "metric": "avg_time_per_user_seconds",
            "value": avg_time_per_user,
        },
    ]

    pd.DataFrame(
        overall_rows
    ).to_csv(
        eval_dir / "evaluation_metrics.csv",
        index=False,
    )

    per_user_df.to_csv(
        eval_dir / "per_user_metrics.csv",
        index=False,
    )

    test_recs_df.to_csv(
        eval_dir / "test_recommendations.csv",
        index=False,
    )

    # ========================================================
    # 21. SUMMARY
    # ========================================================

    summary = f"""
RECOMMENDATION SYSTEM EVALUATION
================================

Total unique users:
{len(all_user_ids):,}

Training users:
{len(training_user_ids):,}

Remaining unseen test users:
{len(test_user_ids):,}

Eligible test users (>=5 interactions):
{len(eligible_test_users):,}

Evaluated users:
{evaluated_users:,}

Candidate-reachable users:
{reachable_users:,}

Candidate-unreachable users:
{unreachable_users:,}

Candidate reach rate:
{candidate_reach_rate:.6f}

Candidate items:
{len(candidates):,}

Top-K:
{TOP_K}


PRIMARY MODEL METRICS
---------------------

These metrics are calculated only for users whose future
contains at least one item from the candidate pool.

Hit Rate@5:
{model_metrics["hit_rate@5"]:.6f}

Precision@5:
{model_metrics["precision@5"]:.6f}

Recall@5:
{model_metrics["recall@5"]:.6f}

NDCG@5:
{model_metrics["ndcg@5"]:.6f}


POPULARITY BASELINE
-------------------

Hit Rate@5:
{popularity_metrics["hit_rate@5"]:.6f}

Precision@5:
{popularity_metrics["precision@5"]:.6f}

Recall@5:
{popularity_metrics["recall@5"]:.6f}

NDCG@5:
{popularity_metrics["ndcg@5"]:.6f}


ALL-USER DIAGNOSTIC METRICS
---------------------------

These include users whose future contains no candidate
items, so they are intentionally reported only as diagnostics.

Hit Rate@5:
{all_user_metrics["hit_rate@5"]:.6f}

Precision@5:
{all_user_metrics["precision@5"]:.6f}

Recall@5:
{all_user_metrics["recall@5"]:.6f}

NDCG@5:
{all_user_metrics["ndcg@5"]:.6f}


CATALOG
-------

Unique recommended items:
{unique_recommended_items:,}

Catalog coverage:
{catalog_coverage:.6f}

Total recommendations:
{total_recommendations:,}

Duplicate user-item recommendations:
{duplicate_count:,}


TIMING
------

Evaluation time:
{evaluation_time:.2f} seconds

Average time per evaluated user:
{avg_time_per_user:.6f} seconds


METHODOLOGY
-----------

- Users in user_embeddings.csv are treated as training users.
- Remaining users are treated as unseen test users.
- Test users require at least 5 interactions.
- Each test user uses a chronological 80/20 split.
- Historical 80% is used to generate recommendations.
- Future 20% is used as ground truth.
- Transformer is used only in inference mode.
- Existing recommendation weights are preserved.
- Recency uses historical interactions only.
- Segment-item affinity uses training-user events only.
- Impossible ground-truth items outside the candidate pool are
  not counted in the primary candidate-restricted metrics.
- A popularity-only baseline is evaluated on the same candidate pool.
"""

    with open(
        eval_dir / "evaluation_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary.strip() + "\n")

    # ========================================================
    # 22. FINAL CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("RECOMMENDATION SYSTEM EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total users: "
        f"{len(all_user_ids):,}"
    )

    print(
        f"Training users: "
        f"{len(training_user_ids):,}"
    )

    print(
        f"Unseen test users: "
        f"{len(test_user_ids):,}"
    )

    print(
        f"Eligible test users (>=5): "
        f"{len(eligible_test_users):,}"
    )

    print(
        f"Evaluated users: "
        f"{evaluated_users:,}"
    )

    print(
        f"Candidate-reachable users: "
        f"{reachable_users:,}"
    )

    print(
        f"Candidate reach rate: "
        f"{candidate_reach_rate:.4f}"
    )

    print(
        f"Candidate items: "
        f"{len(candidates):,}"
    )

    print()
    print("PRIMARY MODEL METRICS")
    print(
        f"Hit Rate@5: "
        f"{model_metrics['hit_rate@5']:.4f}"
    )
    print(
        f"Precision@5: "
        f"{model_metrics['precision@5']:.4f}"
    )
    print(
        f"Recall@5: "
        f"{model_metrics['recall@5']:.4f}"
    )
    print(
        f"NDCG@5: "
        f"{model_metrics['ndcg@5']:.4f}"
    )

    print()
    print("POPULARITY BASELINE")
    print(
        f"Hit Rate@5: "
        f"{popularity_metrics['hit_rate@5']:.4f}"
    )
    print(
        f"Precision@5: "
        f"{popularity_metrics['precision@5']:.4f}"
    )
    print(
        f"Recall@5: "
        f"{popularity_metrics['recall@5']:.4f}"
    )
    print(
        f"NDCG@5: "
        f"{popularity_metrics['ndcg@5']:.4f}"
    )

    print()
    print("ALL-USER DIAGNOSTIC")
    print(
        f"Hit Rate@5: "
        f"{all_user_metrics['hit_rate@5']:.4f}"
    )
    print(
        f"NDCG@5: "
        f"{all_user_metrics['ndcg@5']:.4f}"
    )

    print()
    print(
        f"Catalog coverage: "
        f"{catalog_coverage:.4f}"
    )

    print(
        f"Unique recommended items: "
        f"{unique_recommended_items:,}"
    )

    print()
    print(
        f"Evaluation time: "
        f"{evaluation_time:.2f}s"
    )

    print(
        f"Average time/user: "
        f"{avg_time_per_user:.6f}s"
    )

    print()
    print("Outputs:")
    print(
        f"  {eval_dir / 'evaluation_metrics.csv'}"
    )
    print(
        f"  {eval_dir / 'per_user_metrics.csv'}"
    )
    print(
        f"  {eval_dir / 'test_recommendations.csv'}"
    )
    print(
        f"  {eval_dir / 'evaluation_summary.txt'}"
    )

    print("=" * 70)

    total_time = (
        time.perf_counter()
        - total_start
    )

    print(
        f"Total evaluation runtime: "
        f"{total_time:.2f}s"
    )

    print("=" * 70)
    print()


if __name__ == "__main__":
    run_evaluation()
