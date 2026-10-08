
import pandas as pd
import numpy as np
import torch

from pathlib import Path
from datetime import timezone

from .config import (
    OUTPUT_DIR,
    RANDOM_SEED,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    USER_ITEM_WEIGHT,
    POPULARITY_WEIGHT,
    SEGMENT_WEIGHT,
    RECENCY_WEIGHT,
    ITEM_EMBEDDING_DIM,
    TOP_K,
    RECENCY_DECAY_DAYS,
)

from sklearn.metrics.pairwise import cosine_similarity

from .transformer_embeddings import load_trained_model
from .dynamic_recommendation import generate_recommendations
from .synthetic_sentiment import generate_synthetic_sentiment
from .clv import compute_clv_proxy
from .customer_segmentation import segment_customers
from .recommendation_logs import generate_recommendation_logs


np.random.seed(RANDOM_SEED)


class RealTimeEngine:
    """
    Lightweight real-time recommendation engine.

    Maintains in-memory state for:
        - user features
        - user event sequences
        - user embeddings
        - user segments
        - item statistics
        - historical item interactions

    When a new event arrives:
        1. Update the user's sequence.
        2. Update behavioural features.
        3. Recalculate the user's Transformer embedding.
        4. Re-segment the user.
        5. Update historical interaction/recency state.
        6. Recalculate recommendation scores using:
             - user-item similarity
             - popularity
             - segment affinity
             - recency
        7. Return fresh Top-K recommendations.
    """

    def __init__(self):

        # ====================================================
        # Paths
        # ====================================================

        self.user_features_path = (
            OUTPUT_DIR / "user_features.csv"
        )

        self.user_sequences_path = (
            OUTPUT_DIR / "user_sequences.csv"
        )

        self.embeddings_path = (
            OUTPUT_DIR / "user_embeddings.csv"
        )

        self.segments_path = (
            OUTPUT_DIR / "customer_segments.csv"
        )

        self.item_stats_path = (
            OUTPUT_DIR / "item_statistics.csv"
        )

        # Historical events
        self.events_path = (
            Path(__file__)
            .resolve()
            .parents[1]
            / "data"
            / "events.csv"
        )

        # ====================================================
        # Load data
        # ====================================================

        self._load_dataframes()

        # ====================================================
        # Load historical interaction state
        # ====================================================

        self.last_interaction_lookup = {}
        self.seg_item_score_lookup = {}

        self._load_historical_interactions()

        # ====================================================
        # Load trained Transformer
        # ====================================================

        try:

            self.model, self.vocab = (
                load_trained_model()
            )

        except Exception as e:

            print(
                f"Failed to load trained model: {e}. "
                f"Model will be loaded lazily when needed."
            )

            self.model = None
            self.vocab = None

    # ========================================================
    # LOAD DATAFRAMES
    # ========================================================

    def _load_dataframes(self):

        self.user_features = (
            pd.read_csv(
                self.user_features_path
            )
            if self.user_features_path.is_file()
            else pd.DataFrame()
        )

        self.user_sequences = (
            pd.read_csv(
                self.user_sequences_path
            )
            if self.user_sequences_path.is_file()
            else pd.DataFrame()
        )

        self.embeddings = (
            pd.read_csv(
                self.embeddings_path
            )
            if self.embeddings_path.is_file()
            else pd.DataFrame()
        )

        self.segments = (
            pd.read_csv(
                self.segments_path
            )
            if self.segments_path.is_file()
            else pd.DataFrame()
        )

        self.item_stats = (
            pd.read_csv(
                self.item_stats_path
            )
            if self.item_stats_path.is_file()
            else pd.DataFrame()
        )

        # ----------------------------------------------------
        # Ensure user-level data has one row per user
        # ----------------------------------------------------

        if not self.user_features.empty:
            self.user_features = (
                self.user_features
                .drop_duplicates(
                    subset=["visitorid"],
                    keep="last"
                )
                .reset_index(drop=True)
            )

        if not self.user_sequences.empty:
            self.user_sequences = (
                self.user_sequences
                .drop_duplicates(
                    subset=["visitorid"],
                    keep="last"
                )
                .reset_index(drop=True)
            )

        if not self.embeddings.empty:
            self.embeddings = (
                self.embeddings
                .drop_duplicates(
                    subset=["visitorid"],
                    keep="last"
                )
                .reset_index(drop=True)
            )

        if not self.segments.empty:
            self.segments = (
                self.segments
                .drop_duplicates(
                    subset=["visitorid"],
                    keep="last"
                )
                .reset_index(drop=True)
            )

    # ========================================================
    # HISTORICAL INTERACTION PROCESSING
    # ========================================================

    def _load_historical_interactions(self):

        """
        Load historical events once and build:

        1. Last interaction:
             visitorid -> itemid -> timestamp

        2. Segment-item affinity:
             (segment, itemid) -> normalized score

        This is used by the real-time recommendation stage.
        """

        if not self.events_path.is_file():

            print(
                "Warning: events.csv not found. "
                "Recency and segment affinity will start at zero."
            )

            return

        try:

            events_df = pd.read_csv(
                self.events_path,
                usecols=[
                    "visitorid",
                    "itemid",
                    "event",
                    "timestamp"
                ]
            )

        except Exception as e:

            print(
                f"Warning: failed to load historical events: {e}"
            )

            return

        if events_df.empty:
            return

        print(
            f"Loading historical real-time state: "
            f"{len(events_df):,} events..."
        )

        # ----------------------------------------------------
        # Clean IDs
        # ----------------------------------------------------

        events_df["visitorid"] = (
            events_df["visitorid"]
            .astype(str)
            .str.replace(
                r"tensor\((\d+)\)",
                r"\1",
                regex=True
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
            "view": 1.0,
            "addtocart": 3.0,
            "transaction": 5.0
        }

        events_df["weight"] = (
            events_df["event"]
            .map(event_weights)
            .fillna(0)
            .astype(np.float32)
        )

        # ====================================================
        # LAST INTERACTION
        # ====================================================

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

        for (
            visitor,
            itemid,
            timestamp
        ) in last_ts_df.itertuples(
            index=False
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

            visitor = str(visitor)
            itemid = str(itemid)

            if visitor not in self.last_interaction_lookup:

                self.last_interaction_lookup[
                    visitor
                ] = {}

            self.last_interaction_lookup[
                visitor
            ][itemid] = timestamp_value

        # ====================================================
        # SEGMENT-ITEM AFFINITY
        # ====================================================

        if (
            not self.segments.empty
            and "visitorid" in self.segments.columns
            and "segment" in self.segments.columns
        ):

            visitor_segment_map = (
                self.segments[
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

            visitor_segment_map.index = (
                visitor_segment_map.index
                .astype(str)
            )

            events_df["segment"] = (
                events_df["visitorid"]
                .map(visitor_segment_map)
            )

            seg_item_agg = (
                events_df
                .dropna(
                    subset=["segment"]
                )
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

                self.seg_item_score_lookup = {
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

        del last_ts_df
        del events_df

    # ========================================================
    # UPDATE HISTORICAL STATE FOR NEW EVENT
    # ========================================================

    def _update_realtime_history(
        self,
        event: dict,
        segment: int
    ):

        visitor = str(
            event["visitorid"]
        )

        itemid = str(
            event["itemid"]
        )

        try:

            timestamp = int(
                event["timestamp"]
            )

        except (
            TypeError,
            ValueError
        ):

            timestamp = 0

        # ----------------------------------------------------
        # Update last interaction
        # ----------------------------------------------------

        if visitor not in self.last_interaction_lookup:

            self.last_interaction_lookup[
                visitor
            ] = {}

        previous_timestamp = (
            self.last_interaction_lookup[
                visitor
            ].get(
                itemid,
                0
            )
        )

        self.last_interaction_lookup[
            visitor
        ][itemid] = max(
            previous_timestamp,
            timestamp
        )

        # ----------------------------------------------------
        # Update segment-item affinity
        # ----------------------------------------------------

        event_weights = {
            "view": 1.0,
            "addtocart": 3.0,
            "transaction": 5.0
        }

        event_type = (
            str(event["event"])
            .lower()
        )

        weight = event_weights.get(
            event_type,
            0.0
        )

        key = (
            int(segment),
            itemid
        )

        old_score = (
            self.seg_item_score_lookup
            .get(key, 0.0)
        )

        # Add the new interaction weight.
        #
        # We cap the value at 1.0 because the lookup
        # represents a normalized affinity score.

        self.seg_item_score_lookup[key] = min(
            1.0,
            old_score + (
                weight / 100.0
            )
        )

    # ========================================================
    # PROCESS REAL-TIME EVENT
    # ========================================================

    def process_realtime_event(
        self,
        event: dict
    ):

        """
        Process a single behavioural event.

        Expected keys:

            visitorid
            event
            itemid
            timestamp

        Optional:

            transactionid
        """

        visitor = str(
            event["visitorid"]
        )

        # ====================================================
        # 1. UPDATE USER SEQUENCE
        # ====================================================

        seq_token = (
            f"{event['event']}"
            f"_item_"
            f"{event['itemid']}"
        )

        if (
            not self.user_sequences.empty
            and visitor in
            self.user_sequences[
                "visitorid"
            ].astype(str).values
        ):

            idx = (
                self.user_sequences[
                    self.user_sequences[
                        "visitorid"
                    ].astype(str) == visitor
                ]
                .index[0]
            )

            cur_seq = (
                self.user_sequences
                .at[
                    idx,
                    "event_sequence"
                ]
            )

            new_seq = (
                f"{cur_seq} {seq_token}"
                if pd.notna(cur_seq)
                else seq_token
            )

            self.user_sequences.at[
                idx,
                "event_sequence"
            ] = new_seq

        else:

            new_row = pd.DataFrame([
                {
                    "visitorid": visitor,
                    "event_sequence": seq_token
                }
            ])

            self.user_sequences = pd.concat(
                [
                    self.user_sequences,
                    new_row
                ],
                ignore_index=True
            )

        # ====================================================
        # 2. UPDATE USER FEATURES
        # ====================================================

        if (
            self.user_features.empty
            or visitor not in
            self.user_features[
                "visitorid"
            ].astype(str).values
        ):

            new_row = {
                "visitorid": visitor,
                "view": 0,
                "addtocart": 0,
                "transaction": 0,
                "unique_items_viewed": 0,
                "unique_items_purchased": 0,
                "purchase_frequency": 0.0,
                "cart_conversion_rate": 0.0,
                "purchase_conversion_rate": 0.0,
                "recency_days": 0,
                "total_events": 0,
                "interaction_frequency": 0.0,
            }

            self.user_features = pd.concat(
                [
                    self.user_features,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

        # ----------------------------------------------------
        # Increment event counter
        # ----------------------------------------------------

        ev_type = (
            str(event["event"])
            .lower()
        )

        mask = (
            self.user_features[
                "visitorid"
            ].astype(str) == visitor
        )

        if ev_type == "view":

            self.user_features.loc[
                mask,
                "view"
            ] += 1

        elif ev_type == "addtocart":

            self.user_features.loc[
                mask,
                "addtocart"
            ] += 1

        elif ev_type == "transaction":

            self.user_features.loc[
                mask,
                "transaction"
            ] += 1

        # ----------------------------------------------------
        # Derived metrics
        # ----------------------------------------------------

        u = self.user_features.loc[
            mask
        ]

        view = float(
            u["view"].iloc[0]
        )

        add = float(
            u["addtocart"].iloc[0]
        )

        trans = float(
            u["transaction"].iloc[0]
        )

        self.user_features.loc[
            mask,
            "purchase_frequency"
        ] = trans / (
            view + 1e-6
        )

        self.user_features.loc[
            mask,
            "cart_conversion_rate"
        ] = add / (
            view + 1e-6
        )

        self.user_features.loc[
            mask,
            "purchase_conversion_rate"
        ] = trans / (
            add + 1e-6
        )

        # ----------------------------------------------------
        # Recency
        # ----------------------------------------------------

        event_time = pd.to_datetime(
            event["timestamp"],
            unit="ms",
            utc=True
        )

        now = pd.Timestamp.now(
            tz=timezone.utc
        )

        recency = max(
            0,
            (now - event_time).days
        )

        self.user_features.loc[
            mask,
            "recency_days"
        ] = recency

        # ----------------------------------------------------
        # Interaction frequency
        # ----------------------------------------------------

        total = (
            float(
                self.user_features.loc[
                    mask,
                    "total_events"
                ].iloc[0]
            )
            + 1
        )

        self.user_features.loc[
            mask,
            "total_events"
        ] = total

        self.user_features.loc[
            mask,
            "interaction_frequency"
        ] = total / max(
            recency,
            1
        )

        # ====================================================
        # 3. REGENERATE SENTIMENT + CLV
        # ====================================================

        generate_synthetic_sentiment()
        compute_clv_proxy()

        # ====================================================
        # 4. UPDATE USER EMBEDDING
        # ====================================================

        if (
            self.model is None
            or self.vocab is None
        ):

            self.model, self.vocab = (
                load_trained_model()
            )

        seq_str = (
            self.user_sequences[
                self.user_sequences[
                    "visitorid"
                ].astype(str) == visitor
            ][
                "event_sequence"
            ]
            .iloc[0]
        )

        token_ids = [
            self.vocab.encode(tok)
            for tok in seq_str.split()
        ]

        if len(token_ids) < MAX_SEQUENCE_LENGTH:

            token_ids += [
                self.vocab.encode("<PAD>")
            ] * (
                MAX_SEQUENCE_LENGTH
                - len(token_ids)
            )

        else:

            token_ids = token_ids[
                :MAX_SEQUENCE_LENGTH
            ]

        input_ids = torch.tensor(
            [token_ids[:-1]],
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Forward pass only.
        # NO RETRAINING.
        # ----------------------------------------------------

        with torch.no_grad():

            user_emb = (
                self.model
                .get_user_embedding(
                    input_ids
                )
                .cpu()
                .numpy()
                .flatten()
            )

        # ----------------------------------------------------
        # Store updated embedding
        # ----------------------------------------------------

        embedding_mask = (
            self.embeddings[
                "visitorid"
            ].astype(str) == visitor
        )

        embedding_cols = [
            f"emb_{i}"
            for i in range(
                EMBEDDING_DIM
            )
        ]

        if (
            not self.embeddings.empty
            and embedding_mask.any()
        ):

            self.embeddings.loc[
                embedding_mask,
                embedding_cols
            ] = user_emb

        else:

            new_emb = {
                "visitorid": visitor
            }

            new_emb.update({
                f"emb_{i}": user_emb[i]
                for i in range(
                    EMBEDDING_DIM
                )
            })

            self.embeddings = pd.concat(
                [
                    self.embeddings,
                    pd.DataFrame([new_emb])
                ],
                ignore_index=True
            )

        # ====================================================
        # 5. RE-SEGMENT USER
        # ====================================================

        segment_customers()

        # Reload segments because
        # segment_customers() updates the CSV.

        if self.segments_path.is_file():

            self.segments = pd.read_csv(
                self.segments_path
            )

            self.segments = (
                self.segments
                .drop_duplicates(
                    subset=["visitorid"],
                    keep="last"
                )
                .reset_index(drop=True)
            )

        # Determine current segment

        seg_series = self.segments[
            self.segments[
                "visitorid"
            ].astype(str) == visitor
        ]["segment"]

        segment_val = (
            int(
                seg_series.iloc[0]
            )
            if not seg_series.empty
            else 0
        )

        # ====================================================
        # 6. UPDATE REAL-TIME HISTORY
        # ====================================================

        self._update_realtime_history(
            event,
            segment_val
        )

        # ====================================================
        # 7. LOAD ITEM EMBEDDINGS
        # ====================================================

        item_emb_path = (
            OUTPUT_DIR /
            "item_embeddings.csv"
        )

        if not item_emb_path.exists():

            raise FileNotFoundError(
                f"Item embeddings not found at "
                f"{item_emb_path}. "
                f"Ensure training has run."
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
            for i in range(
                ITEM_EMBEDDING_DIM
            )
        ]

        if not set(
            embedding_cols_item
        ).issubset(
            item_embeddings.columns
        ):

            raise ValueError(
                "Item embedding dimensions mismatch."
            )

        item_embeddings = (
            item_embeddings[
                embedding_cols_item
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .fillna(0)
        )

        # ====================================================
        # 8. ITEM POPULARITY
        # ====================================================

        self.item_stats["itemid"] = (
            self.item_stats["itemid"]
            .astype(str)
        )

        item_pop = (
            self.item_stats
            .set_index("itemid")["view"]
            .fillna(0)
        )

        # ====================================================
        # 9. CANDIDATE POOL
        # ====================================================

        valid_items = [
            itm
            for itm in
            item_pop
            .sort_values(
                ascending=False
            )
            .index
            if itm in item_embeddings.index
        ]

        candidates = valid_items[:500]

        if not candidates:

            print(
                f"No candidates available for "
                f"recommendations for user {visitor}."
            )

            return

        # ====================================================
        # 10. USER-ITEM SIMILARITY
        # ====================================================

        user_vec = (
            user_emb
            .reshape(1, -1)
        )

        cand_embs = (
            item_embeddings
            .loc[candidates]
            .values
            .astype(np.float32)
        )

        similarities = (
            cosine_similarity(
                user_vec,
                cand_embs
            )
            .flatten()
            .astype(np.float32)
        )

        # ====================================================
        # 11. POPULARITY SCORE
        # ====================================================

        pop_vals = np.log1p(
            item_pop
            .reindex(candidates)
            .fillna(0)
            .values
        ).astype(np.float32)

        max_pop = (
            pop_vals.max()
            if pop_vals.size > 0
            else 0
        )

        if max_pop > 0:

            norm_pop = (
                pop_vals / max_pop
            )

        else:

            norm_pop = np.zeros_like(
                pop_vals
            )

        # ====================================================
        # 12. SEGMENT AFFINITY SCORE
        # ====================================================

        seg_scores = np.fromiter(
            (
                self.seg_item_score_lookup.get(
                    (
                        int(segment_val),
                        str(item)
                    ),
                    0.0
                )
                for item in candidates
            ),
            dtype=np.float32,
            count=len(candidates)
        )

        # ====================================================
        # 13. RECENCY SCORE
        # ====================================================

        user_history = (
            self.last_interaction_lookup.get(
                visitor,
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

            current_ts_ms = int(
                pd.Timestamp
                .now(
                    tz=timezone.utc
                )
                .timestamp()
                * 1000
            )

            decay_ms = (
                RECENCY_DECAY_DAYS
                * 86400000
            )

            if decay_ms > 0:

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
                ).astype(
                    np.float32
                )

        else:

            recency_scores = np.zeros(
                len(candidates),
                dtype=np.float32
            )

        # ====================================================
        # 14. FINAL DYNAMIC SCORE
        # ====================================================

        scores = (
            USER_ITEM_WEIGHT
            * similarities

            + POPULARITY_WEIGHT
            * norm_pop

            + SEGMENT_WEIGHT
            * seg_scores

            + RECENCY_WEIGHT
            * recency_scores
        )

        # ====================================================
        # 15. TOP-K
        # ====================================================

        top_idx = np.argsort(
            -scores
        )[:TOP_K]

        # ====================================================
        # 16. BUILD RECOMMENDATIONS
        # ====================================================

        recs = []

        for rank, idx in enumerate(
            top_idx,
            start=1
        ):

            item_id = candidates[idx]

            recs.append({
                "visitorid": visitor,
                "itemid": item_id,
                "score": float(
                    scores[idx]
                ),
                "segment": segment_val,
                "recommendation_rank": rank,
                "similarity_score": float(
                    similarities[idx]
                ),
                "popularity_score": float(
                    norm_pop[idx]
                ),
                "segment_score": float(
                    seg_scores[idx]
                ),
                "recency_score": float(
                    recency_scores[idx]
                ),
            })

        recs_df = pd.DataFrame(
            recs
        )

        # ====================================================
        # 17. LOG RECOMMENDATIONS
        # ====================================================

        logs_path = (
            OUTPUT_DIR /
            "recommendation_logs.csv"
        )

        if not logs_path.is_file():

            recs_df.to_csv(
                logs_path,
                index=False
            )

        else:

            recs_df.to_csv(
                logs_path,
                mode="a",
                header=False,
                index=False
            )

        # ====================================================
        # 18. PRINT UPDATED RECOMMENDATIONS
        # ====================================================

        print(
            f"Updated recommendations for user "
            f"{visitor} (segment {segment_val}):"
        )

        print(
            recs_df[
                [
                    "itemid",
                    "score",
                    "recommendation_rank"
                ]
            ].head()
        )

        # Optional diagnostic output so you can actually
        # see that segment and recency are contributing.

        print(
            "Top recommendation score components:"
        )

        print(
            recs_df[
                [
                    "itemid",
                    "similarity_score",
                    "popularity_score",
                    "segment_score",
                    "recency_score"
                ]
            ].head()
        )

        # ====================================================
        # 19. PERSIST UPDATED REAL-TIME ARTEFACTS
        # ====================================================

        self.user_features.to_csv(
            self.user_features_path,
            index=False
        )

        self.user_sequences.to_csv(
            self.user_sequences_path,
            index=False
        )

        self.embeddings.to_csv(
            self.embeddings_path,
            index=False
        )

        self.segments.to_csv(
            self.segments_path,
            index=False
        )

        # item_stats are not altered by
        # the real-time flow.
