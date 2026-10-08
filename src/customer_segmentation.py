import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR, N_CLUSTERS, RANDOM_SEED, EMBEDDING_DIM

def load_embeddings() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "user_embeddings.csv")

def load_clv() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "customer_value.csv")

def segment_customers() -> None:
    # Load data
    df_emb = load_embeddings()
    df_clv = load_clv()
    # Merge on visitorid
    df_emb["visitorid"] = df_emb["visitorid"].astype(str)
    df_clv["visitorid"] = df_clv["visitorid"].astype(str)

    df = df_emb.merge(
        df_clv,
        on="visitorid",
        how="left"
    )
    # Use embeddings + CLV proxy as features
    feature_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)] + ["CLV_proxy"]
    X = df[feature_cols].fillna(0).values

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init="auto")
    labels = kmeans.fit_predict(X)
    df["segment"] = labels
    # Save segments
    df[["visitorid", "segment"]].to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    # Compute statistics per segment
    stats = []
    for seg in range(N_CLUSTERS):
        seg_df = df[df["segment"] == seg]
        stats.append({
            "segment": seg,
            "size": len(seg_df),
            "avg_CLV_proxy": seg_df["CLV_proxy"].mean(),
            **{f"avg_emb_{i}": seg_df[f"emb_{i}"].mean() for i in range(EMBEDDING_DIM)}
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUTPUT_DIR / "segment_statistics.csv", index=False)

    # Visualize segments using first two principal components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    plt.title("Customer Segments (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Segment")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_segments.png")
    plt.close()

if __name__ == "__main__":
    segment_customers()
