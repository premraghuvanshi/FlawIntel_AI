"""
backend/ml_engine.py
──────────────────────────────────────────────────────────────────
Machine Learning engine:
  1. Embed review texts into 384-D vectors with all-MiniLM-L6-v2
  2. Dynamic k selection (k=2..8) using Silhouette Coefficient
  3. Fit final K-Means on optimal k
  4. Return cluster labels + diagnostics
"""

import traceback
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Lazy model cache (shared with data_preprocessing if loaded in same process)
_model_cache: Any = None

# k search bounds
_K_MIN = 5
_K_MAX = 8

# K-Means convergence settings
_KMEANS_INIT = "k-means++"
_KMEANS_N_INIT = 10
_KMEANS_MAX_ITER = 300
_RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────

def _get_model():
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of strings into a (N, 384) float32 numpy array.
    Raises RuntimeError if embedding fails.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    try:
        model = _get_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        print(f"[ml_engine] Embedded {len(texts)} texts → shape {vectors.shape}")
        return vectors.astype(np.float32)
    except Exception:
        traceback.print_exc()
        raise RuntimeError("Sentence-transformer embedding failed. See terminal for details.")


# ─────────────────────────────────────────────────────────────────
# Dynamic k selection
# ─────────────────────────────────────────────────────────────────

def _silhouette_for_k(vectors: np.ndarray, k: int) -> float:
    """
    Fit K-Means with *k* clusters and return the silhouette score.
    Returns -1.0 on any failure (treated as worst score).
    """
    try:
        km = KMeans(
            n_clusters=k,
            init=_KMEANS_INIT,
            n_init=_KMEANS_N_INIT,
            max_iter=_KMEANS_MAX_ITER,
            random_state=_RANDOM_STATE,
        )
        labels = km.fit_predict(vectors)
        # Silhouette is undefined if all points in one cluster
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return -1.0
        score = float(silhouette_score(vectors, labels, metric="cosine", sample_size=None))
        return score
    except Exception:
        traceback.print_exc()
        return -1.0


def select_optimal_k(vectors: np.ndarray) -> tuple[int, float, dict[int, float]]:
    """
    Iterate k from _K_MIN to _K_MAX, compute silhouette for each,
    and return the best k.

    Returns
    -------
    (optimal_k, max_silhouette, score_map)
        optimal_k      – best number of clusters
        max_silhouette – silhouette score at optimal_k
        score_map      – {k: silhouette_score} for all tested k values
    """
    n_samples = vectors.shape[0]

    # k cannot exceed (n_samples - 1) for silhouette to be defined
    k_upper = min(_K_MAX, n_samples - 1)
    k_lower = min(_K_MIN, k_upper)

    if k_lower < 2:
        raise ValueError(
            f"Not enough samples ({n_samples}) to perform clustering. "
            "Need at least 3 data points."
        )

    score_map: dict[int, float] = {}
    for k in range(k_lower, k_upper + 1):
        score = _silhouette_for_k(vectors, k)
        score_map[k] = score
        print(f"[ml_engine] k={k} → silhouette={score:.4f}")

    optimal_k = max(score_map, key=score_map.__getitem__)
    max_silhouette = score_map[optimal_k]

    print(f"[ml_engine] Optimal k={optimal_k} (silhouette={max_silhouette:.4f})")
    return optimal_k, max_silhouette, score_map


# ─────────────────────────────────────────────────────────────────
# Final clustering
# ─────────────────────────────────────────────────────────────────

def cluster_reviews(
    df: pd.DataFrame,
    text_col: str,
) -> tuple[pd.DataFrame, int, float, dict[int, float], np.ndarray]:
    """
    Full ML pipeline:
      1. Extract texts from df[text_col]
      2. Embed with MiniLM
      3. Select optimal k via silhouette
      4. Fit final K-Means
      5. Attach 'cluster_id' column to df copy

    Returns
    -------
    (df_with_clusters, optimal_k, max_silhouette, score_map, vectors)
        df_with_clusters – input df with 'cluster_id' column added
        optimal_k        – chosen number of clusters
        max_silhouette   – silhouette score for chosen k
        score_map        – silhouette scores for all tested k
        vectors          – (N, 384) embedding array
    """
    texts = df[text_col].astype(str).tolist()

    # ── Embed ────────────────────────────────────────────────────
    vectors = embed_texts(texts)

    # ── Select optimal k ─────────────────────────────────────────
    optimal_k, max_silhouette, score_map = select_optimal_k(vectors)

    # ── Fit final model ──────────────────────────────────────────
    try:
        km_final = KMeans(
            n_clusters=optimal_k,
            init=_KMEANS_INIT,
            n_init=_KMEANS_N_INIT,
            max_iter=_KMEANS_MAX_ITER,
            random_state=_RANDOM_STATE,
        )
        labels = km_final.fit_predict(vectors)
    except Exception:
        traceback.print_exc()
        raise RuntimeError("Final K-Means fitting failed. See terminal for details.")

    df_out = df.copy()
    df_out["cluster_id"] = labels.astype(int)

    # Print cluster distribution
    distribution = pd.Series(labels).value_counts().sort_index().to_dict()
    print(f"[ml_engine] Cluster distribution: {distribution}")

    return df_out, optimal_k, max_silhouette, score_map, vectors


# ─────────────────────────────────────────────────────────────────
# Cluster grouping utility
# ─────────────────────────────────────────────────────────────────

def group_by_cluster(df: pd.DataFrame, text_col: str) -> dict[int, list[str]]:
    """
    Return {cluster_id: [list of review strings]} from a df
    that already has a 'cluster_id' column.
    """
    groups: dict[int, list[str]] = {}
    for cluster_id, group_df in df.groupby("cluster_id"):
        groups[int(cluster_id)] = group_df[text_col].astype(str).tolist()
    return groups
