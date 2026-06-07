"""
backend/data_preprocessing.py
──────────────────────────────────────────────────────────────────
Handles:
  1. CSV upload → DataFrame
  2. Apify JSON list → DataFrame
  3. Aggressive Column Cleaning – drops nulls, spaces, and constant values
  4. AI Column Mapper  – uses all-MiniLM-L6-v2 + cosine similarity
                         to identify 'text' and 'rating' columns
  5. Rating filter     – keep rows where rating <= 3
  6. Hard cap          – truncate to exactly 80 rows
"""

import traceback
from typing import Any

import numpy as np
import pandas as pd

# Lazy-import sentence_transformers to avoid startup cost
_model_cache: Any = None


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _get_embedding_model():
    """Load MiniLM once and cache it in the module namespace."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache


def _cosine_similarity_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────
# AI Column Mapper
# ─────────────────────────────────────────────────────────────────

# Target concept strings used as semantic anchors
_TEXT_CONCEPT = "written customer review opinion feedback comment text description"
_RATING_CONCEPT = "numeric star rating score grade evaluation"

# Minimum confidence to accept a column mapping
_MIN_TEXT_CONFIDENCE = 0.25
_MIN_RATING_CONFIDENCE = 0.20


def map_columns(df: pd.DataFrame) -> tuple[str | None, str | None, float, float]:
    """
    Use MiniLM embeddings + cosine similarity to identify which
    column in *df* corresponds to review text and which to rating.
    """
    columns = list(df.columns)
    if not columns:
        return None, None, 0.0, 0.0

    try:
        model = _get_embedding_model()

        # Embed column names
        col_embeddings = model.encode(columns, convert_to_numpy=True)  # (N, 384)

        # Embed target concepts
        text_vec = model.encode([_TEXT_CONCEPT], convert_to_numpy=True)[0]    # (384,)
        rating_vec = model.encode([_RATING_CONCEPT], convert_to_numpy=True)[0]  # (384,)

        # Score every column against each concept
        text_scores = [_cosine_similarity_1d(col_embeddings[i], text_vec) for i in range(len(columns))]
        rating_scores = [_cosine_similarity_1d(col_embeddings[i], rating_vec) for i in range(len(columns))]

        best_text_idx = int(np.argmax(text_scores))
        best_rating_idx = int(np.argmax(rating_scores))

        text_col = columns[best_text_idx] if text_scores[best_text_idx] >= _MIN_TEXT_CONFIDENCE else None
        rating_col = columns[best_rating_idx] if rating_scores[best_rating_idx] >= _MIN_RATING_CONFIDENCE else None

        # Avoid mapping both concepts to the same column
        if text_col is not None and rating_col is not None and text_col == rating_col:
            # Drop the lower-confidence one
            if text_scores[best_text_idx] >= rating_scores[best_rating_idx]:
                rating_col = None
            else:
                text_col = None

        text_conf = text_scores[best_text_idx] if text_col else 0.0
        rating_conf = rating_scores[best_rating_idx] if rating_col else 0.0

        print(
            f"[column_mapper] text='{text_col}' ({text_conf:.3f}), "
            f"rating='{rating_col}' ({rating_conf:.3f})"
        )
        return text_col, rating_col, text_conf, rating_conf

    except Exception:
        traceback.print_exc()
        return None, None, 0.0, 0.0



# ─────────────────────────────────────────────────────────────────
# check column those are link or urls 
# ─────────────────────────────────────────────────────────────────

import re

def _is_link_series(series: pd.Series) -> bool:
    """
    Return True if most non-null values in the Series look like URLs.
    """
    # Simple regex for http/https links
    url_pattern = re.compile(r'^(https?://|www\.)', re.IGNORECASE)
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    # Count how many values match the URL pattern
    link_count = non_null.str.match(url_pattern).sum()
    # If >80% of non-null values are links, treat column as a link column
    return link_count / len(non_null) > 0.8


# ─────────────────────────────────────────────────────────────────
# CSV ingestion
# ─────────────────────────────────────────────────────────────────

def load_csv(file_obj) -> tuple[pd.DataFrame, str]:
    """Read a CSV upload into a DataFrame."""
    try:
        df = pd.read_csv(file_obj, encoding="utf-8", on_bad_lines="skip")
        df.columns = [str(c).strip() for c in df.columns]
        print(f"[load_csv] Loaded {len(df)} rows, columns: {list(df.columns)}")
        return df, ""
    except UnicodeDecodeError:
        try:
            import io
            file_obj.seek(0)
            raw = file_obj.read()
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
            df.columns = [str(c).strip() for c in df.columns]
            return df, ""
        except Exception:
            traceback.print_exc()
            return pd.DataFrame(), "CSV decoding failed (tried UTF-8 and Latin-1)."
    except Exception:
        traceback.print_exc()
        return pd.DataFrame(), "Failed to parse CSV. Ensure it is a valid comma-separated file."


# ─────────────────────────────────────────────────────────────────
# Apify JSON list → DataFrame
# ─────────────────────────────────────────────────────────────────

def apify_to_dataframe(items: list[dict[str, Any]]) -> tuple[pd.DataFrame, str]:
    """Convert the raw list of dicts from apify_fetch into a DataFrame."""
    if not items:
        return pd.DataFrame(), "Apify returned no items to convert."
    try:
        df = pd.json_normalize(items)
        df.columns = [str(c).strip() for c in df.columns]
        print(f"[apify_to_df] Converted {len(df)} rows, columns: {list(df.columns)}")
        return df, ""
    except Exception:
        traceback.print_exc()
        return pd.DataFrame(), "Failed to normalise Apify JSON into a DataFrame."


# ─────────────────────────────────────────────────────────────────
# Pipeline: filter + cap
# ─────────────────────────────────────────────────────────────────

_ROW_CAP = 80


def preprocess_pipeline(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, int, int, int, float, float]:
    """
    Full preprocessing pipeline.
    """
    raw_rows = len(df)
    
   # ── Step 0: Aggressive Data Cleaning ─────────────────────────
    cols_before = set(df.columns)
    
    # A. Convert pure whitespace strings (e.g., "   ") to proper NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # B. Convert literal string representations of nulls to proper NaN (case-insensitive)
    df = df.replace(to_replace=r'(?i)^(nan|none|null)$', value=np.nan, regex=True)
    
    # C. Drop columns that are entirely NaN/None
    df = df.dropna(axis=1, how='all')
    
    # D. Drop columns with exactly 1 unique value across all rows (zero variance)
    # FIX: We convert everything to strings just for the uniqueness check 
    # to avoid the "unhashable type: list" crash on columns like 'reviewImages'.
    nunique = df.astype(str).nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index
    df = df.drop(columns=constant_cols)
    
    cols_after = set(df.columns)
    dropped_cols = cols_before - cols_after

    # E. Drop columns that are mostly links (URLs)
    link_cols = [col for col in df.columns if _is_link_series(df[col])]
    if link_cols:
       df = df.drop(columns=link_cols)
       print(f"[pipeline] Dropped {len(link_cols)} link columns: {link_cols}")

    
    if dropped_cols:
        print(f"[pipeline] Dropped {len(dropped_cols)} useless columns: {dropped_cols}")

    # ── Step 1: AI column mapping ────────────────────────────────
    text_col, rating_col, text_conf, rating_conf = map_columns(df)

    if text_col is None:
        raise ValueError(
            "AI Column Mapper could not identify a review-text column. "
            "Ensure the dataset contains a text/comment/review column."
        )

    # ── Step 2: Coerce rating to numeric ─────────────────────────
    if rating_col is not None:
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")

    # ── Step 3: Filter by rating <= 3 ────────────────────────────
    if rating_col is not None:
        pre_filter = len(df)
        df = df[df[rating_col].notna() & (df[rating_col] <= 3.0)].copy()
        print(f"[pipeline] Rating filter: {pre_filter} → {len(df)} rows (rating ≤ 3)")
    else:
        print("[pipeline] No rating column found; skipping rating filter.")

    # ── Step 4: Drop empty text rows ─────────────────────────────
    df = df[df[text_col].notna()].copy()
    df = df[df[text_col].astype(str).str.strip() != ""].copy()

    filtered_rows = len(df)

    if filtered_rows == 0:
        raise ValueError(
            "Zero rows remain. The dataset either contained only high-rating reviews, "
            "or the negative reviews had no written text."
        )

    # ── Step 5: Hard cap ─────────────────────────────────────────
    if len(df) > _ROW_CAP:
        df = df.head(_ROW_CAP).copy()

    capped_rows = len(df)

    # Ensure the text column is a clean string Series
    df[text_col] = df[text_col].astype(str).str.strip()

    print(
        f"[pipeline] raw={raw_rows} → filtered={filtered_rows} → capped={capped_rows} | "
        f"text_col='{text_col}', rating_col='{rating_col}'"
    )

    return df, text_col, rating_col, raw_rows, filtered_rows, capped_rows, text_conf, rating_conf