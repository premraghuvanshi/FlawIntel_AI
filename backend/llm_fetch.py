"""
backend/llm_fetch.py
──────────────────────────────────────────────────────────────────
LLM extraction layer:
  PRIMARY  – concurrent async Groq API (llama3-8b-8192)
  FALLBACK – deterministic local heuristic engine

Both paths return a uniform list of dicts with keys:
  { cluster_id, Feature_Mentioned, Sentiment_Score, Specific_Complaint }
"""

import asyncio
import collections
import json
import re
import string
import traceback
from typing import Any

import nest_asyncio
nest_asyncio.apply()  # Allow asyncio in Streamlit's already-running event loop

try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a product quality analyst AI.
You will receive a batch of negative customer reviews for a single product cluster.
Extract the primary product flaw discussed in these reviews.

Respond ONLY with a valid JSON object — no markdown, no explanation, no preamble.
The JSON must contain exactly three keys:
  "Feature_Mentioned"   : string  – the specific product feature or component being criticised
  "Sentiment_Score"     : float   – a negative sentiment score between -1.0 (extremely negative) and 0.0 (mildly negative)
  "Specific_Complaint"  : string  – one precise sentence summarising the core complaint

Example:
{"Feature_Mentioned":"battery life","Sentiment_Score":-0.82,"Specific_Complaint":"Battery drains completely within 3 hours of moderate use."}
"""

_MAX_REVIEWS_PER_CLUSTER = 15   # truncate to keep token count manageable
_MAX_REVIEW_CHARS = 300         # truncate individual reviews
_GROQ_MODEL = "llama-3.1-8b-instant"
_GROQ_TEMPERATURE = 0.15
_GROQ_MAX_TOKENS = 256
_GROQ_TIMEOUT = 30              # seconds per request


def _sanitize_llm_output(parsed: dict[str, Any], cluster_id: int) -> dict[str, Any]:
    """Ensures all fields are populated and valid to prevent DB IntegrityErrors."""
    
    # 1. Ensure Feature_Mentioned is a valid string
    if not parsed.get("Feature_Mentioned") or not isinstance(parsed.get("Feature_Mentioned"), str):
        parsed["Feature_Mentioned"] = "General Quality"
        
    # 2. Ensure Specific_Complaint is a valid string
    if not parsed.get("Specific_Complaint") or not isinstance(parsed.get("Specific_Complaint"), str):
        parsed["Specific_Complaint"] = "No specific complaint captured."
        
    # 3. Defensive Sentiment Parsing
    raw_score = parsed.get("Sentiment_Score")
    try:
        parsed["Sentiment_Score"] = max(-1.0, min(0.0, float(raw_score)))
    except (ValueError, TypeError):
        parsed["Sentiment_Score"] = -0.5
        
    parsed["cluster_id"] = cluster_id
    return parsed



# ─────────────────────────────────────────────────────────────────
# Primary: Async Groq
# ─────────────────────────────────────────────────────────────────

def _build_user_message(cluster_id: int, reviews: list[str]) -> str:
    """Format reviews for the LLM prompt."""
    sample = reviews[:_MAX_REVIEWS_PER_CLUSTER]
    truncated = [r[:_MAX_REVIEW_CHARS] for r in sample]
    reviews_block = "\n".join(f"- {r}" for r in truncated)
    return (
        f"Cluster {cluster_id} ({len(sample)} negative reviews):\n"
        f"{reviews_block}\n\n"
        "Return the JSON object now."
    )


async def _call_groq_single(
    client,
    cluster_id: int,
    reviews: list[str],
) -> dict[str, Any]:
    """
    Call Groq API for a single cluster and parse the JSON response.
    Raises on any failure so the caller can route to heuristic.
    """
    user_msg = _build_user_message(cluster_id, reviews)

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=_GROQ_TEMPERATURE,
            max_tokens=_GROQ_MAX_TOKENS,
        ),
        timeout=_GROQ_TIMEOUT,
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip any accidental markdown fences
    raw_text = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    parsed = json.loads(raw_text)  # raises json.JSONDecodeError on bad output

    # Validate required keys
    # Validate required keys
    required = {"Feature_Mentioned", "Sentiment_Score", "Specific_Complaint"}
    if not required.issubset(parsed.keys()):
        # Log it and provide default keys so sanitization can fill them in
        print(f"[llm_fetch] LLM missing keys. Filling defaults.")
        for key in required:
            if key not in parsed:
                parsed[key] = None

    # Sanitize data to prevent SQLite IntegrityErrors
    return _sanitize_llm_output(parsed, cluster_id)


async def _extract_all_clusters_async(
    cluster_groups: dict[int, list[str]],
    groq_api_key: str,
) -> list[dict[str, Any]]:
    """
    Launch one Groq call per cluster concurrently.
    Raises if the client cannot be initialised.
    """
    if AsyncGroq is None:
        raise ImportError("groq package is not installed.")

    client = AsyncGroq(api_key=groq_api_key, timeout=_GROQ_TIMEOUT)

    tasks = [
        _call_groq_single(client, cluster_id, reviews)
        for cluster_id, reviews in cluster_groups.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)


def extract_with_llm(
    cluster_groups: dict[int, list[str]],
    groq_api_key: str,
) -> tuple[list[dict[str, Any]], str]:
    """
    Public entry point for LLM extraction.

    Returns
    -------
    (results, engine_label)
        results      – list of extraction dicts (one per cluster)
        engine_label – "LLM" on success, "Heuristic" on fallback
    """
    if not groq_api_key or not groq_api_key.strip():
        print("[llm_fetch] No Groq API key — routing to heuristic.")
        return extract_with_heuristic(cluster_groups), "Heuristic"

    try:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            _extract_all_clusters_async(cluster_groups, groq_api_key.strip())
        )
        print(f"[llm_fetch] LLM extraction complete for {len(results)} clusters.")
        return results, "LLM"

    except Exception:
        traceback.print_exc()
        print("[llm_fetch] Groq failed — falling back to heuristic engine.")
        return extract_with_heuristic(cluster_groups), "Heuristic"


# ─────────────────────────────────────────────────────────────────
# Fallback: Local Heuristic Engine
# ─────────────────────────────────────────────────────────────────

# Common English stopwords to exclude from feature detection
_STOPWORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "it", "is", "was", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "a", "an", "the", "and", "but", "or",
    "nor", "for", "so", "yet", "in", "on", "at", "to", "from", "by",
    "with", "about", "as", "into", "through", "this", "that", "these",
    "those", "of", "not", "no", "very", "just", "also", "so", "if",
    "then", "than", "too", "its", "it's", "when", "where", "which",
    "who", "what", "how", "all", "any", "both", "each", "more", "most",
    "other", "some", "such", "up", "out", "get", "got", "use", "used",
})


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w and w not in _STOPWORDS and len(w) > 2]


def _heuristic_feature(reviews: list[str]) -> str:
    """
    Most frequent non-stopword token across all reviews in this cluster.
    Returns 'general quality' if nothing meaningful is found.
    """
    counter: collections.Counter = collections.Counter()
    for review in reviews:
        tokens = _tokenise(review)
        counter.update(tokens)

    if not counter:
        return "general quality"

    # Deduplicate: take the single most common unique term
    most_common = counter.most_common(1)[0][0]
    return most_common


def _heuristic_complaint(reviews: list[str]) -> str:
    """Longest individual review string as the most descriptive complaint."""
    if not reviews:
        return "No specific complaint captured."
    longest = max(reviews, key=len)
    # Truncate to 300 chars for DB storage
    return longest[:300].strip()


def _heuristic_sentiment(cluster_id: int, n_clusters: int) -> float:
    """
    Mathematically derive a sentiment score from cluster position.
    Clusters are evenly distributed across [-1.0, -0.3].
    cluster_id=0 → most negative, higher ids → less negative.
    """
    if n_clusters <= 1:
        return -0.65
    # Linear interpolation: cluster 0 = -1.0, cluster (n-1) = -0.3
    score = -1.0 + (cluster_id / max(n_clusters - 1, 1)) * 0.7
    return round(max(-1.0, min(-0.1, score)), 4)


def extract_with_heuristic(
    cluster_groups: dict[int, list[str]],
) -> list[dict[str, Any]]:
    """
    Local heuristic extraction — zero API calls, always succeeds.

    Returns the same schema as LLM:
    [{ cluster_id, Feature_Mentioned, Sentiment_Score, Specific_Complaint }]
    """
    n_clusters = len(cluster_groups)
    results = []

    for cluster_id, reviews in cluster_groups.items():
        feature = _heuristic_feature(reviews)
        complaint = _heuristic_complaint(reviews)
        sentiment = _heuristic_sentiment(cluster_id, n_clusters)

        results.append({
            "cluster_id": cluster_id,
            "Feature_Mentioned": feature,
            "Sentiment_Score": sentiment,
            "Specific_Complaint": complaint,
        })
        print(
            f"[heuristic] cluster={cluster_id} → feature='{feature}', "
            f"sentiment={sentiment}, complaint='{complaint[:60]}...'"
        )

    return results
