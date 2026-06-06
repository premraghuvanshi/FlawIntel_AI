"""
backend/apify_fetch.py
──────────────────────────────────────────────────────────────────
Live e-commerce review scraping via Apify cloud actors.
Supports Amazon and generic review URLs.
Returns a raw list of dicts that data_preprocessing.py normalises.
"""

import traceback
from typing import Any

try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None  # type: ignore[assignment]


# ── Actor IDs for popular review scrapers on the Apify marketplace ──
_ACTOR_MAP = {
    "amazon": "junglee/amazon-reviews-scraper",
    "generic": "apify/website-content-crawler",
}

# Maximum items to pull from Apify per run (buffer before the 80-row cap)
_APIFY_MAX_ITEMS = 200


def _detect_actor(url: str) -> str:
    """Choose the best actor based on the domain in the URL."""
    lowered = url.lower()
    if "amazon." in lowered:
        return _ACTOR_MAP["amazon"]
    return _ACTOR_MAP["generic"]


def _normalise_item(item: dict[str, Any], actor: str) -> dict[str, Any]:
    """
    Translate actor-specific field names into a uniform shape:
      { "review_text": str, "star_rating": float | None, ... raw fields ... }
    """
    normalised = dict(item)  # keep all raw fields for the AI column mapper

    if "amazon" in actor:
        # junglee/amazon-reviews-scraper field names
        normalised.setdefault("review_text", item.get("text", item.get("reviewText", "")))
        raw_rating = item.get("rating", item.get("starRating", None))
    else:
        # apify/website-content-crawler is generic; no guaranteed review fields
        normalised.setdefault("review_text", item.get("text", item.get("content", "")))
        raw_rating = item.get("rating", None)

    # Coerce rating to float or None
    try:
        normalised["star_rating"] = float(str(raw_rating).split()[0]) if raw_rating is not None else None
    except (ValueError, AttributeError):
        normalised["star_rating"] = None

    return normalised


def fetch_reviews_from_url(url: str, apify_token: str) -> tuple[list[dict], str]:
    """
    Scrape reviews from a given e-commerce URL using Apify.

    Parameters
    ----------
    url : str
        The product / review page URL.
    apify_token : str
        Apify API token supplied by the user at runtime.

    Returns
    -------
    (items, error_message)
        items        – list of normalised dicts (empty on failure)
        error_message – empty string on success, descriptive string on failure
    """
    if ApifyClient is None:
        return [], "apify-client package is not installed."

    if not apify_token or not apify_token.strip():
        return [], "Apify API token is required for URL-based ingestion."

    if not url or not url.strip().startswith("http"):
        return [], f"Invalid URL: '{url}'. Must start with http/https."

    actor_id = _detect_actor(url)

    try:
        client = ApifyClient(apify_token.strip())

        # Build run input based on actor type
        if "amazon" in actor_id:
            run_input = {
                "productUrls": [{"url": url}],
                "maxReviews": _APIFY_MAX_ITEMS,
                "reviewsSort": "recent",
            }
        else:
            run_input = {
                "startUrls": [{"url": url}],
                "maxCrawlPages": 5,
                "crawlerType": "cheerio",
            }

        print(f"[apify_fetch] Starting actor '{actor_id}' for URL: {url}")
        run = client.actor(actor_id).call(run_input=run_input)

        if run is None or run.get("status") not in ("SUCCEEDED", "FINISHED"):
            status = run.get("status") if run else "NO_RESPONSE"
            return [], f"Apify actor finished with status: {status}."

        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            return [], "Apify run succeeded but returned no dataset ID."

        raw_items = list(
            client.dataset(dataset_id).iterate_items(limit=_APIFY_MAX_ITEMS)
        )

        if not raw_items:
            return [], "Apify returned an empty dataset. The URL may not contain review data."

        normalised = [_normalise_item(item, actor_id) for item in raw_items]
        print(f"[apify_fetch] Retrieved {len(normalised)} items from Apify.")
        return normalised, ""

    except Exception:
        traceback.print_exc()
        return [], "Apify fetch failed. Check token validity and URL accessibility."
