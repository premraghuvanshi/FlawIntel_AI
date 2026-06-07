"""
backend/apify_fetch.py
──────────────────────────────────────────────────────────────────
Industrial-grade e-commerce review scraping module using Apify.
"""

import re
import traceback
from typing import Any, Tuple, List

try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────────────────────────

_TARGET_FETCH_LIMIT = 150 

_ACTOR_MAP = {
    "amazon": "junglee/amazon-reviews-scraper",
    "flipkart": "epctex/flipkart-reviews-scraper",
    "playstore": "epctex/google-play-scraper",
    "myntra": "dtrush/myntra-scraper",
    "generic": "apify/website-content-crawler",
}


# ─────────────────────────────────────────────────────────────────
# 2. PLATFORM DETECTION & SANITIZATION
# ─────────────────────────────────────────────────────────────────

def _detect_platform(url: str) -> str:
    lowered_url = url.lower()
    if "amazon." in lowered_url: return "amazon"
    elif "flipkart." in lowered_url: return "flipkart"
    elif "play.google.com" in lowered_url: return "playstore"
    elif "myntra." in lowered_url: return "myntra"
    return "generic"


def _sanitize_url(platform: str, url: str) -> str:
    """Extracts ASIN and reconstructs a pristine URL to prevent Actor 'INVALID INPUT' errors."""
    clean_url = url.strip()

    if platform == "amazon":
        # Regex to find Amazon ASIN (10 characters)
        asin_match = re.search(r'/(?:dp|product|product-reviews|gp/product)/([A-Z0-9]{10})', clean_url, re.IGNORECASE)
        # Regex to find the domain
        domain_match = re.search(r'(https?://(?:www\.)?amazon\.[a-z\.]+)', clean_url, re.IGNORECASE)
        
        if asin_match and domain_match:
            asin = asin_match.group(1).upper()
            domain = domain_match.group(1).lower()
            clean_url = f"{domain}/dp/{asin}"
        else:
            clean_url = clean_url.split("?")[0].split("ref=")[0]
            
    elif platform == "flipkart":
        clean_url = clean_url.split("?")[0]
        
    return clean_url


def _build_actor_payload(platform: str, url: str) -> dict:
    """Constructs the JSON payload without filtering (let Pandas filter later)."""
    
    if platform == "amazon":
        return {
            "productUrls": [{"url": url}],
            "maxReviews": _TARGET_FETCH_LIMIT,
            "reviewsSort": "recent",
            "scrapeReviewerProfile": False,
            "extractImages": False,
            "includeHtml": False
        }
    elif platform == "flipkart":
        return {
            "startUrls": [{"url": url}],
            "maxItems": _TARGET_FETCH_LIMIT,
            "maxReviews": _TARGET_FETCH_LIMIT,
            "sort": "mostRecent", 
            "includeHtml": False
        }
    # ... (Keep other platform blocks as you had them)
    return {"startUrls": [{"url": url}], "maxCrawlPages": 5, "crawlerType": "cheerio"}


# ─────────────────────────────────────────────────────────────────
# 3. DATA NORMALISATION & EXECUTION
# ─────────────────────────────────────────────────────────────────

def _normalise_item(item: dict[str, Any], platform: str) -> dict[str, Any]:
    normalised = dict(item)
    # -- Map Text --
    normalised["review_text"] = item.get("reviewDescription") or item.get("text") or item.get("content") or ""
    # -- Map Rating --
    raw_rating = item.get("ratingScore") or item.get("rating") or item.get("score") or item.get("starRating")
    try:
        normalised["star_rating"] = float(str(raw_rating).split()[0]) if raw_rating else None
    except:
        normalised["star_rating"] = None
    return normalised

def fetch_reviews_from_url(url: str, apify_token: str) -> Tuple[List[dict], str]:
    if ApifyClient is None: return [], "apify-client not installed."
    if not apify_token: return [], "Token missing."
    
    platform = _detect_platform(url)
    clean_url = _sanitize_url(platform, url)
    actor_id = _ACTOR_MAP.get(platform, _ACTOR_MAP["generic"])
    run_input = _build_actor_payload(platform, clean_url)

    try:
        client = ApifyClient(apify_token.strip())
        run = client.actor(actor_id).call(run_input=run_input)
        
        if not run or run.get("status") != "SUCCEEDED":
            return [], f"Actor failed with status: {run.get('status')}"

        dataset_id = run.get("defaultDatasetId")
        raw_items = list(client.dataset(dataset_id).iterate_items(limit=_TARGET_FETCH_LIMIT))
        
        return [_normalise_item(item, platform) for item in raw_items], ""
    except Exception as e:
        traceback.print_exc()
        return [], str(e)