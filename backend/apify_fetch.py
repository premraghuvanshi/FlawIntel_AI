"""
backend/apify_fetch.py
──────────────────────────────────────────────────────────────────
Industrial-grade e-commerce review scraping module using Apify.

Responsibilities:
  1. Detect Target Platform: Identifies the e-commerce domain from the URL.
  2. Payload Construction: Builds platform-specific JSON configurations 
     (enforcing negative reviews, recent sorting, and privacy filters).
  3. API Orchestration: Manages the synchronous execution of Apify actors.
  4. Data Normalisation: Translates varied JSON responses into a uniform schema.
"""

import traceback
from typing import Any, Tuple, List

try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────────────────────────

# Target fetch limit. We request slightly more to ensure ~70 survive filtering
_TARGET_FETCH_LIMIT = 100 

# Map internal platform names to established Apify Actors
_ACTOR_MAP = {
    "amazon": "junglee/amazon-reviews-scraper",
    "flipkart": "epctex/flipkart-reviews-scraper",
    "playstore": "epctex/google-play-scraper",
    "myntra": "dtrush/myntra-scraper",
    "generic": "apify/website-content-crawler",
}


# ─────────────────────────────────────────────────────────────────
# 2. PLATFORM DETECTION & ROUTING
# ─────────────────────────────────────────────────────────────────

def _detect_platform(url: str) -> str:
    """
    Analyzes the URL domain to route the request to the correct Apify actor.
    """
    lowered_url = url.lower()
    
    if "amazon." in lowered_url:
        return "amazon"
    elif "flipkart." in lowered_url:
        return "flipkart"
    elif "play.google.com" in lowered_url:
        return "playstore"
    elif "myntra." in lowered_url:
        return "myntra"
    
    return "generic"


def _build_actor_payload(platform: str, url: str) -> dict:
    """
    Constructs the specific JSON payload required for the chosen platform's Actor.
    Enforces strict rules: Recent sort, negative ratings (1-3), and minimal metadata.
    """
    
    if platform == "amazon":
        return {
            "productUrls": [{"url": url}],
            "maxReviews": _TARGET_FETCH_LIMIT,
            "reviewsSort": "recent",
            "filterByStar": ["one_star", "two_star", "three_star"],
            "scrapeReviewerProfile": False, # Privacy / Speed
            "extractImages": False,
            "includeHtml": False
        }
        
    elif platform == "flipkart":
        return {
            "startUrls": [{"url": url}],
            "maxItems": _TARGET_FETCH_LIMIT,
            "sort": "mostRecent", 
            "rating": [1, 2, 3], 
            "includeHtml": False
        }
        
    elif platform == "playstore":
        return {
            "startUrls": [{"url": url}],
            "maxReviews": _TARGET_FETCH_LIMIT,
            "sort": "NEWEST", 
            "stars": [1, 2, 3],
        }
        
    elif platform == "myntra":
         return {
            "startUrls": [{"url": url}],
            "maxItems": _TARGET_FETCH_LIMIT,
        }
         
    # Fallback for Generic Crawler
    return {
        "startUrls": [{"url": url}],
        "maxCrawlPages": 5,
        "crawlerType": "cheerio",
    }


# ─────────────────────────────────────────────────────────────────
# 3. DATA NORMALISATION PIPELINE
# ─────────────────────────────────────────────────────────────────

def _normalise_item(item: dict[str, Any], platform: str) -> dict[str, Any]:
    """
    Translates wild JSON field names from different actors into a uniform schema.
    Guarantees 'review_text' and 'star_rating' exist for downstream ML tasks.
    """
    normalised = dict(item)  # Preserve all raw fields for safety/inspection

    # -- Map Text Fields --
    if platform == "amazon":
        normalised["review_text"] = item.get("text", item.get("reviewText", ""))
    elif platform == "flipkart":
        normalised["review_text"] = item.get("reviewDescription", item.get("text", ""))
    elif platform == "playstore":
        normalised["review_text"] = item.get("text", item.get("content", ""))
    else:
        normalised["review_text"] = item.get("text", item.get("content", ""))

    # -- Map Rating Fields --
    raw_rating = None
    if platform == "amazon":
        raw_rating = item.get("rating", item.get("starRating", None))
    elif platform == "flipkart":
         raw_rating = item.get("ratingScore", item.get("rating", None))
    elif platform == "playstore":
         raw_rating = item.get("score", item.get("rating", None))
    else:
         raw_rating = item.get("rating", None)

    # -- Strict Type Coercion for Ratings --
    try:
        if raw_rating is not None:
            # Handles string formats like "3.5 out of 5 stars" -> 3.5
            normalised["star_rating"] = float(str(raw_rating).split()[0])
        else:
            normalised["star_rating"] = None
    except (ValueError, AttributeError, IndexError):
        normalised["star_rating"] = None

    return normalised


# ─────────────────────────────────────────────────────────────────
# 4. MAIN EXECUTION ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────

def fetch_reviews_from_url(url: str, apify_token: str) -> Tuple[List[dict], str]:
    """
    Executes the full Apify extraction pipeline.
    
    Returns:
        (items, error_message): A tuple containing the list of normalised dictionaries 
                                and an error string (empty if successful).
    """
    # -- 1. Pre-flight Checks --
    if ApifyClient is None:
        return [], "System Error: 'apify-client' package is not installed."

    if not apify_token or not apify_token.strip():
        return [], "Authentication Error: Apify API token is missing."

    if not url or not url.strip().startswith("http"):
        return [], f"Validation Error: Invalid URL '{url}'. Must include http/https."

    # -- 2. Routing & Payload Prep --
    platform = _detect_platform(url)
    actor_id = _ACTOR_MAP[platform]
    run_input = _build_actor_payload(platform, url)

    print(f"[API_FETCH] Initiating extraction pipeline for: {platform.upper()}")
    print(f"[API_FETCH] Target URL: {url}")
    print(f"[API_FETCH] Assigned Actor: {actor_id}")

    # -- 3. Apify API Execution --
    try:
        client = ApifyClient(apify_token.strip())

        # Call the actor synchronously (blocks until the run is finished)
        run = client.actor(actor_id).call(run_input=run_input)

        # -- 4. Response Validation --
        if run is None or run.get("status") not in ("SUCCEEDED", "FINISHED"):
            status = run.get("status") if run else "NO_RESPONSE"
            return [], f"Actor Failure: Apify process terminated with status '{status}'."

        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            return [], "Extraction Error: Run succeeded but no dataset was generated."

        # -- 5. Data Retrieval & Normalisation --
        raw_items = list(
            client.dataset(dataset_id).iterate_items(limit=_TARGET_FETCH_LIMIT)
        )

        if not raw_items:
            return [], "Empty Result: The scraper completed but found 0 matching reviews. Try a different URL or relax filtering."

        normalised_items = [_normalise_item(item, platform) for item in raw_items]
        
        print(f"[API_FETCH] Success: Retrieved and normalised {len(normalised_items)} records.")
        return normalised_items, ""

    # -- 6. Exception Handling --
    except Exception as e:
        print("[API_FETCH] Critical Pipeline Failure:")
        traceback.print_exc()
        
        # Determine if it's an auth error vs a timeout/network error
        if "401" in str(e) or "unauthorized" in str(e).lower():
            return [], "Authentication Error: Your Apify API key was rejected."
        elif "400" in str(e):
            return [], "Payload Error: The specific Apify actor rejected the configuration parameters."
            
        return [], "Network Error: Apify fetch failed unexpectedly. Check the console logs."