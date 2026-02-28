"""Hybrid intent classifier: keyword pre-filter + DeBERTa zero-shot NLI.

Keyword patterns catch deterministic IOT commands and explicit search requests
instantly (<1ms). DeBERTa NLI handles the ambiguous search vs general distinction
by reasoning about whether the query entails "seeking factual information."
"""
import logging
import re
import threading
from typing import Optional, Tuple

from transformers import pipeline

logger = logging.getLogger('langchain_agent.nli_classifier')

_classifier = None
_init_lock = threading.Lock()

# --- Keyword patterns (fast, deterministic) ---

IOT_PATTERNS = [
    r'\b(turn|switch|toggle)\b.*(on|off)\b',
    r'\b(turn|switch)\b.*(back)\b',
    r'\b(on|off)\b.*(turn|switch)\b',
    r'\b(dim|brighten|brightness)\b',
    r'\b(set|change|adjust)\b.*(volume|brightness|temperature|color|colour)\b',
    r'\b(volume)\s*(up|down|\d)',
    r'\b(play|pause|stop|resume|skip|next|previous)\b.*(music|song|track|media|podcast|radio)',
    r'\b(play|pause|stop|resume)\b\s+(music|song|track|media|something|it)',
    r'\b(play)\b.*(in the|in my|on the)',
    r'\b(lock|unlock)\b',
    r'\b(add|put)\b.*(list|todo|to-do)',
    r'\b(remove|delete|cross off|check off|mark)\b.*(list|todo|to-do)',
    r"\b(what'?s on|show|check)\b.*(list|todo|to-do)",
    r'\b(shopping list|todo list|to-do list)\b',
    r'\b(how many|what).*(on my|on the).*(list)\b',
]

SEARCH_PATTERNS = [
    r'^(search|google|look up|find)\b',
    r'\b(search for|look up|google)\b',
    r'\b(weather|forecast|rain|temperature outside)\b',
]

# --- NLI hypotheses (for search vs general distinction) ---

NLI_HYPOTHESES = {
    "search": "This is a question seeking factual information, knowledge, or a web search.",
    "general": "This is casual conversation, a greeting, small talk, or a context-dependent follow-up.",
}

SEARCH_THRESHOLD = 0.70


def _init_nli():
    global _classifier

    if _classifier is not None:
        return _classifier

    with _init_lock:
        if _classifier is not None:
            return _classifier

        logger.info("[nli_classifier] Loading DeBERTa zero-shot NLI model...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        )
        logger.info("[nli_classifier] Model loaded successfully")

    return _classifier


def _keyword_match(query: str) -> Optional[str]:
    """Fast keyword matching. Returns intent or None."""
    q_lower = query.lower().strip()

    for pattern in IOT_PATTERNS:
        if re.search(pattern, q_lower):
            return "iot"

    for pattern in SEARCH_PATTERNS:
        if re.search(pattern, q_lower):
            return "search"

    return None


def classify_intent(query: str) -> str:
    """Classify query intent using keyword pre-filter + zero-shot NLI.

    Strategy:
        1. Keyword patterns catch IOT commands and explicit search requests (<1ms)
        2. DeBERTa NLI distinguishes search questions from general conversation (~100-300ms)
        3. Low-confidence NLI results default to "general" (safe fallback)

    Returns:
        Route name: "iot", "search", or "general"
    """
    # Step 1: Fast keyword match
    keyword_intent = _keyword_match(query)
    if keyword_intent:
        logger.info(f"[nli_classifier] '{query}' -> {keyword_intent} (keyword)")
        return keyword_intent

    # Step 2: NLI classification for search vs general
    clf = _init_nli()

    result = clf(
        query,
        candidate_labels=list(NLI_HYPOTHESES.values()),
        multi_label=True,
    )

    hypothesis_to_intent = {v: k for k, v in NLI_HYPOTHESES.items()}
    scores = {
        hypothesis_to_intent[h]: s
        for h, s in zip(result["labels"], result["scores"])
    }
    score_summary = ", ".join(f"{k}={v:.3f}" for k, v in scores.items())

    if scores["search"] >= SEARCH_THRESHOLD:
        logger.info(f"[nli_classifier] '{query}' -> search ({score_summary})")
        return "search"

    logger.info(f"[nli_classifier] '{query}' -> general ({score_summary})")
    return "general"
