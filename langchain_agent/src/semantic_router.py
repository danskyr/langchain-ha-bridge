import logging
import threading
from typing import Optional

from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

logger = logging.getLogger('langchain_agent.semantic_router')

_semantic_router: Optional[SemanticRouter] = None
_init_lock = threading.Lock()


def _init_router() -> SemanticRouter:
    """Initialize the semantic router with routes and encoder."""
    global _semantic_router

    if _semantic_router is not None:
        return _semantic_router

    with _init_lock:
        if _semantic_router is not None:
            return _semantic_router

        logger.info("[semantic_router] Initializing with HuggingFace encoder...")

        encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

        iot_route = Route(
            name="iot",
            utterances=[
                "turn on the light",
                "turn off the light",
                "turn on the lights",
                "turn off the lights",
                "switch on the light",
                "switch off the light",
                "turn it on",
                "turn it off",
                "turn them on",
                "turn them off",
                "switch it on",
                "switch it off",
                "now back on",
                "back on",
                "back off",
                "turn it back on",
                "turn it back off",
                "do it again",
                "same thing",
                "undo that",
                "again",
                "dim the lights",
                "brighten the lights",
                "set brightness to 50",
                "make it brighter",
                "make it dimmer",
                "dim them",
                "make it red",
                "make it blue",
                "change color to blue",
                "set it to amber",
                "turn it amber",
                "set the color",
                "turn on bedroom lights",
                "turn off bedroom lights",
                "office lights on",
                "office lights off",
                "living room lights",
                "turn on the office",
                "turn off the office",
                "add milk to shopping list",
                "add to my list",
                "what's on my list",
                "what's on the shopping list",
                "remove eggs from list",
                "complete the task",
                "mark it done",
                "play music",
                "pause the music",
                "stop playing",
                "next song",
                "previous track",
                "volume up",
                "volume down",
                "set volume to 50",
            ]
        )

        search_route = Route(
            name="search",
            utterances=[
                "what's the weather",
                "what is the weather",
                "weather forecast",
                "is it going to rain",
                "how's the weather outside",
                "search for",
                "look up",
                "find information about",
                "who is",
                "what is",
                "when did",
                "where is",
                "how do I",
                "latest news about",
                "tell me about",
                "google",
                "search the web",
            ]
        )

        general_route = Route(
            name="general",
            utterances=[
                "hello",
                "hi there",
                "hey",
                "good morning",
                "good evening",
                "how are you",
                "what can you do",
                "help",
                "help me",
                "thank you",
                "thanks",
                "goodbye",
                "bye",
                "what are you",
                "who are you",
            ]
        )

        _semantic_router = SemanticRouter(
            encoder=encoder,
            routes=[iot_route, search_route, general_route],
            auto_sync="local"
        )

        logger.info("[semantic_router] Initialized successfully")

    return _semantic_router


def classify_intent(query: str) -> str:
    """Classify query intent using semantic similarity.

    Args:
        query: The user's query string

    Returns:
        Route name: "iot", "search", or "general"
    """
    router = _init_router()
    result = router(query)
    route_name = result.name if result else "general"
    logger.info(f"[semantic_router] '{query}' -> {route_name}")
    return route_name
