# Session Report - January 28, 2026

## Summary

Research and planning session focused on making the semantic router context-aware. No code changes were made — this was design exploration only.

## Problem Statement

The semantic router (`langchain_agent/src/semantic_router.py`) classifies queries in isolation. Follow-up messages that depend on conversation context get misrouted.

**Example:**
```
User: Turn off lights in office
Agent: Lights are off
User: back on again        ← router sees this in isolation, can't reliably route it
```

The previous session (Jan 15) tried to work around this by adding ambiguous utterances ("back on", "again", "do it again", "same thing") directly to the IoT route's utterance list (lines 45-52 of `semantic_router.py`). This only works when those phrases happen after IoT commands — "back on again" after a search conversation would still misroute to IoT.

## Key Findings

### 1. Conversation history IS available at routing time
`RouterState` contains `messages` (full conversation history via LangGraph checkpointing), but `router_node` only extracts `state["query"]` and passes it to `classify_intent(query)`. The context is there — it's just not used.

### 2. The semantic_router library exposes confidence scores
`SemanticRouter.__call__()` returns a `RouteChoice` object with:
- `name`: matched route (or `None` if nothing passes threshold)
- `similarity_score`: float confidence value
- Per-route `score_threshold` configuration is supported
- `limit=None` parameter returns ALL routes that pass threshold with scores

This means the semantic router can signal "I'm not sure" — which is the key enabler for the proposed design.

### 3. Context injection into the semantic router is unreliable
Prepending conversation context to the query string before embedding is a hack. The encoder (`all-MiniLM-L6-v2`) was trained on sentence-level semantics, not concatenated conversation fragments. Embedding quality is unpredictable.

## Proposed Design: Two-Tier Routing

```
query → semantic_router(query)
         ├─ confident (score >= threshold) → use route       [fast path, no LLM]
         └─ uncertain (score < threshold)  → llm_classify()  [slow path, uses history]
```

The semantic router stays doing what it's good at (fast matching of clear queries). When it's unsure, an LLM classifier takes over with full conversation history.

### Draft plan file
Full implementation plan is at: `.claude/plans/mellow-squishing-sun.md`

### Files to modify
- `langchain_agent/src/semantic_router.py` — expose confidence scores, remove hacky ambiguous utterances
- `langchain_agent/src/nodes/router.py` — two-tier routing logic
- `langchain_agent/src/router_agent_v2.py` — pass LLM to router node (factory pattern)
- `langchain_agent/src/llm_router.py` — new LLM fallback classifier
- `tests/test_router_context.py` — new tests

## Open Questions (to decide next session)

1. **Which Ollama model for the LLM fallback?** Reuse existing `ROUTER_MODEL` config, or add a separate `LLM_ROUTER_MODEL` env var for independent tuning?
2. **Should the LLM fallback also resolve the query?** Option A: just classify ("iot"/"search"/"general") and let the downstream handler resolve context from history. Option B: also rewrite the query (e.g., "back on again" → "turn on office lights") and store it in state.
3. **Confidence threshold value**: Needs experimentation. Start around 0.3-0.5 and log all scores to calibrate.

## No Code Changes

This was a research/planning session. No files were modified.
