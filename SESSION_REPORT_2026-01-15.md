# Session Report - January 15, 2026

## Summary

Continued work from previous session. Main focus was investigating and fixing intent routing issues where follow-up commands like "now back on" weren't being handled correctly.

## What Was Accomplished

### 1. Semantic Router Implementation (Complete)

Replaced keyword-based routing with embedding-based semantic routing using the `semantic-router` library.

**Files created/modified:**
- `langchain_agent/src/semantic_router.py` (NEW) - Semantic intent classification
- `langchain_agent/src/nodes/router.py` - Updated to use semantic router
- `pyproject.toml` - Added dependencies: `semantic-router[local]`, `transformers`, `torch`

**How it works:**
- Uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2` encoder
- Pre-defined routes with example utterances for: `iot`, `search`, `general`
- Routes by nearest-neighbor similarity (no LLM generation needed)
- Lazy initialization with thread-safe singleton pattern

**Key fix:** Added `auto_sync="local"` to fix "Index is not ready" error (GitHub Issue #519)

### 2. Duplicate Message History Bug (Fixed)

**Problem:** Conversation history was being duplicated - HA sends full history AND LangGraph checkpointer stores state.

**Fix in `router_agent_v2.py`:**
```python
existing_checkpoint = self.checkpointer.get(config)
has_existing_state = (
    existing_checkpoint is not None
    and isinstance(existing_checkpoint, dict)
    and existing_checkpoint.get("channel_values", {}).get("messages")
)
```

### 3. JSON Serialization Error (Fixed)

**Problem:** `TypeError: Object of type ExecutionTrace is not JSON serializable`

**Fix:** Used `asdict()` from dataclasses:
```python
from dataclasses import asdict
"execution_trace": asdict(execution_trace) if execution_trace else None
```

## Current State

### What's Working
- Semantic routing correctly classifies "now back on", "turn it off", etc. as IOT intents
- Execution traces are being logged to `logs/conversations/YYYY-MM-DD.log`
- Tool calls are being generated and returned to HA
- Server hot-reloads on code changes

### Test Results (End of Session)
```
"turn on the light in the office" → routed to 'iot' → HassTurnOn called ✓
"now turn it back off" → routed to 'iot' → HassTurnOff called ✓
```

## Known Issues / Not Yet Addressed

### 1. Area Parameter Missing on Follow-ups
When user says "now off" after "turn on office light", the LLM sometimes uses an empty `area` parameter instead of remembering "office". The system prompt was updated to encourage reusing previous tool arguments, but this may need further tuning.

### 2. Router is Not Context-Aware
The semantic router classifies based on the current query only - it doesn't look at conversation history. For most cases this works (semantic similarity catches "now back on"), but for truly ambiguous queries, checking if previous messages had IOT tool calls could improve accuracy.

**Potential enhancement:** In `router_node()`, check `state["messages"]` for recent tool calls before falling back to semantic classification.

## File Locations

| Purpose | Path |
|---------|------|
| Main agent | `langchain_agent/src/router_agent_v2.py` |
| Semantic router | `langchain_agent/src/semantic_router.py` |
| Router node | `langchain_agent/src/nodes/router.py` |
| Agent node | `langchain_agent/src/nodes/agent.py` |
| Execution tracer | `langchain_agent/src/execution_tracer.py` |
| WebSocket handler | `langchain_agent/src/websocket_handler.py` |
| Server | `langchain_agent/src/server.py` |
| Conversation logs | `logs/conversations/YYYY-MM-DD.log` |
| Main log | `logs/langchain_agent.log` |

## Commands

```bash
# Start server (hot-reloads automatically)
poetry run langchain-ha-bridge

# View conversation logs
tail -f logs/conversations/$(date +%Y-%m-%d).log

# View main logs
tail -f logs/langchain_agent.log
```

## Next Steps (Suggested)

1. **Test more edge cases** - Try various follow-up commands to ensure semantic routing handles them
2. **Context-aware routing** - Consider passing conversation history to router for truly ambiguous cases
3. **Area parameter persistence** - Investigate why LLM doesn't always reuse area from previous tool calls
4. **Add more utterances** - Expand semantic router training examples based on real usage patterns

## Dependencies Added This Session

```toml
semantic-router = {extras = ["local"], version = "^0.1.1"}
transformers = "^4.47.1"
torch = "^2.5.1"
```

---

## Session 2 - Later on January 15, 2026

### Summary

Added two debugging/testing features:
1. **Mock data capture** - Save raw JSON request/response pairs for building test fixtures
2. **Execution path visualization** - ASCII graph showing which nodes were visited during graph execution

### 4. Mock Data Capture (Complete)

Captures raw JSON requests and responses between the LangChain service and Home Assistant for building test fixtures.

**Files created/modified:**
- `langchain_agent/src/mock_data_capture.py` (NEW) - MockDataCapture class
- `langchain_agent/src/server.py` - Added capture calls in `/v1/completions`
- `.env.shape` - Added `CAPTURE_MOCK_DATA=false`

**How it works:**
- Enabled via `CAPTURE_MOCK_DATA=true` env var
- Saves to `logs/mock_data/YYYY-MM-DD/{conversation_id}/`
- Files numbered sequentially: `001_request.json`, `001_response.json`, etc.
- Multi-turn conversations (tool calls) get sequential numbers

**Directory structure:**
```
logs/mock_data/
└── 2026-01-15/
    └── test-light-1/
        ├── 001_request.json
        ├── 001_response.json
        ├── 002_request.json  # Tool result continuation
        └── 002_response.json
```

### 5. Execution Path Visualization (Complete)

Visual ASCII representation of the LangGraph execution path logged after each response.

**Files created/modified:**
- `langchain_agent/src/execution_tracer.py` (NEW) - ExecutionTracer class with ASCII renderer
- `langchain_agent/src/router_agent_v2.py` - Added `_invoke_with_tracing()` using `astream_events()`
- `langchain_agent/src/server.py` - Logs trace after HTTP responses
- `langchain_agent/src/websocket_handler.py` - Logs trace after WebSocket responses
- `.env.shape` - Added `LOG_EXECUTION_PATH=true`

**How it works:**
- Uses LangGraph's `astream_events()` instead of `ainvoke()` to capture node visits
- Records node start/end times and metadata (route_types, tool_calls, decisions)
- Renders ASCII box diagram showing execution flow

**Example output:**
```
================================================================================
EXECUTION PATH | Conversation: test-final...
================================================================================

  START
    │
    ▼
┌───────────┐
│   router  │  route_types: ['iot'] (2295ms)
└─────┬─────┘
     │
     ├────────┬───────────────┐
     ▼        ▼               ▼
┌─────────────┐  ┌──────────────┐
│ iot_handler │  │ announcement │
└──────┬──────┘  └───────┬──────┘
       └─────────────────┘
                │
                ▼
          ┌──────────────┐
          │  aggregator  │
          └──────────────┘
                │
                ▼
          ┌──────────────┐
          │    agent     │ (4186ms)
          └──────────────┘
                │
                ▼
    ┌────────────────────────┐
    │  tool_call_validation  │
    └────────────────────────┘
                │
                ▼
          ┌──────────────┐
          │  formatter   │
          └──────────────┘
                │
                ▼
              [END]

Total: 6.49s | Nodes: 7 | Tool calls: 0
================================================================================
```

### Bug Fix: WebSocket Handler Missing Trace Logging

The execution trace was only being logged for HTTP requests, not WebSocket. Fixed by adding trace logging to `websocket_handler.py`.

### Updated File Locations

| Purpose | Path |
|---------|------|
| Mock data capture | `langchain_agent/src/mock_data_capture.py` |
| Execution tracer | `langchain_agent/src/execution_tracer.py` |
| Mock data output | `logs/mock_data/YYYY-MM-DD/{conversation_id}/` |

### Environment Variables Added

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAPTURE_MOCK_DATA` | `false` | Enable JSON request/response capture |
| `LOG_EXECUTION_PATH` | `true` | Enable ASCII execution path logging |

### Next Steps (Updated)

1. Use captured mock data to write unit tests
2. Consider adding timing breakdown per node in execution trace
3. Add trace visualization to streaming endpoint if needed
