# Session Report - 2025-12-03

## Summary
Debugging and fixing issues with the LangChain Home Assistant voice assistant, focusing on tool call execution, conversation context, and LLM behavior.

## Bugs Fixed

### 1. Tool Calls Not Triggering on Subsequent Requests
**File:** `langchain_agent/src/nodes/agent.py`

**Problem:** After a tool was called once in a conversation thread, all subsequent requests skipped tool invocation and falsely reported success.

**Root Cause:** Line 23 checked `any(isinstance(msg, ToolMessage) for msg in messages)` which returned True for the entire conversation history, not just pending results.

**Fix:** Added `has_pending_tool_results()` function that correctly detects if we're resuming after tool execution by checking if:
- The most recent messages are `ToolMessage`s
- They follow an `AIMessage` with `tool_calls`
- A new `HumanMessage` hasn't been added (indicating a new query)

### 2. Tool Results Incorrectly Marked as Failed
**File:** `langchain_agent/src/nodes/agent.py`

**Problem:** Successful Home Assistant responses like `'response_type': 'action_done'` were being marked as failed.

**Fix:** Updated `analyze_tool_results()` to recognize HA success patterns:
- `'response_type': 'action_done'`
- `'success': [`
- Also detects HA error patterns: `'response_type': 'error'`

### 3. Collecting ALL Tool Results Instead of Latest Batch
**File:** `langchain_agent/src/router_agent_v2.py`

**Problem:** When resuming with tool results, the code collected ALL `tool_result` messages from the entire conversation, not just the latest batch.

**Fix:** Changed to iterate backwards from the end and only collect consecutive `tool_result` messages:
```python
for m in reversed(messages):
    if m.get("role") == "tool_result":
        tool_results.insert(0, {...})
    else:
        break
```

Also added `name` parameter to `ToolMessage` creation which was missing.

### 4. Empty Strings in Tool Arguments
**File:** `langchain_agent/src/router_agent_v2.py`

**Problem:** LLM sent empty strings `""` for optional parameters (e.g., `area: ""`), which Home Assistant rejected with `InvalidSlotInfo` error.

**Fix:** Filter out empty strings along with None values:
```python
"args": {k: v for k, v in tc["args"].items() if v is not None and v != ""}
```

### 5. Missing Conversation Context
**File:** `langchain_agent/src/router_agent_v2.py`

**Problem:** When user said "dim them to 50%", the LLM didn't know "them" referred to "office lights" from previous messages. The full conversation history was received but discarded.

**Fix:**
- Added `convert_ha_messages_to_langchain()` helper function
- Added `MAX_HISTORY_MESSAGES = 10` constant
- Include conversation history in initial state:
```python
"messages": conversation_history + [HumanMessage(content=query)]
```

### 6. System Prompt Updates
**File:** `langchain_agent/src/nodes/agent.py`

Added instructions for:
- Using conversation history for pronouns ("them", "it")
- Not calling tools for conversation history questions
- Using `HassLightSet` for brightness/color changes
- Always trying color changes instead of assuming device can't do it

## Outstanding Issue: LLM Not Calling HassLightSet for Color

**Status:** NOT FIXED - needs further investigation

**Problem:** When user says "turn it amber", the Ollama LLM decides NOT to call any tool and instead responds "This device is not able to change its color".

**Logs show:**
```
[agent] LLM did not call tools despite having 20 available
[agent] Generating final response without tools
```

**What we tried:**
- Updated system prompt with explicit `HassLightSet` examples for color
- Added instruction: "Always TRY to set colors/brightness with HassLightSet. Don't assume a light can't change color"

**Possible next steps:**
1. Try a different/better Ollama model
2. Further prompt engineering
3. Add few-shot examples in the system prompt
4. Check if the model is seeing the full tool schema for HassLightSet
5. Consider adding a "color" keyword detection in the router to force IOT routing
6. Debug what the LLM is actually receiving vs responding with

## Files Modified This Session

1. `langchain_agent/src/nodes/agent.py`
   - `has_pending_tool_results()` - new function
   - `analyze_tool_results()` - fixed HA success detection
   - System prompt updates for context and color handling
   - Added debug logging for messages in state

2. `langchain_agent/src/router_agent_v2.py`
   - `convert_ha_messages_to_langchain()` - new function
   - Fixed tool result collection (only latest batch)
   - Added `name` parameter to ToolMessage
   - Added conversation history to initial state
   - Filter empty strings from tool args

## Debug Logging Added

In `agent.py`, added logging to show messages the LLM receives:
```
[agent] Messages in state: 18
[agent]   1. SystemMessage: You are a voice assistant...
[agent]   2. HumanMessage: turn on the light in the office
...
```

This is useful for debugging context issues.

## Test Commands

After making changes, test with:
1. "turn on the office lights"
2. "dim them to 50%" (tests pronoun resolution)
3. "turn it amber" (tests color - currently failing)
4. "what was my last request?" (tests conversation history)

---

## Part 2: WebSocket Concurrency & Log Forwarding (Evening Session)

### Background (From Nov 26th)
The client uses a single WebSocket connection where multiple tasks compete for access:
- `send_conversation()` - handles LLM requests
- `ping()` - heartbeat checks
- `WebSocketLogHandler` - forwards HA logs to server

The original lock-based approach caused blocking during long conversations.

### Solution Implemented: Dedicated Reader Task Pattern

**Architectural Change:**
```
WebSocket Connection
       |
  +----+----+
  |         |
send()   _reader_loop() (single background task)
  |         |
  |    +----+----+----+
  |    |         |    |
  |  pong→    response→  log→
  |  Future   Queue    (ignore)
```

**Files Modified:**

1. `custom_components/langchain_conversation/client.py`
   - Removed `_receive_lock` (old locking approach)
   - Added `_reader_task`, `_response_queues`, `_queues_lock`, `_pending_ping`
   - New methods: `_start_reader_task()`, `_stop_reader_task()`, `_reader_loop()`, `_route_message()`, `_notify_all_waiters_of_disconnect()`
   - `send_conversation()` now creates a queue per conversation_id, sends request, waits on queue
   - `ping()` now uses a Future instead of blocking receive
   - `WebSocketLogHandler` changed to use thread-safe `queue.Queue` instead of `asyncio.Queue` (emit() is called synchronously from logging threads)

2. `langchain_agent/src/websocket_handler.py`
   - Added debug logging in `_handle_log()`: `logger.info(f"Received HA log: {level} - {message[:50]}...")`

3. `langchain_agent/src/server.py`
   - Configured `home_assistant` logger for forwarded HA logs

### Outstanding Issue: Log Forwarding Not Working

**Status:** NOT FIXED - needs investigation

**Symptom:** Logs from `custom_components.langchain_conversation` are not appearing on the LangChain server. Conversations work fine, but the "Received HA log" message never appears in server logs.

**What we verified:**
- WebSocket connects successfully (server shows "Client connected")
- Conversations are processed correctly
- Server-side `_handle_log()` has debug logging
- Thread-safe queue is used in client

**Debugging code added (uncommitted):**

In `client.py` `emit()`:
```python
print(f"[LOG_HANDLER] emit called: connected={self.client.is_connected}, logger={record.name}")
```

In `client.py` `_send_logs()`:
```python
_LOGGER.info("Log forwarding background task started, queue size: %d", self._queue.qsize())
_LOGGER.info("Sending log to server: %s (connected: %s)", ...)
```

In `__init__.py`:
```python
_LOGGER.info("Added log handler to logger: %s (effective level: %s)", ...)
```

**These changes are NOT committed** - they were rejected to avoid noisy debug logs in production.

### Theories for Log Forwarding Failure

1. **emit() not being called** - Handler might not be receiving logs (logger hierarchy issue?)
2. **Background task not running** - `asyncio.create_task()` might not work as expected in HA's event loop
3. **is_connected returning False** - Connection might be lost between emit() and send()
4. **Silent exceptions** - All errors are swallowed for "best effort" logging

### How to Continue This Work

1. **Enable debug output temporarily:**
   - Uncomment/add the print() statements in `emit()` method
   - These print to stdout which HA captures
   - Look for `[LOG_HANDLER]` lines in HA logs

2. **Check if handler is registered:**
   ```python
   # In __init__.py after adding handler:
   _LOGGER.info("Handlers on logger: %s", component_logger.handlers)
   ```

3. **Test background task:**
   - Add a simple periodic log in `_send_logs()` that doesn't depend on the queue
   - This verifies the task is actually running

4. **Check if logs reach emit():**
   - The filter skips `client` module logs to prevent loops
   - Logs from `conversation.py` and `__init__.py` should pass through

5. **Files to examine:**
   - `custom_components/langchain_conversation/client.py` - WebSocketLogHandler class
   - `custom_components/langchain_conversation/__init__.py` - Handler setup at lines 47-61
   - `langchain_agent/src/websocket_handler.py` - Server-side `_handle_log()` at line 97

6. **Deployment note:**
   - Server uses hot reload (`uvicorn --reload`)
   - Custom components require HACS re-download + HA integration reload

### Commits Made Today

- `d02953d` - log streaming & websocket concurrency (main implementation)
- Previous commits from Nov 26th established the WebSocket infrastructure
