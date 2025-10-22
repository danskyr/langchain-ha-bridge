# Logging Documentation

## Overview

The LangChain HA Bridge uses comprehensive file-based logging to track all conversations, errors, and system events. This makes debugging and analyzing conversations super easy.

## Log Files

All logs are stored in the `logs/` directory at the project root.

### 1. Main Application Log
**Location**: `logs/langchain_agent.log`

Contains all application logs including:
- HTTP requests and responses
- Graph node execution
- LLM interactions
- Tool calls
- System events

**Features**:
- Rotating log file (max 10MB per file)
- Keeps 5 backup files
- Format: `YYYY-MM-DD HH:MM:SS | module_name | LEVEL | message`

**Example**:
```
2025-10-22 14:30:15 | server | INFO | 🌐 Incoming request: 45 chars, has_tools=True, has_results=False
2025-10-22 14:30:15 | router_agent_v2 | INFO | [router] Analyzing query: turn on the living room light
2025-10-22 14:30:16 | router_agent_v2 | INFO | [agent] LLM requested 1 tool calls
```

### 2. Error-Only Log
**Location**: `logs/langchain_agent_errors.log`

Contains only ERROR and CRITICAL level logs for quick error diagnosis.

**Features**:
- Rotating log file (max 10MB per file)
- Keeps 5 backup files
- Same format as main log

### 3. Daily Conversation Logs
**Location**: `logs/conversations/YYYY-MM-DD.log`

Simplified, human-readable logs focused on conversation flow. Each day gets its own file.

**Features**:
- One file per day
- Clean format focused on user interactions
- Easy to read and share

**Example**:
```
14:30:15 | ================================================================================
14:30:15 | 📥 NEW REQUEST | Conversation: abc12345...
14:30:15 | Query: turn on the living room light
14:30:15 | Has tools: True (12 tools)
14:30:15 | Has tool results: False (0 results)
14:30:16 | 🔧 TOOL CALLS REQUESTED: 1
14:30:16 |   1. HassLightSet(name='living room', brightness=100)
14:30:16 | Conversation ID: abc12345-6789-...
14:30:16 | ================================================================================
14:30:17 | ================================================================================
14:30:17 | 📥 NEW REQUEST | Conversation: abc12345...
14:30:17 | Query: turn on the living room light
14:30:17 | Has tools: True (12 tools)
14:30:17 | Has tool results: True (1 results)
14:30:17 | Tool Results:
14:30:17 |   1. HassLightSet: {'success': True, 'message': 'Light turned on'}
14:30:18 | ✅ FINAL RESPONSE:
14:30:18 |    I've turned on the living room light
14:30:18 | ================================================================================
```

## Quick Reference for Claude Code

When debugging conversations, you can:

### Find Today's Conversations
```bash
cat logs/conversations/$(date +%Y-%m-%d).log
```

### Find Recent Errors
```bash
tail -n 50 logs/langchain_agent_errors.log
```

### Follow Live Logs
```bash
tail -f logs/langchain_agent.log
```

### Search for Specific Conversation
```bash
grep -A 20 "abc12345" logs/conversations/*.log
```

### Find All Tool Calls Today
```bash
grep "TOOL CALLS REQUESTED" logs/conversations/$(date +%Y-%m-%d).log
```

## Log Levels

The application uses standard Python logging levels:

- **DEBUG**: Detailed diagnostic information (not logged to file by default)
- **INFO**: General informational messages (default level)
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical issues that may cause system failure

## Configuration

Log level can be configured via environment variable:

```bash
# In .env file
LANGCHAIN_LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Conversation Flow Example

Here's what a typical conversation looks like in the logs:

### 1. Initial Request with Tools
```
📥 NEW REQUEST | Conversation: new
Query: what lights are on?
Has tools: True (12 tools)
🔧 TOOL CALLS REQUESTED: 1
  1. HassGet(domain='light', attributes='state')
Conversation ID: f7a9c2d1-...
```

### 2. Tool Execution (by Home Assistant)
Home Assistant executes the tool and sends results back.

### 3. Continuation with Results
```
📥 NEW REQUEST | Conversation: f7a9c2d1...
Query: what lights are on?
Has tool results: True (1 results)
Tool Results:
  1. HassGet: {'living_room': 'on', 'bedroom': 'off', 'office': 'on'}
✅ FINAL RESPONSE:
   The living room and office lights are on. The bedroom light is off.
```

## Troubleshooting

### Logs Not Appearing?
1. Check that the server has write permissions to the `logs/` directory
2. Verify `LANGCHAIN_LOG_LEVEL` is set to INFO or lower
3. Ensure the server is running (logs are initialized on startup)

### Logs Too Verbose?
Set `LANGCHAIN_LOG_LEVEL=WARNING` to reduce log volume.

### Need More Detail?
Set `LANGCHAIN_LOG_LEVEL=DEBUG` for maximum detail.

### Old Logs Taking Up Space?
The rotating file handler automatically manages log file sizes. Main logs rotate when they exceed 10MB, keeping only the 5 most recent files. Conversation logs are daily and don't rotate (you may want to manually archive old ones).

## Best Practices

1. **Always check conversation logs first** - They're human-readable and show the high-level flow
2. **Check error logs for failures** - Quick way to see what went wrong
3. **Use main logs for detailed debugging** - Full system state and execution trace
4. **Archive old conversation logs** - Keep a few months of daily logs for reference
5. **Monitor log file sizes** - Though rotation is automatic, keep an eye on disk usage

## Integration with Monitoring Tools

The log format is compatible with:
- **grep/awk**: Standard Unix text processing
- **Logstash**: Can parse the structured format
- **Grafana Loki**: Can ingest and visualize
- **CloudWatch/Datadog**: Standard log format for cloud monitoring
