# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the LangChain Server
```bash
# Start the FastAPI server
poetry run langchain-ha-bridge
# or
./scripts/run_server.sh
```

**Note:** The server uses hot reload (uvicorn with `--reload`). When modifying server-side code, do NOT restart the server - it will automatically detect changes and reload. Check `logs/langchain_agent.log` to verify reload happened. The HA custom component requires reloading the integration in HA to pick up changes.

### Managing Dependencies
```bash
# Install dependencies
poetry install

# Add new dependency
poetry add <package>

# Development dependencies
poetry add --group dev <package>
```

### Testing
```bash
# Run tests
poetry run pytest

# Run async tests
poetry run pytest -v
```

### Running Supporting Services
```bash
# Start all Docker services (Home Assistant, Whisper, Piper)
./scripts/run_containers_dan.sh

# Individual services
cd services/home-assistant && docker-compose up -d
cd services/whisper && docker-compose up -d
cd services/piper && docker-compose up -d
cd services/ollama && docker-compose up -d
```

### Debugging and Logs

The application uses comprehensive file-based logging for easy debugging:

```bash
# View today's conversations (most useful for debugging)
cat logs/conversations/$(date +%Y-%m-%d).log

# Follow live logs
tail -f logs/langchain_agent.log

# View recent errors
tail -n 50 logs/langchain_agent_errors.log

# Search for specific conversation
grep -A 20 "conversation_id" logs/conversations/*.log
```

**Log Files:**
- `logs/langchain_agent.log` - Main application log (all events)
- `logs/langchain_agent_errors.log` - Errors only
- `logs/conversations/YYYY-MM-DD.log` - Daily human-readable conversation logs

**Configuration:**
```bash
# Set log level in .env
LANGCHAIN_LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
```

See `docs/Logging.md` for detailed documentation.

## Architecture Overview

### Core Components

**LangChain Router Agent V2** (`langchain_agent/src/router_agent_v2.py`) - **Current Implementation**:
- LangGraph StateGraph with checkpointing for distributed tool execution
- Parallel handler execution (IOT, Search, General can run concurrently)
- State management with automatic message merging using reducers
- Proper tool integration with bind_tools() and interrupt support
- Maintains conversation state across HTTP calls using thread-based checkpointing
- Graph structure: `router` → `[parallel handlers]` → `aggregator` → `agent` → `{tools?}` → `formatter`
- Supports multiple rounds of tool calls with Home Assistant

**LangChain Router Agent V1** (`langchain_agent/src/router_agent.py`) - **Legacy**:
- Original implementation with linear graph flow
- Handler-based system with `BaseHandler` abstract class
- Three handlers: `IOTHandler`, `SearchHandler`, `GeneralHandler`
- Note: Does not maintain state between tool execution rounds

**FastAPI Server** (`langchain_agent/src/server.py`):
- Exposes `/v1/completions` endpoint compatible with OpenAI API format
- Accepts `OpenAITextCompletionRequest` with prompt field
- Returns `OurResponse` with response field
- Health check at `/health` and test endpoint at `/test`

**Home Assistant Integration** (`custom_components/langchain_conversation/`):
- `RemoteConversationAgent` extends `AbstractConversationAgent` and `ConversationEntity`
- Forwards requests to FastAPI server via HTTP POST to `/v1/completions`
- Handles Home Assistant conversation pipeline integration
- Error handling for connection, SSL, timeout, and parsing issues

### Key Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI integration
- `ROUTER_MODEL`: Model for routing (default: gpt-3.5-turbo)
- `DEVICE_MODEL`: Model for device control (default: gpt-3.5-turbo) 
- `QUERY_MODEL`: Model for general queries (default: gpt-4)
- `TAVILY_API_KEY`: Required for web search functionality

### Service Architecture
The system requires multiple services:
1. **LangChain Server**: FastAPI service (port 8000)
2. **Ollama**: Local LLM server (port 11434)
3. **Whisper**: Speech-to-text service
4. **Piper**: Text-to-speech service
5. **Home Assistant**: Smart home platform with custom component

### Data Flow
1. User input → Home Assistant conversation interface
2. Home Assistant → FastAPI server `/v1/completions`
3. FastAPI server → LangChain Router Agent
4. Router Agent → Handler Selection (IOT, Search, or General)
5. Selected Handler → Process Query
6. Response Formatter → Unified formatting with consistent tone
7. Formatted Response → FastAPI → Home Assistant → User

### Important File Locations
- **Current router agent**: `langchain_agent/src/router_agent_v2.py` (with checkpointing)
- Legacy router agent: `langchain_agent/src/router_agent.py`
- FastAPI server with logging: `langchain_agent/src/server.py`
- HA conversation handler: `custom_components/langchain_conversation/conversation.py:74-137`
- Logging documentation: `docs/Logging.md`
- Conversation logs: `logs/conversations/YYYY-MM-DD.log`
- Main application log: `logs/langchain_agent.log`
- Error log: `logs/langchain_agent_errors.log`
- Poetry configuration: `pyproject.toml`
- Docker services: `services/*/docker-compose.yml`

### Development Notes
- Python 3.13.2+ required
- Uses Poetry for dependency management
- LangGraph for stateful agent workflows
- Async/await patterns throughout Home Assistant integration
- Error handling includes specific Home Assistant IntentResponseErrorCode types

### Adding New Handlers
To extend the system with new routing capabilities:

1. **Create Handler Class**: Inherit from `BaseHandler`
2. **Implement Methods**:
   - `can_handle(query: str) -> bool`: Classification logic
   - `get_route_type() -> RouteType`: Return appropriate enum value
   - `process(state: AgentState) -> AgentState`: Process the query
3. **Add Route Type**: Extend `RouteType` enum if needed
4. **Register Handler**: Add to `self.handlers` list in `LangChainRouterAgent.__init__()`

The system automatically routes to the first handler that returns `True` from `can_handle()`.