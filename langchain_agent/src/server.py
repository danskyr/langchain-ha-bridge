import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, WebSocket

from langchain_agent.src.websocket_handler import WebSocketHandler
from langchain_agent.src.router_agent_v2 import LangChainRouterAgentV2


def setup_file_logging():
    """
    Set up comprehensive file logging for easy debugging.

    Creates:
    - logs/langchain_agent.log - All logs (rotating, max 10MB, keep 5 backups)
    - logs/langchain_agent_errors.log - Errors only
    - logs/conversations/YYYY-MM-DD.log - Daily conversation logs
    """
    # Create logs directory
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create conversations subdirectory
    conversations_dir = log_dir / "conversations"
    conversations_dir.mkdir(exist_ok=True)

    # Detailed formatter with timestamps
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Simple formatter for conversation logs
    conversation_formatter = logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # 1. Main rotating log file - All logs
    main_file_handler = RotatingFileHandler(
        log_dir / "langchain_agent.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    main_file_handler.setLevel(logging.INFO)
    main_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(main_file_handler)

    # 2. Error-only log file
    error_file_handler = RotatingFileHandler(
        log_dir / "langchain_agent_errors.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_file_handler)

    # 3. Daily conversation log file
    today = datetime.now().strftime("%Y-%m-%d")
    conversation_file_handler = logging.FileHandler(
        conversations_dir / f"{today}.log",
        encoding='utf-8'
    )
    conversation_file_handler.setLevel(logging.INFO)
    conversation_file_handler.setFormatter(conversation_formatter)

    # Create a separate logger for conversations
    conversation_logger = logging.getLogger('conversations')
    conversation_logger.setLevel(logging.INFO)
    conversation_logger.addHandler(conversation_file_handler)
    conversation_logger.propagate = False  # Don't propagate to root

    # Configure home_assistant logger for forwarded HA logs
    ha_logger = logging.getLogger('home_assistant')
    ha_logger.setLevel(logging.INFO)

    # 4. Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(console_handler)

    # Log startup
    root_logger.info("=" * 80)
    root_logger.info("File logging initialized")
    root_logger.info(f"Main log: {log_dir / 'langchain_agent.log'}")
    root_logger.info(f"Error log: {log_dir / 'langchain_agent_errors.log'}")
    root_logger.info(f"Conversation log: {conversations_dir / f'{today}.log'}")
    root_logger.info("=" * 80)

    return logging.getLogger('conversations')


# Initialize file logging
conversation_logger = setup_file_logging()

# Set up module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

router_agent = LangChainRouterAgentV2()
ws_handler = WebSocketHandler(router_agent)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time bidirectional communication with HA."""
    await ws_handler.handle_connection(websocket)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "langchain-conversation-agent"}

@app.post("/test")
async def test_connection():
    """Test endpoint for Home Assistant integration."""
    return {"status": "ok", "message": "Connection successful"}
