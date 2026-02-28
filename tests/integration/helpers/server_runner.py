"""Run the langchain FastAPI server in a background thread for tests."""
import logging
import threading
import time

import httpx
import uvicorn

logger = logging.getLogger(__name__)

TEST_SERVER_PORT = 8002


class ServerRunner:
    """Manages the langchain server lifecycle in a background daemon thread."""

    def __init__(self, port: int = TEST_SERVER_PORT):
        self.port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self, startup_timeout: float = 120):
        """Start the server and block until it's healthy."""
        config = uvicorn.Config(
            "langchain_agent.src.server:app",
            host="0.0.0.0",
            port=self.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        self._wait_healthy(startup_timeout)
        logger.info("Langchain server is healthy on port %d", self.port)

    def _wait_healthy(self, timeout: float):
        """Poll /health until the server responds 200."""
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + timeout
        last_err = None
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout=5)
                if resp.status_code == 200:
                    return
            except Exception as exc:
                last_err = exc
            time.sleep(2)
        raise RuntimeError(
            f"Langchain server did not become healthy within {timeout}s. Last error: {last_err}"
        )

    def stop(self):
        """Signal the server to shut down."""
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=10)
            logger.info("Langchain server stopped")
