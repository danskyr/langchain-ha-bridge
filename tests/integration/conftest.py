"""Integration test fixtures: server, HA container, API client, state reset."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import aiohttp
import docker
import pytest
import pytest_asyncio

from .helpers.ha_client import HAClient
from .helpers.llm_judge import LLMJudge
from .helpers.server_runner import ServerRunner, TEST_SERVER_PORT
from .helpers.state_manager import StateManager

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCKER_DIR = Path(__file__).resolve().parent / "docker"
CUSTOM_COMPONENTS_DIR = PROJECT_ROOT / "custom_components" / "langchain_conversation"

HA_PORT = 8123
HA_BASE_URL = f"http://localhost:{HA_PORT}"
HA_CONTAINER_NAME = "ha_integration_test"

# Onboarding credentials
ONBOARD_NAME = "Test User"
ONBOARD_USERNAME = "test"
ONBOARD_PASSWORD = "testpassword123"
ONBOARD_CLIENT_ID = "http://localhost/"


# ---------------------------------------------------------------------------
# Session-scoped event loop (required for session-scoped async fixtures)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for all async fixtures and tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Langchain server (session-scoped)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def langchain_server():
    """Start the langchain FastAPI server in a background thread on port 8002."""
    runner = ServerRunner(port=TEST_SERVER_PORT)
    runner.start(startup_timeout=120)
    yield runner
    runner.stop()


# ---------------------------------------------------------------------------
# HA Docker container (session-scoped)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ha_container(langchain_server):
    """Run a Home Assistant container with demo integration and our custom component."""
    client = docker.from_env()

    # Remove any stale container from a previous run
    try:
        old = client.containers.get(HA_CONTAINER_NAME)
        old.remove(force=True)
        logger.info("Removed stale HA container")
    except docker.errors.NotFound:
        pass

    container = client.containers.run(
        image="ghcr.io/home-assistant/home-assistant:stable",
        name=HA_CONTAINER_NAME,
        detach=True,
        ports={f"{HA_PORT}/tcp": HA_PORT},
        volumes={
            str(DOCKER_DIR / "configuration.yaml"): {
                "bind": "/config/configuration.yaml",
                "mode": "ro",
            },
            str(DOCKER_DIR / "automations.yaml"): {
                "bind": "/config/automations.yaml",
                "mode": "ro",
            },
            str(CUSTOM_COMPONENTS_DIR): {
                "bind": "/config/custom_components/langchain_conversation",
                "mode": "ro",
            },
        },
        extra_hosts={"host.docker.internal": "host-gateway"},
        environment={"TZ": "Australia/Sydney"},
    )
    logger.info("Started HA container %s", container.short_id)

    # Wait for HA to become ready (onboarding endpoint returns 200)
    _wait_for_ha(timeout=120)

    yield container

    container.remove(force=True)
    logger.info("Removed HA container")


def _wait_for_ha(timeout: float = 120):
    """Block until HA's onboarding API responds."""
    import httpx

    deadline = time.monotonic() + timeout
    url = f"{HA_BASE_URL}/api/onboarding"
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("HA is ready (onboarding available)")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"HA did not become ready within {timeout}s")


# ---------------------------------------------------------------------------
# HA onboarding helpers
# ---------------------------------------------------------------------------
async def _get_onboarding_status(session: aiohttp.ClientSession) -> dict[str, bool]:
    """Return a dict of {step_name: done} from the onboarding API."""
    async with session.get(f"{HA_BASE_URL}/api/onboarding") as resp:
        steps = await resp.json()
        logger.info("Onboarding steps: %s", steps)
        return {s["step"]: s["done"] for s in steps}


async def _create_owner_and_get_token(session: aiohttp.ClientSession) -> str:
    """Run the onboarding user-creation step and return an access token."""
    async with session.post(
        f"{HA_BASE_URL}/api/onboarding/users",
        json={
            "name": ONBOARD_NAME,
            "username": ONBOARD_USERNAME,
            "password": ONBOARD_PASSWORD,
            "client_id": ONBOARD_CLIENT_ID,
            "language": "en",
        },
    ) as resp:
        resp.raise_for_status()
        auth_code = (await resp.json())["auth_code"]
        logger.info("Created owner user, got auth_code")

    return await _exchange_auth_code(session, auth_code)


async def _login_existing_user(session: aiohttp.ClientSession) -> str:
    """Authenticate an already-created user via the HA login flow."""
    # 1. Start a login flow
    async with session.post(
        f"{HA_BASE_URL}/auth/login_flow",
        json={
            "client_id": ONBOARD_CLIENT_ID,
            "handler": ["homeassistant", None],
            "redirect_uri": f"{ONBOARD_CLIENT_ID}?auth_callback=1",
        },
    ) as resp:
        resp.raise_for_status()
        flow = await resp.json()
        flow_id = flow["flow_id"]
        logger.info("Started login flow %s", flow_id)

    # 2. Submit credentials
    async with session.post(
        f"{HA_BASE_URL}/auth/login_flow/{flow_id}",
        json={
            "username": ONBOARD_USERNAME,
            "password": ONBOARD_PASSWORD,
            "client_id": ONBOARD_CLIENT_ID,
        },
    ) as resp:
        resp.raise_for_status()
        result = await resp.json()
        auth_code = result["result"]
        logger.info("Login flow complete, got auth_code")

    return await _exchange_auth_code(session, auth_code)


async def _exchange_auth_code(session: aiohttp.ClientSession, auth_code: str) -> str:
    """Exchange an auth code for a long-lived access token."""
    async with session.post(
        f"{HA_BASE_URL}/auth/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": ONBOARD_CLIENT_ID,
        },
    ) as resp:
        resp.raise_for_status()
        token_data = await resp.json()
        logger.info("Got access token")
        return token_data["access_token"]


async def _complete_onboarding_step(
    session: aiohttp.ClientSession,
    step: str,
    auth_headers: dict,
    payload: dict | None = None,
):
    """POST to an onboarding step endpoint, tolerating already-done or not-applicable."""
    url = f"{HA_BASE_URL}/api/onboarding/{step}"
    async with session.post(url, headers=auth_headers, json=payload or {}) as resp:
        if resp.status in (403, 400):
            body = await resp.text()
            logger.info(
                "Onboarding step '%s' returned %d (skipping): %s",
                step, resp.status, body[:200],
            )
            return
        resp.raise_for_status()
        logger.info("Onboarding step '%s' done", step)


# ---------------------------------------------------------------------------
# HA onboarding + config entry (session-scoped)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session")
async def ha_setup(ha_container) -> dict:
    """Complete HA onboarding and register the langchain_conversation integration.

    Idempotent: handles partial onboarding from a previous failed run by
    skipping already-completed steps and re-authenticating via login flow.

    Returns a dict with ``token`` (Bearer access token).
    """
    async with aiohttp.ClientSession() as session:
        # Check which onboarding steps are already done
        steps_done = await _get_onboarding_status(session)

        # Get an access token — either by creating the owner or logging in
        if not steps_done.get("user"):
            access_token = await _create_owner_and_get_token(session)
        else:
            access_token = await _login_existing_user(session)

        auth_headers = {"Authorization": f"Bearer {access_token}"}

        # Complete remaining onboarding steps (each tolerates already-done).
        # Order matters: HA enforces core_config → analytics → integration.
        await _complete_onboarding_step(session, "core_config", auth_headers)
        await _complete_onboarding_step(
            session, "analytics", auth_headers, payload={"preferences": {}}
        )
        await _complete_onboarding_step(session, "integration", auth_headers)

        # Set up langchain_conversation config entry (if not already present)
        await _ensure_langchain_config_entry(session, auth_headers)

    return {"token": access_token}


async def _ensure_langchain_config_entry(
    session: aiohttp.ClientSession, auth_headers: dict
):
    """Create the langchain_conversation config entry if it doesn't already exist."""
    # Check for existing config entries
    async with session.get(
        f"{HA_BASE_URL}/api/config/config_entries/entry",
        headers=auth_headers,
    ) as resp:
        resp.raise_for_status()
        entries = await resp.json()
        for entry in entries:
            if entry.get("domain") == "langchain_conversation":
                logger.info(
                    "langchain_conversation config entry already exists: %s",
                    entry.get("title"),
                )
                return

    # Start config flow
    async with session.post(
        f"{HA_BASE_URL}/api/config/config_entries/flow",
        headers=auth_headers,
        json={"handler": "langchain_conversation"},
    ) as resp:
        resp.raise_for_status()
        flow = await resp.json()
        flow_id = flow["flow_id"]
        logger.info("Started config flow %s", flow_id)

    # Complete the flow with our test server URL
    async with session.post(
        f"{HA_BASE_URL}/api/config/config_entries/flow/{flow_id}",
        headers=auth_headers,
        json={
            "url": f"http://host.docker.internal:{TEST_SERVER_PORT}",
            "timeout": 90,
            "verify_ssl": False,
            "streaming": False,
        },
    ) as resp:
        resp.raise_for_status()
        entry = await resp.json()
        logger.info("Config entry created: %s", entry.get("title", entry))


# ---------------------------------------------------------------------------
# HA async client (session-scoped)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session")
async def ha_client(ha_setup) -> HAClient:
    """Provide an authenticated HAClient for the test session."""
    client = HAClient(HA_BASE_URL, ha_setup["token"])
    yield client
    await client.close()


# ---------------------------------------------------------------------------
# Discover the langchain agent entity_id (session-scoped)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session")
async def langchain_agent_id(ha_client: HAClient) -> str:
    """Find the conversation entity registered by our custom component."""
    states = await ha_client.get_all_states()
    convos = [s for s in states if s["entity_id"].startswith("conversation.")]

    # First pass: look for "langchain" in entity_id or friendly_name
    for state in convos:
        eid = state["entity_id"]
        friendly = state.get("attributes", {}).get("friendly_name", "")
        if "langchain" in eid.lower() or "langchain" in friendly.lower():
            logger.info("Discovered langchain agent: %s", eid)
            return eid

    # Second pass: any conversation entity that isn't the built-in HA one
    for state in convos:
        eid = state["entity_id"]
        if eid != "conversation.home_assistant":
            logger.info("Discovered langchain agent (non-builtin): %s", eid)
            return eid

    raise RuntimeError(
        f"Could not find langchain conversation entity. "
        f"Available: {[s['entity_id'] for s in convos]}"
    )


# ---------------------------------------------------------------------------
# Warmup: trigger NLI model loading (session-scoped, autouse)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session", autouse=True)
async def warmup(ha_client: HAClient, langchain_agent_id: str):
    """Send a throwaway query to trigger model loading so first real test isn't slow."""
    try:
        await ha_client.send_conversation("hello", agent_id=langchain_agent_id)
        logger.info("Warmup query complete")
    except Exception as exc:
        logger.warning("Warmup query failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# State manager (session-scoped)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session")
async def state_manager(ha_client: HAClient) -> StateManager:
    return StateManager(ha_client)


# ---------------------------------------------------------------------------
# LLM judge (session-scoped)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def llm_judge() -> LLMJudge:
    model = os.environ.get("JUDGE_MODEL", "qwen2.5:3b")
    return LLMJudge(model=model)


# ---------------------------------------------------------------------------
# Reset demo light states before each test (function-scoped, autouse)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(autouse=True)
async def reset_states(state_manager: StateManager):
    """Reset all demo lights to their initial states before each test."""
    await state_manager.reset_all_lights()
    # Small delay for HA to settle
    await asyncio.sleep(0.5)
