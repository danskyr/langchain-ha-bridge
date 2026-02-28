#!/usr/bin/env bash
#
# Run integration tests with prerequisite checking.
#
# Usage:
#   ./scripts/run_integration_tests.sh                  # run all integration tests
#   ./scripts/run_integration_tests.sh test_light        # pass extra args to pytest
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colours (disabled when piped)
if [ -t 1 ]; then
  GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
else
  GREEN=''; RED=''; YELLOW=''; CYAN=''; NC=''
fi

ok()   { printf "${GREEN}  ✔ %s${NC}\n" "$*"; }
fail() { printf "${RED}  ✘ %s${NC}\n" "$*"; }
warn() { printf "${YELLOW}  ⚠ %s${NC}\n" "$*"; }
info() { printf "${CYAN}  ℹ %s${NC}\n" "$*"; }
step() { printf "\n${CYAN}▸ %s${NC}\n" "$*"; }

FAILED=0

# ── 1. Check prerequisites ────────────────────────────────────────────────────

step "Checking prerequisites"

# 1a. Docker
if command -v docker &>/dev/null && docker info &>/dev/null; then
  ok "Docker is running"
else
  fail "Docker is not running or not installed"
  FAILED=1
fi

# 1b. Poetry
if command -v poetry &>/dev/null; then
  ok "Poetry found"
else
  fail "Poetry not found — install it: https://python-poetry.org/docs/"
  FAILED=1
fi

# 1c. .env file
if [ -f "$PROJECT_ROOT/.env" ]; then
  ok ".env file exists"
else
  fail ".env file not found (copy from .env.shape and fill in keys)"
  FAILED=1
fi

# 1d. Ollama
OLLAMA_URL="http://localhost:11434"
if curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
  ok "Ollama is running at $OLLAMA_URL"
else
  warn "Ollama is not running — attempting to start it"
  if command -v ollama &>/dev/null; then
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # Wait up to 15 seconds for Ollama to come online
    for i in $(seq 1 15); do
      if curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
      ok "Ollama started (pid $OLLAMA_PID)"
    else
      fail "Could not start Ollama"
      FAILED=1
    fi
  else
    fail "Ollama not installed — https://ollama.com"
    FAILED=1
  fi
fi

# 1e. qwen2.5:3b model
if curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
  if ollama list 2>/dev/null | grep -q "qwen2.5:3b"; then
    ok "Model qwen2.5:3b is available"
  else
    warn "Model qwen2.5:3b not found — pulling it now (this may take a few minutes)"
    if ollama pull qwen2.5:3b; then
      ok "Model qwen2.5:3b pulled successfully"
    else
      fail "Failed to pull qwen2.5:3b"
      FAILED=1
    fi
  fi
fi

# 1f. Port 8123 free (dev HA container would conflict)
if lsof -i :8123 -sTCP:LISTEN &>/dev/null 2>&1; then
  # Check if it's the dev HA container
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^homeassistant$"; then
    warn "Dev HA container is using port 8123 — stopping it for tests"
    docker stop homeassistant >/dev/null 2>&1 || true
    STOPPED_DEV_HA=1
    # Wait for port to free up
    for i in $(seq 1 10); do
      if ! lsof -i :8123 -sTCP:LISTEN &>/dev/null 2>&1; then break; fi
      sleep 1
    done
    if lsof -i :8123 -sTCP:LISTEN &>/dev/null 2>&1; then
      fail "Port 8123 is still in use after stopping dev container"
      FAILED=1
    else
      ok "Port 8123 is now free"
    fi
  else
    fail "Port 8123 is in use by another process (needed for test HA container)"
    lsof -i :8123 -sTCP:LISTEN 2>/dev/null || true
    FAILED=1
  fi
else
  ok "Port 8123 is free"
fi

# 1g. Port 8002 free (test langchain server)
if lsof -i :8002 -sTCP:LISTEN &>/dev/null 2>&1; then
  fail "Port 8002 is in use (needed for test langchain server)"
  lsof -i :8002 -sTCP:LISTEN 2>/dev/null || true
  FAILED=1
else
  ok "Port 8002 is free"
fi

# 1h. HA Docker image
HA_IMAGE="ghcr.io/home-assistant/home-assistant:stable"
if docker image inspect "$HA_IMAGE" &>/dev/null; then
  ok "HA Docker image is available"
else
  warn "HA Docker image not found — pulling it now"
  if docker pull "$HA_IMAGE"; then
    ok "HA Docker image pulled"
  else
    fail "Failed to pull HA Docker image"
    FAILED=1
  fi
fi

# Bail out if any hard prerequisite failed
if [ "$FAILED" -ne 0 ]; then
  printf "\n${RED}Prerequisites not met — fix the issues above and retry.${NC}\n"
  exit 1
fi

# ── 2. Install / sync dependencies ────────────────────────────────────────────

step "Installing dependencies"
poetry install --quiet 2>&1
ok "Dependencies installed"

# ── 3. Clean up stale test container ──────────────────────────────────────────

step "Cleaning up stale test resources"
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^ha_integration_test$"; then
  docker rm -f ha_integration_test >/dev/null 2>&1
  ok "Removed stale ha_integration_test container"
else
  ok "No stale test container found"
fi

# ── 4. Run tests ──────────────────────────────────────────────────────────────

step "Running integration tests"
info "Server will start on :8002, HA container on :8123"
info "First run may be slow (NLI model loading, HA boot)"
echo ""

# Pass through any extra arguments (e.g. specific test file, -k filter)
PYTEST_ARGS=("tests/integration/" "-v" "--timeout=180" "-s")
if [ $# -gt 0 ]; then
  PYTEST_ARGS+=("$@")
fi

TEST_EXIT=0
poetry run pytest "${PYTEST_ARGS[@]}" || TEST_EXIT=$?

# ── 5. Cleanup ────────────────────────────────────────────────────────────────

step "Cleaning up"

# Remove the test container if it's still around (conftest should do this, but just in case)
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^ha_integration_test$"; then
  docker rm -f ha_integration_test >/dev/null 2>&1
  ok "Removed ha_integration_test container"
fi

# Restart the dev HA container if we stopped it
if [ "${STOPPED_DEV_HA:-0}" = "1" ]; then
  info "Restarting dev HA container that was stopped for tests"
  docker start homeassistant >/dev/null 2>&1 || warn "Could not restart dev HA container"
fi

echo ""
if [ "$TEST_EXIT" -eq 0 ]; then
  printf "${GREEN}All tests passed!${NC}\n"
else
  printf "${RED}Some tests failed (exit code $TEST_EXIT).${NC}\n"
fi

exit "$TEST_EXIT"
