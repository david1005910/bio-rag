#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# validate.sh - Full validation pipeline for the Chinese Learning App
#
# Usage:
#   ./scripts/validate.sh            # run unit tests + E2E tests
#   ./scripts/validate.sh --unit     # unit tests only
#   ./scripts/validate.sh --e2e      # E2E tests only (app must already be running)
#   ./scripts/validate.sh --e2e --start-app  # start app automatically + E2E
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/opt/anaconda3/envs/chinese/bin/python}"
APP_URL="http://localhost:8501"
APP_PID=""

# ── Colors ─────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; RESET="\033[0m"
info()    { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
section() { echo -e "\n${GREEN}══════════════════════════════════════${RESET}"; echo -e "${GREEN} $*${RESET}"; echo -e "${GREEN}══════════════════════════════════════${RESET}"; }

# ── Argument parsing ────────────────────────────────────────────────────────
RUN_UNIT=true
RUN_E2E=true
START_APP=false

for arg in "$@"; do
  case "$arg" in
    --unit)       RUN_UNIT=true;  RUN_E2E=false ;;
    --e2e)        RUN_UNIT=false; RUN_E2E=true  ;;
    --start-app)  START_APP=true                ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

cd "$PROJECT_ROOT"

# ── Cleanup on exit ─────────────────────────────────────────────────────────
cleanup() {
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    info "Stopping Streamlit app (PID $APP_PID)..."
    kill "$APP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── Wait for app ─────────────────────────────────────────────────────────────
wait_for_app() {
  info "Waiting for Streamlit at $APP_URL ..."
  for i in $(seq 1 30); do
    if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('$APP_URL', timeout=2)" 2>/dev/null; then
      info "App is ready!"
      return 0
    fi
    sleep 1
  done
  error "App did not start within 30 seconds"
  return 1
}

# ── Start app if requested ────────────────────────────────────────────────────
start_streamlit() {
  info "Starting Streamlit app..."
  "$PYTHON" -m streamlit run src/ui/app.py \
    --server.headless true \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    > /tmp/streamlit_validate.log 2>&1 &
  APP_PID=$!
  wait_for_app
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit Tests
# ─────────────────────────────────────────────────────────────────────────────
if $RUN_UNIT; then
  section "Unit Tests"
  "$PYTHON" -m pytest tests/unit/ -v --tb=short 2>&1
  info "Unit tests passed!"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. E2E Tests (Playwright)
# ─────────────────────────────────────────────────────────────────────────────
if $RUN_E2E; then
  section "E2E Tests (Playwright)"

  # Start app if requested, otherwise check it's already running
  if $START_APP; then
    start_streamlit
  else
    if ! "$PYTHON" -c "import urllib.request; urllib.request.urlopen('$APP_URL', timeout=2)" 2>/dev/null; then
      warn "Streamlit app not detected at $APP_URL"
      info "Starting app automatically..."
      start_streamlit
      START_APP=true  # ensure cleanup kills it
    else
      info "Detected running app at $APP_URL"
    fi
  fi

  # Run playwright E2E tests
  "$PYTHON" -m pytest tests/e2e/ -v --tb=short \
    --base-url "$APP_URL" \
    --browser chromium \
    --headed=false \
    2>&1
  info "E2E tests passed!"
fi

section "All validations passed!"
echo ""
echo "  Unit tests : $(if $RUN_UNIT; then echo 'OK'; else echo 'skipped'; fi)"
echo "  E2E tests  : $(if $RUN_E2E; then echo 'OK'; else echo 'skipped'; fi)"
echo ""
