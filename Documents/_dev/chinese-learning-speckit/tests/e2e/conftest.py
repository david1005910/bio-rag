"""
E2E test configuration for Playwright tests.
Requires Streamlit app to be running on localhost:8501.
"""
import pytest
import subprocess
import time
import sys
import os

APP_URL = "http://localhost:8501"


@pytest.fixture(scope="session")
def streamlit_app():
    """Start Streamlit app for the test session, yield URL, then stop."""
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    python = sys.executable

    proc = subprocess.Popen(
        [python, "-m", "streamlit", "run", "src/ui/app.py",
         "--server.headless", "true",
         "--server.port", "8501",
         "--browser.gatherUsageStats", "false"],
        cwd=os.path.normpath(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait until the app is ready
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(APP_URL, timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        proc.kill()
        pytest.fail(f"Streamlit app did not start at {APP_URL}")

    yield APP_URL

    proc.terminate()
    proc.wait(timeout=10)
