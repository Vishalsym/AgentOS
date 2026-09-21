"""Launch the Sentari control dashboard.

Usage:
    uv run python scripts/dashboard.py
    then open http://127.0.0.1:8000 in a browser.

Starts a real kernel and runs the same scenario as scripts/demo.py, live,
in the background -- the page polls actual kernel/SQLite state twice a
second. Use the "Restart scenario" button to run it again.
"""

from __future__ import annotations

from sentari.dashboard.app import main

if __name__ == "__main__":
    main()
