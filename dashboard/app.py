"""Streamlit entry point expected by ``make dashboard``.

Keeping this file at the repository root matches tech.md section 10.3. The
application itself lives in :mod:`tradingbot.dashboard.app` so it can be
imported and tested without Streamlit's file watcher.
"""

from tradingbot.dashboard.app import run

run()
