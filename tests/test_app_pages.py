"""Smoke test: every dashboard page executes without raising."""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = [
    "views/home.py",
    "views/health.py",
    "views/trends.py",
    "views/forecast.py",
    "views/scenario.py",
    "views/research.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception, f"{page} raised: {[str(e.value) for e in at.exception]}"
