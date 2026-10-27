from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_runs_and_shows_all_controllers() -> None:
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert [h.value for h in app.subheader][:3] == [
        "Q-learning",
        "Fixed timer 30/30",
        "Fixed timer 15/15",
    ]
    assert len(app.metric) == 12
    assert all(m.value for m in app.metric)
