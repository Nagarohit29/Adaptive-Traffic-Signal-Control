from pathlib import Path

import pytest

from traffic_rl.baselines.fixed_timer import tuned_timer

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def test_app_runs_and_shows_all_controllers() -> None:
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    tuned_label, _ = tuned_timer(ROOT / "results" / "evaluation.csv")
    assert [h.value for h in app.subheader][:3] == [
        "Q-learning",
        tuned_label,
        "Fixed timer 30/30 s",
    ]
    assert len(app.metric) == 12
    assert all(m.value for m in app.metric)
