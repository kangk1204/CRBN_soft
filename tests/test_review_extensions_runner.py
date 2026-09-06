"""Execution records must distinguish unavailable science from failed execution."""
import json
from types import SimpleNamespace

import pytest

from scripts import run_review_extensions as runner


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(runner.DEFAULT_CONFIG.read_text())
    return path


def test_offline_and_output_are_forwarded(monkeypatch, config_path, tmp_path):
    calls = []
    def operation(config, output, offline):
        calls.append((config, output, offline))
        return {"eligible_external_open": 0}
    monkeypatch.setattr(runner, "load_stage", lambda _: SimpleNamespace(run=operation))
    out = tmp_path / "output"
    result = runner.run(config_path, out, True, ("external",))
    assert calls == [(config_path.resolve(), out / "analysis/external", True)]
    assert result["status"] == "complete"
    assert json.loads((out / "verification/analysis_run.json").read_text())["offline"] is True


def test_failed_execution_is_not_a_completed_stage(monkeypatch, config_path, tmp_path):
    def fail(*args, **kwargs):
        raise ArithmeticError("unexpected singular system")
    monkeypatch.setattr(runner, "load_stage", lambda _: SimpleNamespace(run=fail))
    out = tmp_path / "output"
    with pytest.raises(ArithmeticError):
        runner.run(config_path, out, True, ("window",))
    result = json.loads((out / "verification/analysis_run.json").read_text())
    assert result["status"] == "failed"
    assert "window" not in result["stages"]


def test_frozen_observable_and_output_config_cannot_drift(config_path, tmp_path):
    config = json.loads(config_path.read_text())
    config["window_extension"]["core_internal_dimension"] = 1041
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="801"):
        runner.run(config_path, tmp_path / "output", True, ("window",))
    assert not (tmp_path / "output").exists()


def test_different_frozen_configuration_is_rejected(config_path, tmp_path):
    out = tmp_path / "output"
    (out / "protocol").mkdir(parents=True)
    config = json.loads(config_path.read_text())
    config["seed"] += 1
    (out / "protocol/frozen_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="different frozen"):
        runner.run(config_path, out, True, ("window",))
