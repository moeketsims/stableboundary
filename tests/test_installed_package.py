"""Installed-artifact smoke test for the public fixed-seed example."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SMOKE_RUNNER_TIMEOUT_SECONDS = 1_800.0


def _run_artifact_smoke(repository: Path) -> None:
    try:
        subprocess.run(
            [sys.executable, str(repository / "scripts" / "smoke_wheel.py")],
            cwd=repository,
            check=True,
            shell=False,
            timeout=SMOKE_RUNNER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "artifact smoke runner exceeded its "
            f"{SMOKE_RUNNER_TIMEOUT_SECONDS:g}-second timeout"
        ) from error


def test_artifact_smoke_runner_reports_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def time_out(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=[sys.executable, "smoke_wheel.py"],
            timeout=SMOKE_RUNNER_TIMEOUT_SECONDS,
        )

    monkeypatch.setattr(subprocess, "run", time_out)

    with pytest.raises(RuntimeError, match="1800-second timeout"):
        _run_artifact_smoke(tmp_path)


@pytest.mark.installed
def test_built_archives_run_public_example() -> None:
    """Delegate artifact inspection, installation, and execution to one runner."""
    repository = Path(__file__).resolve().parents[1]
    _run_artifact_smoke(repository)
