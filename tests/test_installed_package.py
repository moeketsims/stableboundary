"""Installed-artifact smoke test for the public fixed-seed example."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SMOKE_RUNNER_TIMEOUT_SECONDS = 1_800.0
SYSTEM_ENVIRONMENT_KEYS = frozenset({"SYSTEMROOT", "WINDIR"})


def _outer_environment(root: Path) -> dict[str, str]:
    environment = {
        name: os.environ[name] for name in SYSTEM_ENVIRONMENT_KEYS if name in os.environ
    }
    executable_directory = str(Path(sys.executable).resolve().parent)
    if os.name == "nt" and "SYSTEMROOT" in environment:
        system32 = str(Path(environment["SYSTEMROOT"]) / "System32")
        environment["PATH"] = os.pathsep.join((executable_directory, system32))
    else:
        environment["PATH"] = executable_directory
    environment.update(
        {
            "HOME": str(root),
            "USERPROFILE": str(root),
            "TEMP": str(root),
            "TMP": str(root),
            "TMPDIR": str(root),
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONUTF8": "1",
        }
    )
    return environment


def _run_artifact_smoke(repository: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="stableboundary-outer-") as temporary:
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(repository / "scripts" / "smoke_wheel.py"),
                ],
                cwd=repository,
                env=_outer_environment(Path(temporary)),
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


def test_outer_runner_ignores_hostile_startup_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    completed = repository / "completed.txt"
    marker = repository / "startup-hook-ran.txt"
    (scripts / "smoke_wheel.py").write_text(
        f"from pathlib import Path\nPath({str(completed)!r}).write_text('ok')\n",
        encoding="utf-8",
    )
    hook = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    (scripts / "sitecustomize.py").write_text(hook, encoding="utf-8")
    hostile_path = tmp_path / "hostile-python-path"
    hostile_path.mkdir()
    (hostile_path / "sitecustomize.py").write_text(hook, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(hostile_path))
    monkeypatch.setenv("PYTHONSTARTUP", str(hostile_path / "sitecustomize.py"))

    _run_artifact_smoke(repository)

    assert completed.read_text(encoding="utf-8") == "ok"
    assert not marker.exists()


@pytest.mark.installed
def test_built_archives_run_public_example() -> None:
    """Delegate artifact inspection, installation, and execution to one runner."""
    repository = Path(__file__).resolve().parents[1]
    _run_artifact_smoke(repository)
