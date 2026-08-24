"""Installed-artifact smoke test for the public fixed-seed example."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.installed
def test_built_archives_run_public_example() -> None:
    """Delegate artifact inspection, installation, and execution to one runner."""
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(repository / "scripts" / "smoke_wheel.py")],
        cwd=repository,
        check=True,
        shell=False,
    )
