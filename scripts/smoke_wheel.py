"""Inspect and exercise both built stableboundary distribution archives."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
DIST = REPOSITORY / "dist"
EXAMPLE = REPOSITORY / "examples" / "known_nuisance_fit.py"
QUANTITIES = {"h", "p", "alpha", "beta", "tau_plus", "tau_minus"}
FORBIDDEN_PARTS = {
    ".planning",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "tests",
}
FORBIDDEN_SUFFIXES = (
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pyc",
    ".pyo",
    ".run.xml",
    ".synctex.gz",
    ".tex",
    ".toc",
    ".xdv",
)


def _archives() -> tuple[Path, Path]:
    if not DIST.is_dir():
        raise RuntimeError(f"distribution directory does not exist: {DIST}")
    wheels = sorted(path.resolve() for path in DIST.glob("*.whl") if path.is_file())
    sdists = sorted(
        path.resolve()
        for path in DIST.iterdir()
        if path.is_file()
        and (path.name.endswith(".tar.gz") or path.suffix.lower() == ".zip")
    )
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            "dist must contain exactly one wheel and one sdist; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )
    for artifact in (*wheels, *sdists):
        if not artifact.is_relative_to(DIST.resolve()):
            raise RuntimeError(
                f"artifact escaped the repository dist directory: {artifact}"
            )
    return wheels[0], sdists[0]


def _assert_members_safe(artifact: Path, members: list[str], *, wheel: bool) -> None:
    normalized = [name.replace("\\", "/").lstrip("./") for name in members]
    for name in normalized:
        path = PurePosixPath(name)
        lowered_parts = {part.lower() for part in path.parts}
        lowered_name = name.lower()
        if lowered_parts & FORBIDDEN_PARTS:
            raise RuntimeError(f"forbidden directory in {artifact.name}: {name}")
        if lowered_name.endswith(FORBIDDEN_SUFFIXES):
            raise RuntimeError(f"forbidden file in {artifact.name}: {name}")
        if "gaussian_boundary_stable_manuscript" in lowered_name:
            raise RuntimeError(f"manuscript leaked into {artifact.name}: {name}")
    if wheel and "stableboundary/py.typed" not in normalized:
        raise RuntimeError(f"{artifact.name} does not contain stableboundary/py.typed")


def _inspect_archives(wheel: Path, sdist: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        _assert_members_safe(wheel, archive.namelist(), wheel=True)
    with tarfile.open(sdist, mode="r:*") as archive:
        _assert_members_safe(sdist, archive.getnames(), wheel=False)


def _venv_python(environment: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    executable = (environment / relative).resolve()
    if not executable.is_file():
        raise RuntimeError(f"virtual-environment interpreter is missing: {executable}")
    return executable


def _run(command: list[str], *, cwd: Path, capture: bool = False) -> str:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        shell=False,
        capture_output=capture,
        text=capture,
    )
    return completed.stdout if capture else ""


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"installed example returned nonnumeric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"installed example returned nonfinite {name}")
    return result


def _validate_example(payload: dict[str, Any]) -> None:
    if payload.get("status") != "research_uncertified":
        raise RuntimeError("installed example returned an unexpected status")
    if payload.get("parameterization") != "S0":
        raise RuntimeError("installed example did not report S0")
    nuisance = payload.get("known_nuisance")
    if not isinstance(nuisance, dict) or nuisance.get("mode") != "externally_known":
        raise RuntimeError("installed example did not retain fixed nuisance provenance")
    counts = payload.get("counts")
    if not isinstance(counts, dict) or sum(
        int(counts[name]) for name in ("n_minus", "n_zero", "n_plus")
    ) != int(counts["n"]):
        raise RuntimeError("installed example returned invalid cell counts")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != QUANTITIES:
        raise RuntimeError("installed example did not return all six summaries")
    for quantity, summary in parameters.items():
        interval = (
            summary.get("credible_interval") if isinstance(summary, dict) else None
        )
        values = (
            summary.get("mean") if isinstance(summary, dict) else None,
            summary.get("median") if isinstance(summary, dict) else None,
            interval.get("lower") if isinstance(interval, dict) else None,
            interval.get("upper") if isinstance(interval, dict) else None,
        )
        for label, value in zip(
            ("mean", "median", "lower", "upper"), values, strict=True
        ):
            _finite_float(f"{quantity} {label}", value)
    mass = payload.get("posterior_mass")
    if abs(_finite_float("posterior mass", mass) - 1.0) > 1e-12:
        raise RuntimeError("installed example posterior mass is not normalized")
    refinement = payload.get("refinement")
    if not isinstance(refinement, dict) or refinement.get("converged") is not True:
        raise RuntimeError(
            "installed example did not retain passing refinement evidence"
        )


def _exercise_archive(artifact: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stableboundary-smoke-") as temporary:
        root = Path(temporary).resolve()
        if root.is_relative_to(REPOSITORY.resolve()):
            raise RuntimeError("temporary smoke environment is inside the repository")
        environment = root / "venv"
        work = root / "work"
        work.mkdir()
        venv.EnvBuilder(with_pip=True).create(environment)
        python = _venv_python(environment)
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(artifact),
            ],
            cwd=work,
        )
        origin = Path(
            _run(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import stableboundary as sb; print(sb.__file__)",
                ],
                cwd=work,
                capture=True,
            ).strip()
        ).resolve()
        if not origin.is_relative_to(environment) or origin.is_relative_to(REPOSITORY):
            raise RuntimeError(
                f"stableboundary imported from the wrong location: {origin}"
            )
        copied_example = work / EXAMPLE.name
        shutil.copy2(EXAMPLE, copied_example)
        decoded = json.loads(
            _run(
                [str(python), "-I", str(copied_example)],
                cwd=work,
                capture=True,
            )
        )
        if not isinstance(decoded, dict):
            raise RuntimeError("installed example did not return a JSON object")
        payload: dict[str, Any] = decoded
        _validate_example(payload)
        print(
            json.dumps(
                {
                    "artifact": artifact.name,
                    "origin": str(origin),
                    "status": payload["status"],
                    "counts": payload["counts"],
                    "posterior_mass": payload["posterior_mass"],
                    "joint_total_variation": payload["refinement"][
                        "joint_total_variation"
                    ],
                },
                sort_keys=True,
            )
        )
        return payload


def main() -> None:
    """Inspect, install, and run the real example from both fresh archives."""
    wheel, sdist = _archives()
    _inspect_archives(wheel, sdist)
    _exercise_archive(wheel)
    _exercise_archive(sdist)


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        tarfile.TarError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as error:
        print(f"artifact smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
