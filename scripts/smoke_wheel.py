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
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
DIST = REPOSITORY / "dist"
EXAMPLE = REPOSITORY / "examples" / "known_nuisance_fit.py"
QUANTITIES = {"h", "p", "alpha", "beta", "tau_plus", "tau_minus"}
VENV_TIMEOUT_SECONDS = 180.0
INSTALL_TIMEOUT_SECONDS = 600.0
IMPORT_TIMEOUT_SECONDS = 60.0
EXAMPLE_TIMEOUT_SECONDS = 180.0
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
        if path.is_file() and path.name.endswith(".tar.gz")
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


def _validated_archive_path(artifact: Path, name: str, *, subject: str) -> str:
    """Return one normalized member path after rejecting extraction hazards."""
    if not name or "\x00" in name:
        raise RuntimeError(f"invalid {subject} in {artifact.name}: {name!r}")
    if "\\" in name:
        raise RuntimeError(
            f"backslash is forbidden in {subject} in {artifact.name}: {name}"
        )
    if name.startswith("/") or PureWindowsPath(name).drive:
        raise RuntimeError(f"absolute {subject} in {artifact.name}: {name}")

    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"parent traversal in {subject} in {artifact.name}: {name}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise RuntimeError(f"invalid {subject} in {artifact.name}: {name!r}")
    return normalized


def _assert_members_safe(
    artifact: Path, members: Iterable[str], *, wheel: bool
) -> None:
    normalized = [
        _validated_archive_path(artifact, name, subject="archive member")
        for name in members
    ]
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
        members = archive.getmembers()
        _assert_members_safe(sdist, (member.name for member in members), wheel=False)
        for member in members:
            if member.issym() or member.islnk():
                _validated_archive_path(
                    sdist,
                    member.linkname,
                    subject=f"link target for {member.name}",
                )
            if member.ischr() or member.isblk() or member.isfifo():
                raise RuntimeError(
                    f"special archive member in {sdist.name}: {member.name}"
                )


def _venv_python(environment: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    # Do not resolve the POSIX venv launcher: it is normally a symlink to the
    # base interpreter, and resolving it bypasses the adjacent pyvenv.cfg.
    executable = environment / relative
    if not executable.is_file():
        raise RuntimeError(f"virtual-environment interpreter is missing: {executable}")
    return executable


def _run(
    command: list[str],
    *,
    cwd: Path,
    stage: str,
    timeout_seconds: float,
    capture: bool = False,
) -> str:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=True,
            shell=False,
            capture_output=capture,
            text=capture,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{stage} exceeded its {timeout_seconds:g}-second timeout"
        ) from error
    return completed.stdout if capture else ""


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"installed example returned nonnumeric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"installed example returned nonfinite {name}")
    return result


def _strict_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"installed example returned noninteger {name}")
    if value < 0:
        raise RuntimeError(f"installed example returned negative {name}")
    return value


def _validate_parameter_value(quantity: str, label: str, value: float) -> None:
    if quantity == "alpha" and not 0.0 < value <= 2.0:
        raise RuntimeError(f"installed example returned out-of-domain alpha {label}")
    if quantity == "beta" and not -1.0 <= value <= 1.0:
        raise RuntimeError(f"installed example returned out-of-domain beta {label}")
    if quantity == "p" and not 0.0 <= value <= 1.0:
        raise RuntimeError(f"installed example returned out-of-domain p {label}")
    if quantity in {"h", "tau_plus", "tau_minus"} and value < 0.0:
        raise RuntimeError(
            f"installed example returned out-of-domain {quantity} {label}"
        )


def _validate_parameter_summary(quantity: str, summary: object) -> None:
    if not isinstance(summary, dict) or set(summary) != {
        "mean",
        "median",
        "credible_interval",
    }:
        raise RuntimeError(f"installed example returned malformed {quantity} summary")
    interval = summary["credible_interval"]
    if not isinstance(interval, dict) or set(interval) != {"lower", "upper", "mass"}:
        raise RuntimeError(f"installed example returned malformed {quantity} interval")

    mean = _finite_float(f"{quantity} mean", summary["mean"])
    median = _finite_float(f"{quantity} median", summary["median"])
    lower = _finite_float(f"{quantity} lower", interval["lower"])
    upper = _finite_float(f"{quantity} upper", interval["upper"])
    mass = _finite_float(f"{quantity} interval mass", interval["mass"])
    for label, value in (
        ("mean", mean),
        ("median", median),
        ("lower", lower),
        ("upper", upper),
    ):
        _validate_parameter_value(quantity, label, value)
    if not lower <= median <= upper:
        raise RuntimeError(
            f"installed example returned unordered {quantity} interval/median"
        )
    if not 0.0 < mass < 1.0:
        raise RuntimeError(
            f"installed example returned invalid {quantity} credible mass"
        )


def _validate_example(payload: dict[str, Any]) -> None:
    if payload.get("status") != "research_uncertified":
        raise RuntimeError("installed example returned an unexpected status")
    if payload.get("method") != "exact_finite_three_cell":
        raise RuntimeError("installed example returned an unexpected method")
    if payload.get("parameterization") != "S0":
        raise RuntimeError("installed example did not report S0")
    nuisance = payload.get("known_nuisance")
    if not isinstance(nuisance, dict) or set(nuisance) != {
        "loc",
        "scale",
        "mode",
        "provenance",
    }:
        raise RuntimeError("installed example returned malformed nuisance provenance")
    if nuisance.get("mode") != "externally_known":
        raise RuntimeError("installed example did not retain fixed nuisance provenance")
    _finite_float("known location", nuisance["loc"])
    if _finite_float("known scale", nuisance["scale"]) <= 0.0:
        raise RuntimeError("installed example returned a nonpositive known scale")
    provenance = nuisance["provenance"]
    if not isinstance(provenance, str) or not provenance.strip():
        raise RuntimeError("installed example returned invalid nuisance provenance")

    counts = payload.get("counts")
    count_names = {"n_minus", "n_zero", "n_plus", "n"}
    if not isinstance(counts, dict) or set(counts) != count_names:
        raise RuntimeError("installed example returned invalid cell counts")
    validated_counts = {
        name: _strict_nonnegative_int(f"cell count {name}", counts[name])
        for name in count_names
    }
    if (
        validated_counts["n"] == 0
        or sum(validated_counts[name] for name in ("n_minus", "n_zero", "n_plus"))
        != validated_counts["n"]
    ):
        raise RuntimeError("installed example returned invalid cell counts")

    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != QUANTITIES:
        raise RuntimeError("installed example did not return all six summaries")
    for quantity, summary in parameters.items():
        _validate_parameter_summary(quantity, summary)

    mass = payload.get("posterior_mass")
    if abs(_finite_float("posterior mass", mass) - 1.0) > 1e-12:
        raise RuntimeError("installed example posterior mass is not normalized")

    refinement = payload.get("refinement")
    if not isinstance(refinement, dict) or refinement.get("converged") is not True:
        raise RuntimeError(
            "installed example did not retain passing refinement evidence"
        )
    tolerance = _finite_float("refinement tolerance", refinement.get("tolerance"))
    total_variation = _finite_float(
        "refinement total variation", refinement.get("joint_total_variation")
    )
    normalizer_change = _finite_float(
        "refinement log-normalizer change",
        refinement.get("log_normalizer_change"),
    )
    common_grid_points = _strict_nonnegative_int(
        "refinement common grid points", refinement.get("common_grid_points")
    )
    if tolerance <= 0.0 or not 0.0 <= total_variation <= tolerance:
        raise RuntimeError("installed example returned invalid refinement accuracy")
    if normalizer_change < 0.0 or common_grid_points < 3:
        raise RuntimeError("installed example returned invalid refinement diagnostics")


def _exercise_archive(artifact: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stableboundary-smoke-") as temporary:
        root = Path(temporary).resolve()
        if root.is_relative_to(REPOSITORY.resolve()):
            raise RuntimeError("temporary smoke environment is inside the repository")
        environment = root / "venv"
        work = root / "work"
        work.mkdir()
        _run(
            [sys.executable, "-m", "venv", str(environment)],
            cwd=work,
            stage="virtual-environment creation",
            timeout_seconds=VENV_TIMEOUT_SECONDS,
        )
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
            stage=f"installation of {artifact.name}",
            timeout_seconds=INSTALL_TIMEOUT_SECONDS,
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
                stage=f"import check for {artifact.name}",
                timeout_seconds=IMPORT_TIMEOUT_SECONDS,
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
                stage=f"public example for {artifact.name}",
                timeout_seconds=EXAMPLE_TIMEOUT_SECONDS,
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
