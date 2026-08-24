"""Focused regression tests for the distribution-artifact smoke runner."""

from __future__ import annotations

import copy
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import smoke_wheel


def _valid_summary(quantity: str) -> dict[str, object]:
    values = {
        "h": (1.0, 1.5, 2.0),
        "p": (0.2, 0.5, 0.8),
        "alpha": (1.5, 1.75, 1.9),
        "beta": (-0.5, 0.0, 0.5),
        "tau_plus": (0.1, 0.5, 1.0),
        "tau_minus": (0.1, 0.5, 1.0),
    }
    lower, median, upper = values[quantity]
    return {
        "mean": median,
        "median": median,
        "credible_interval": {"lower": lower, "upper": upper, "mass": 0.9},
    }


def _valid_payload() -> dict[str, object]:
    return {
        "status": "research_uncertified",
        "method": "exact_finite_three_cell",
        "parameterization": "S0",
        "known_nuisance": {
            "loc": 0.0,
            "scale": 1.0,
            "mode": "externally_known",
            "provenance": "fixed independently",
        },
        "counts": {"n_minus": 1, "n_zero": 8, "n_plus": 1, "n": 10},
        "parameters": {
            quantity: _valid_summary(quantity) for quantity in smoke_wheel.QUANTITIES
        },
        "posterior_mass": 1.0,
        "refinement": {
            "converged": True,
            "tolerance": 0.002,
            "joint_total_variation": 0.001,
            "log_normalizer_change": 1e-10,
            "common_grid_points": 65,
        },
    }


@pytest.mark.parametrize(
    "member",
    [
        "/absolute.py",
        "../../payload.py",
        "package/../payload.py",
        "C:/payload.py",
        "C:payload.py",
        "package\\payload.py",
        "\\\\server\\share\\payload.py",
    ],
)
def test_archive_member_paths_reject_extraction_hazards(member: str) -> None:
    artifact = Path("hostile.whl")
    with pytest.raises(RuntimeError):
        smoke_wheel._assert_members_safe(artifact, [member], wheel=False)


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
@pytest.mark.parametrize(
    "target",
    ["../../outside.py", "/outside.py", "C:/outside.py", "dir\\outside.py"],
)
def test_sdist_rejects_unsafe_link_targets(
    tmp_path: Path, link_type: bytes, target: str
) -> None:
    wheel = tmp_path / "stableboundary.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("stableboundary/py.typed", "")

    sdist = tmp_path / "stableboundary.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = b"safe"
        regular = tarfile.TarInfo("stableboundary-0.1.0/safe.txt")
        regular.size = len(payload)
        archive.addfile(regular, io.BytesIO(payload))
        link = tarfile.TarInfo("stableboundary-0.1.0/link.txt")
        link.type = link_type
        link.linkname = target
        archive.addfile(link)

    with pytest.raises(RuntimeError, match="link target"):
        smoke_wheel._inspect_archives(wheel, sdist)


def test_sdist_discovery_ignores_zip_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "stableboundary.whl").touch()
    expected_sdist = tmp_path / "stableboundary.tar.gz"
    expected_sdist.touch()
    (tmp_path / "unexpected.zip").touch()
    monkeypatch.setattr(smoke_wheel, "DIST", tmp_path)

    wheel, sdist = smoke_wheel._archives()

    assert wheel == (tmp_path / "stableboundary.whl").resolve()
    assert sdist == expected_sdist.resolve()


def test_installed_payload_requires_exact_finite_cell_method() -> None:
    payload = _valid_payload()
    payload["method"] = "limiting_poisson"

    with pytest.raises(RuntimeError, match="unexpected method"):
        smoke_wheel._validate_example(payload)


@pytest.mark.parametrize("bad_count", ["1", 1.0, True, -1])
def test_installed_payload_rejects_noninteger_or_negative_counts(
    bad_count: object,
) -> None:
    payload = _valid_payload()
    counts = payload["counts"]
    assert isinstance(counts, dict)
    counts["n_minus"] = bad_count

    with pytest.raises(RuntimeError, match="cell count"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_rejects_inconsistent_count_total() -> None:
    payload = _valid_payload()
    counts = payload["counts"]
    assert isinstance(counts, dict)
    counts["n"] = 11

    with pytest.raises(RuntimeError, match="invalid cell counts"):
        smoke_wheel._validate_example(payload)


@pytest.mark.parametrize(
    ("quantity", "location", "bad_value"),
    [
        ("alpha", "upper", 2.1),
        ("beta", "lower", -1.1),
        ("p", "mean", 1.1),
        ("h", "lower", -0.1),
        ("tau_plus", "median", -0.1),
    ],
)
def test_installed_payload_rejects_parameter_domain_violations(
    quantity: str, location: str, bad_value: float
) -> None:
    payload = _valid_payload()
    parameters = payload["parameters"]
    assert isinstance(parameters, dict)
    summary = parameters[quantity]
    assert isinstance(summary, dict)
    if location in {"lower", "upper"}:
        interval = summary["credible_interval"]
        assert isinstance(interval, dict)
        interval[location] = bad_value
    else:
        summary[location] = bad_value

    with pytest.raises(RuntimeError, match="out-of-domain"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_rejects_unordered_interval() -> None:
    payload = copy.deepcopy(_valid_payload())
    parameters = payload["parameters"]
    assert isinstance(parameters, dict)
    summary = parameters["p"]
    assert isinstance(summary, dict)
    interval = summary["credible_interval"]
    assert isinstance(interval, dict)
    interval["lower"] = 0.6

    with pytest.raises(RuntimeError, match="unordered p"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_rejects_invalid_interval_mass() -> None:
    payload = _valid_payload()
    parameters = payload["parameters"]
    assert isinstance(parameters, dict)
    summary = parameters["alpha"]
    assert isinstance(summary, dict)
    interval = summary["credible_interval"]
    assert isinstance(interval, dict)
    interval["mass"] = 1.0

    with pytest.raises(RuntimeError, match="credible mass"):
        smoke_wheel._validate_example(payload)


def test_stage_subprocess_timeout_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def time_out(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=17.0)

    monkeypatch.setattr(smoke_wheel.subprocess, "run", time_out)

    with pytest.raises(RuntimeError, match="installation.*17-second timeout"):
        smoke_wheel._run(
            ["python"],
            cwd=tmp_path,
            stage="installation",
            timeout_seconds=17.0,
        )
