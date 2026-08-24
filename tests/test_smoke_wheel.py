"""Focused regression tests for the distribution-artifact smoke runner."""

from __future__ import annotations

import copy
import io
import math
import os
import subprocess
import sys
import tarfile
import unicodedata
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest

from scripts import smoke_wheel

METADATA = b"Metadata-Version: 2.4\nName: stableboundary\nVersion: 0.1.0\n"


def _write_minimal_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("stableboundary/py.typed", "")
        archive.writestr("stableboundary/core.py", "VALUE = 1\n")
        archive.writestr("stableboundary-0.1.0.dist-info/METADATA", METADATA)


def _add_tar_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _write_minimal_sdist(path: Path, *, example: bytes | None = None) -> None:
    example_content = smoke_wheel.EXAMPLE.read_bytes() if example is None else example
    with tarfile.open(path, "w:gz") as archive:
        root = smoke_wheel.EXPECTED_SDIST_ROOT
        _add_tar_bytes(archive, f"{root}/PKG-INFO", METADATA)
        _add_tar_bytes(archive, f"{root}/src/stableboundary/py.typed", b"")
        _add_tar_bytes(
            archive,
            f"{root}/src/stableboundary/core.py",
            b"VALUE = 1\n",
        )
        _add_tar_bytes(
            archive,
            f"{root}/examples/{smoke_wheel.EXAMPLE.name}",
            example_content,
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX venv launchers are symlinks")
def test_venv_python_preserves_posix_launcher_symlink(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    executable = environment / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(Path(sys.executable))

    selected = smoke_wheel._venv_python(environment)

    assert selected == executable
    assert selected.is_symlink()
    assert selected.resolve() != selected


def _valid_summary(quantity: str) -> dict[str, object]:
    r_value = 0.007771638764269451
    values = {
        "h": (1.0, 2.0, 3.0, 2.0),
        "p": (0.2, 0.5, 0.8, 0.5),
        "alpha": (
            2.0 - 3.0 * r_value,
            2.0 - 2.0 * r_value,
            2.0 - r_value,
            2.0 - 2.0 * r_value,
        ),
        "beta": (-0.6, 0.0, 0.6, 0.0),
        "tau_plus": (0.1 * r_value, r_value, 1.9 * r_value, r_value),
        "tau_minus": (0.1 * r_value, r_value, 1.9 * r_value, r_value),
    }
    lower, median, upper, mean = values[quantity]
    return {
        "mean": mean,
        "median": median,
        "credible_interval": {"lower": lower, "upper": upper, "mass": 0.9},
    }


def _valid_payload() -> dict[str, object]:
    r_value = 0.007771638764269451
    log_inverse_r = math.log(1.0 / r_value)
    threshold = 2.0 * math.sqrt(log_inverse_r + 2.0 * math.log(log_inverse_r))
    return {
        "schema_version": 1,
        "package_version": "0.1.0",
        "status": "research_uncertified",
        "method": "exact_finite_three_cell",
        "parameterization": "S0",
        "known_nuisance": {
            "loc": 0.0,
            "scale": 1.0,
            "mode": "externally_known",
            "provenance": "fixed independently",
        },
        "seed": 20_260_824,
        "truth": {
            "alpha": 2.0 - 1.5 * r_value,
            "beta": 0.35,
            "loc": 0.0,
            "scale": 1.0,
        },
        "design": {
            "n": 5_000,
            "c": 1.0,
            "r": r_value,
            "threshold": threshold,
            "formula_id": "critical-rate-lambertw-loglog-threshold",
            "formula_version": 1,
            "critical_rate_relative_residual": abs(
                5_000 * r_value / log_inverse_r - 8.0
            )
            / 8.0,
        },
        "prior": {
            "family": "compact_uniform_rectangle",
            "h_min": 0.25,
            "h_max": 4.0,
            "p_min": 0.05,
            "p_max": 0.95,
        },
        "counts": {
            "n_minus": 1,
            "n_zero": 4_996,
            "n_plus": 3,
            "n": 5_000,
            "threshold": threshold,
        },
        "quadrature": {
            "base_nodes": 20,
            "refined_nodes": 32,
            "interval_mass": 0.9,
            "log_normalizer": -17.0,
        },
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
            "summary_changes": {
                quantity: {
                    "mean": 1e-8,
                    "median": 1e-8,
                    "interval_lower": 1e-8,
                    "interval_upper": 1e-8,
                }
                for quantity in smoke_wheel.QUANTITIES
            },
            "predictive_tail": {"negative": 1e-8, "positive": 1e-8},
        },
        "identification": {
            "evidence_status": "two_sided_evidence",
            "precision_status": "not_assessed",
            "p_kl_divergence": 0.1,
            "p_interval_width_contraction": 0.2,
        },
        "backend": {
            "method": "scipy-piecewise-s0-direct-log-tails",
            "tolerance": 1.2e-14,
            "origin": "canonical_scipy_s0",
            "parameterization": "S0",
            "library": "scipy",
            "library_version": "1.18.0",
            "effective_settings": {
                "parameterization": "S0",
                "pdf_default_method": "piecewise",
                "cdf_default_method": "piecewise",
                "quad_eps": 1.2e-14,
                "piecewise_x_tol_near_zeta": 0.005,
                "piecewise_alpha_tol_near_one": 0.005,
                "pdf_fft_min_points_threshold": None,
                "pdf_fft_grid_spacing": 0.001,
                "pdf_fft_n_points_two_power": None,
                "pdf_fft_interpolation_level": 3,
                "pdf_fft_interpolation_degree": 3,
            },
        },
        "warnings": [
            "research_uncertified: not a certificate.",
            "Signed-tail evidence is two-sided; precision is not assessed.",
        ],
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
        "package//payload.py",
        "./package/payload.py",
        "package/./payload.py",
        "package/payload.py/",
        "package/payload.py.",
        "package/payload.py ",
        "package/payload:stream.py",
        "package/con.py",
        "package/AUX.txt",
        "package/lpt9.config",
        "package/control\x1f.py",
        "package/control\x85.py",
        "package/bidi\u202e.py",
        "package/delete\x7f.py",
        f"package/{unicodedata.normalize('NFD', 'é')}.py",
    ],
)
def test_archive_member_paths_reject_extraction_hazards(member: str) -> None:
    artifact = Path("hostile.whl")
    with pytest.raises(RuntimeError):
        smoke_wheel._assert_members_safe(artifact, [member], wheel=False)


@pytest.mark.parametrize(
    "members",
    [
        ["package/PAYLOAD.py", "package/payload.py"],
        ["package/file.py", "PACKAGE/other.py", "package/OTHER.py"],
        ["package/module.py", "package/module.py/child"],
        ["package/module.py/child", "package/module.py"],
    ],
)
def test_archive_member_paths_reject_portable_collisions(
    members: list[str],
) -> None:
    with pytest.raises(RuntimeError, match="collision"):
        smoke_wheel._assert_members_safe(Path("hostile.whl"), members, wheel=False)


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
@pytest.mark.parametrize(
    "target",
    ["../../outside.py", "/outside.py", "C:/outside.py", "dir\\outside.py"],
)
def test_sdist_rejects_unsafe_link_targets(
    tmp_path: Path, link_type: bytes, target: str
) -> None:
    wheel = tmp_path / "stableboundary.whl"
    _write_minimal_wheel(wheel)

    sdist = tmp_path / "stableboundary.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        root = smoke_wheel.EXPECTED_SDIST_ROOT
        _add_tar_bytes(archive, f"{root}/PKG-INFO", METADATA)
        _add_tar_bytes(archive, f"{root}/src/stableboundary/py.typed", b"")
        _add_tar_bytes(
            archive,
            f"{root}/src/stableboundary/core.py",
            b"VALUE = 1\n",
        )
        _add_tar_bytes(
            archive,
            f"{root}/examples/{smoke_wheel.EXAMPLE.name}",
            smoke_wheel.EXAMPLE.read_bytes(),
        )
        link = tarfile.TarInfo("stableboundary-0.1.0/link.txt")
        link.type = link_type
        link.linkname = target
        archive.addfile(link)

    with pytest.raises(RuntimeError, match="link target"):
        smoke_wheel._inspect_archives(wheel, sdist)


def test_sdist_rejects_even_canonical_links(tmp_path: Path) -> None:
    wheel = tmp_path / "stableboundary.whl"
    _write_minimal_wheel(wheel)
    sdist = tmp_path / "stableboundary.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        root = smoke_wheel.EXPECTED_SDIST_ROOT
        _add_tar_bytes(archive, f"{root}/PKG-INFO", METADATA)
        link = tarfile.TarInfo(f"{root}/link.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "safe.txt"
        archive.addfile(link)

    with pytest.raises(RuntimeError, match="links are forbidden"):
        smoke_wheel._inspect_archives(wheel, sdist)


def test_sdist_discovery_ignores_zip_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / smoke_wheel.EXPECTED_WHEEL).touch()
    expected_sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    expected_sdist.touch()
    (tmp_path / "unexpected.zip").touch()
    monkeypatch.setattr(smoke_wheel, "DIST", tmp_path)

    wheel, sdist = smoke_wheel._archives()

    assert wheel == (tmp_path / smoke_wheel.EXPECTED_WHEEL).resolve()
    assert sdist == expected_sdist.resolve()


@pytest.mark.parametrize(
    ("artifact_name", "expected_message"),
    [
        ("stable_boundary-0.1.0-py3-none-any.whl", "wheel filename"),
        ("stableboundary-0.1.tar.gz", "sdist filename"),
    ],
)
def test_archive_discovery_requires_canonical_name_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    expected_message: str,
) -> None:
    wheel_name = (
        artifact_name if artifact_name.endswith(".whl") else smoke_wheel.EXPECTED_WHEEL
    )
    sdist_name = (
        artifact_name
        if artifact_name.endswith(".tar.gz")
        else smoke_wheel.EXPECTED_SDIST
    )
    (tmp_path / wheel_name).touch()
    (tmp_path / sdist_name).touch()
    monkeypatch.setattr(smoke_wheel, "DIST", tmp_path)

    with pytest.raises(RuntimeError, match=expected_message):
        smoke_wheel._archives()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (b"Name: substitute\nVersion: 0.1.0\n", "unexpected Name"),
        (b"Name: stableboundary\nVersion: 9.9.9\n", "unexpected Version"),
        (
            b"Name: stableboundary\nName: substitute\nVersion: 0.1.0\n",
            "unexpected Name",
        ),
        (
            b"Name: stableboundary\nVersion: 0.1.0\nVersion: 9.9.9\n",
            "unexpected Version",
        ),
    ],
)
def test_archive_metadata_requires_project_identity(
    metadata: bytes, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        smoke_wheel._validate_metadata(
            Path(smoke_wheel.EXPECTED_WHEEL), metadata, subject="METADATA"
        )


def test_archive_scientific_payloads_must_match() -> None:
    with pytest.raises(RuntimeError, match="scientific payload bytes differ"):
        smoke_wheel._scientific_payloads_match(
            {"core.py": b"VALUE = 1\n"},
            {"core.py": b"VALUE = 2\n"},
        )


def test_archive_scientific_payload_file_sets_must_match() -> None:
    with pytest.raises(RuntimeError, match="package payloads differ"):
        smoke_wheel._scientific_payloads_match(
            {"core.py": b"VALUE = 1\n"},
            {"other.py": b"VALUE = 1\n"},
        )


def test_minimal_canonical_archives_pass_identity_and_payload_checks(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / smoke_wheel.EXPECTED_WHEEL
    sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    _write_minimal_wheel(wheel)
    _write_minimal_sdist(sdist)

    smoke_wheel._inspect_archives(wheel, sdist)


@pytest.mark.skipif(os.name != "nt", reason="Windows extraction semantics")
def test_windows_zip_extraction_demonstrates_colon_collision(
    tmp_path: Path,
) -> None:
    """Exercise the real stdlib sanitizer behind the portable-path rejection."""
    artifact = tmp_path / "hostile.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("payload/data:raw.txt", "hostile")
        archive.writestr("payload/data_raw.txt", "safe")

    with pytest.raises(RuntimeError, match="colon"):
        smoke_wheel._assert_members_safe(
            artifact,
            ["payload/data:raw.txt", "payload/data_raw.txt"],
            wheel=False,
        )

    extraction = tmp_path / "extracted"
    with zipfile.ZipFile(artifact) as archive:
        archive.extractall(extraction)
    extracted = extraction / "payload" / "data_raw.txt"
    assert extracted.is_file()
    assert extracted.read_text(encoding="utf-8") == "safe"


def test_installed_payload_requires_exact_finite_cell_method() -> None:
    payload = _valid_payload()
    payload["method"] = "limiting_poisson"

    with pytest.raises(RuntimeError, match="unexpected method"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_requires_complete_schema() -> None:
    payload = _valid_payload()
    del payload["truth"]

    with pytest.raises(RuntimeError, match="incomplete schema"):
        smoke_wheel._validate_example(payload)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("schema_version", 2, "schema version"),
        ("package_version", "0.2.0", "package version"),
        ("seed", 1, "simulation seed"),
        ("parameterization", "S1", "S0"),
    ],
)
def test_installed_payload_rejects_fixed_experiment_identity_changes(
    field: str, bad_value: object, message: str
) -> None:
    payload = _valid_payload()
    payload[field] = bad_value

    with pytest.raises(RuntimeError, match=message):
        smoke_wheel._validate_example(payload)


@pytest.mark.parametrize(
    ("container", "field", "bad_value", "message"),
    [
        ("design", "n", 4_999, "sample size"),
        ("design", "formula_version", 2, "design formula"),
        ("design", "threshold", 4.0, "inconsistent threshold"),
        ("truth", "alpha", 1.5, "derive truth alpha"),
        ("truth", "beta", 0.4, "simulation truth"),
        ("prior", "h_max", 5.0, "compact prior"),
        ("quadrature", "base_nodes", 16, "quadrature orders"),
        ("quadrature", "interval_mass", 0.95, "credible mass"),
        ("backend", "parameterization", "S1", "canonical SciPy S0"),
    ],
)
def test_installed_payload_rejects_scientific_contract_changes(
    container: str,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    payload = _valid_payload()
    values = payload[container]
    assert isinstance(values, dict)
    values[field] = bad_value

    with pytest.raises(RuntimeError, match=message):
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


def test_installed_payload_rejects_changed_fixed_seed_counts() -> None:
    payload = _valid_payload()
    counts = payload["counts"]
    assert isinstance(counts, dict)
    counts["n_minus"] = 2
    counts["n_zero"] = 4_995

    with pytest.raises(RuntimeError, match="fixed-seed cell counts"):
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


def test_installed_payload_rejects_prior_support_violation_that_is_globally_valid() -> (
    None
):
    payload = _valid_payload()
    parameters = payload["parameters"]
    assert isinstance(parameters, dict)
    summary = parameters["p"]
    assert isinstance(summary, dict)
    summary["mean"] = 0.99

    with pytest.raises(RuntimeError, match="outside its prior support"):
        smoke_wheel._validate_example(payload)


@pytest.mark.parametrize(
    ("component_group", "quantity", "component"),
    [
        ("scalar", "", "joint_total_variation"),
        ("scalar", "", "log_normalizer_change"),
        ("summary", "h", "mean"),
        ("summary", "p", "median"),
        ("summary", "alpha", "interval_lower"),
        ("summary", "beta", "interval_upper"),
        ("summary", "tau_plus", "mean"),
        ("summary", "tau_minus", "median"),
        ("predictive", "", "negative"),
        ("predictive", "", "positive"),
    ],
)
def test_installed_payload_rejects_every_refinement_component_above_tolerance(
    component_group: str,
    quantity: str,
    component: str,
) -> None:
    payload = _valid_payload()
    refinement = payload["refinement"]
    assert isinstance(refinement, dict)
    if component_group == "scalar":
        refinement[component] = 0.0020000000000001
    elif component_group == "summary":
        summary_changes = refinement["summary_changes"]
        assert isinstance(summary_changes, dict)
        changes = summary_changes[quantity]
        assert isinstance(changes, dict)
        changes[component] = 0.0020000000000001
    else:
        predictive = refinement["predictive_tail"]
        assert isinstance(predictive, dict)
        predictive[component] = 0.0020000000000001

    with pytest.raises(RuntimeError, match="above 0.002"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_requires_exact_common_grid() -> None:
    payload = _valid_payload()
    refinement = payload["refinement"]
    assert isinstance(refinement, dict)
    refinement["common_grid_points"] = 64

    with pytest.raises(RuntimeError, match="common grid"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_rejects_identification_contradiction() -> None:
    payload = _valid_payload()
    identification = payload["identification"]
    assert isinstance(identification, dict)
    identification["evidence_status"] = "prior_dominated"

    with pytest.raises(RuntimeError, match="identification labels"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_rejects_warning_contradiction() -> None:
    payload = _valid_payload()
    payload["warnings"] = ["research_uncertified: not a certificate."]

    with pytest.raises(RuntimeError, match="incomplete warnings"):
        smoke_wheel._validate_example(payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("parameterization", "S1"),
        ("pdf_default_method", "fft-simpson"),
        ("cdf_default_method", "dni"),
        ("quad_eps", 1e-8),
        ("piecewise_x_tol_near_zeta", 0.1),
        ("pdf_fft_interpolation_degree", 5),
    ],
)
def test_installed_payload_rejects_backend_setting_contradictions(
    field: str, bad_value: object
) -> None:
    payload = _valid_payload()
    backend = payload["backend"]
    assert isinstance(backend, dict)
    settings = backend["effective_settings"]
    assert isinstance(settings, dict)
    settings[field] = bad_value

    with pytest.raises(RuntimeError, match="contradict"):
        smoke_wheel._validate_example(payload)


def test_artifact_install_separates_dependencies_and_uses_no_deps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def record(command: list[str], **kwargs: object) -> str:
        del kwargs
        calls.append(command)
        return ""

    monkeypatch.setattr(smoke_wheel, "_run", record)
    artifact = tmp_path / smoke_wheel.EXPECTED_WHEEL
    artifact.touch()

    smoke_wheel._install_archive(Path("python"), artifact, cwd=tmp_path)

    assert len(calls) == 3
    assert "numpy>=2.2" in calls[0]
    assert "scipy>=1.18" in calls[0]
    assert not any("stableboundary" in argument for argument in calls[0])
    assert "find_spec('stableboundary') is None" in calls[1][-1]
    assert "--no-deps" in calls[2]
    assert calls[2][-1] == str(artifact)


def _valid_distribution_probe(artifact: Path, origin: Path) -> dict[str, object]:
    digest = sha256(artifact.read_bytes()).hexdigest()
    return {
        "import_origin": str(origin),
        "metadata_version": smoke_wheel.PROJECT_VERSION,
        "direct_url": {
            "url": artifact.resolve().as_uri(),
            "archive_info": {
                "hash": f"sha256={digest}",
                "hashes": {"sha256": digest},
            },
        },
    }


def test_installed_distribution_requires_exact_artifact_provenance(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "venv"
    origin = environment / "site-packages" / "stableboundary" / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.touch()
    artifact = tmp_path / smoke_wheel.EXPECTED_WHEEL
    artifact.write_bytes(b"artifact")

    selected = smoke_wheel._validate_installed_distribution(
        _valid_distribution_probe(artifact, origin),
        artifact=artifact,
        environment=environment,
    )

    assert selected == origin.resolve()


def test_installed_distribution_decodes_artifact_url_exactly_once(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "venv"
    origin = environment / "site-packages" / "stableboundary" / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.touch()
    artifact_directory = tmp_path / "literal%20directory"
    artifact_directory.mkdir()
    artifact = artifact_directory / smoke_wheel.EXPECTED_WHEEL
    artifact.write_bytes(b"artifact")

    selected = smoke_wheel._validate_installed_distribution(
        _valid_distribution_probe(artifact, origin),
        artifact=artifact,
        environment=environment,
    )

    assert selected == origin.resolve()


@pytest.mark.parametrize("mutation", ["version", "url", "hash"])
def test_installed_distribution_rejects_substituted_artifact(
    tmp_path: Path, mutation: str
) -> None:
    environment = tmp_path / "venv"
    origin = environment / "site-packages" / "stableboundary" / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.touch()
    artifact = tmp_path / smoke_wheel.EXPECTED_WHEEL
    artifact.write_bytes(b"artifact")
    probe = _valid_distribution_probe(artifact, origin)
    if mutation == "version":
        probe["metadata_version"] = "0.2.0"
    else:
        direct_url = probe["direct_url"]
        assert isinstance(direct_url, dict)
        if mutation == "url":
            substitute = tmp_path / "substitute.whl"
            substitute.touch()
            direct_url["url"] = substitute.as_uri()
        else:
            archive_info = direct_url["archive_info"]
            assert isinstance(archive_info, dict)
            archive_info["hashes"] = {"sha256": "0" * 64}

    with pytest.raises(RuntimeError):
        smoke_wheel._validate_installed_distribution(
            probe,
            artifact=artifact,
            environment=environment,
        )


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
