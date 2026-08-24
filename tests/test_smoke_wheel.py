"""Focused regression tests for the distribution-artifact smoke runner."""

from __future__ import annotations

import copy
import csv
import io
import math
import os
import stat
import subprocess
import sys
import tarfile
import unicodedata
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest

from scripts import smoke_wheel


def _metadata_bytes() -> bytes:
    expected, readme, license_content = smoke_wheel._metadata_expectations()
    lines: list[bytes] = []
    ordered = (
        "Metadata-Version",
        "Name",
        "Version",
        "Summary",
        "Author",
    )
    for name in ordered:
        lines.append(f"{name}: {expected[name][0]}".encode())
    license_lines = license_content.rstrip(b"\n").split(b"\n")
    lines.append(b"License: " + license_lines[0])
    lines.extend(b"        " + line for line in license_lines[1:])
    for name in (
        "License-File",
        "Requires-Python",
        "Requires-Dist",
        "Provides-Extra",
        "Description-Content-Type",
    ):
        lines.extend(f"{name}: {value}".encode() for value in expected[name])
    return b"\n".join(lines) + b"\n\n" + readme


METADATA = _metadata_bytes()
WHEEL = (
    b"Wheel-Version: 1.0\n"
    b"Generator: hatchling 1.32.0\n"
    b"Root-Is-Purelib: true\n"
    b"Tag: py3-none-any\n\n"
)


def _wheel_members() -> dict[str, bytes]:
    package = smoke_wheel._repository_package_payload()
    return {
        **{f"stableboundary/{name}": content for name, content in package.items()},
        f"{smoke_wheel.DIST_INFO}/METADATA": METADATA,
        f"{smoke_wheel.DIST_INFO}/WHEEL": WHEEL,
        f"{smoke_wheel.DIST_INFO}/licenses/LICENSE": (
            smoke_wheel.REPOSITORY / "LICENSE"
        ).read_bytes(),
    }


def _record_bytes(members: dict[str, bytes]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, content in members.items():
        writer.writerow(
            [name, f"sha256={smoke_wheel._record_digest(content)}", len(content)]
        )
    writer.writerow([f"{smoke_wheel.DIST_INFO}/RECORD", "", ""])
    return output.getvalue().encode()


def _write_minimal_wheel(
    path: Path,
    *,
    additions: dict[str, bytes] | None = None,
    overrides: dict[str, bytes] | None = None,
    record_override: bytes | None = None,
) -> None:
    members = _wheel_members()
    members.update(overrides or {})
    members.update(additions or {})
    record = _record_bytes(members) if record_override is None else record_override
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        archive.writestr(f"{smoke_wheel.DIST_INFO}/RECORD", record)


def _add_tar_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _write_minimal_sdist(
    path: Path,
    *,
    example: bytes | None = None,
    additions: dict[str, bytes] | None = None,
    overrides: dict[str, bytes] | None = None,
) -> None:
    root = smoke_wheel.EXPECTED_SDIST_ROOT
    package = smoke_wheel._repository_package_payload()
    members = {
        f"{root}/uv.lock": (smoke_wheel.REPOSITORY / "uv.lock").read_bytes(),
        f"{root}/examples/{smoke_wheel.EXAMPLE.name}": (
            smoke_wheel.EXAMPLE.read_bytes() if example is None else example
        ),
        **{
            f"{root}/src/stableboundary/{name}": content
            for name, content in package.items()
        },
        f"{root}/.gitignore": (smoke_wheel.REPOSITORY / ".gitignore").read_bytes(),
        f"{root}/LICENSE": (smoke_wheel.REPOSITORY / "LICENSE").read_bytes(),
        f"{root}/README.md": (smoke_wheel.REPOSITORY / "README.md").read_bytes(),
        f"{root}/pyproject.toml": (
            smoke_wheel.REPOSITORY / "pyproject.toml"
        ).read_bytes(),
        f"{root}/PKG-INFO": METADATA,
    }
    members.update(overrides or {})
    members.update(additions or {})
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            _add_tar_bytes(archive, name, content)


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


def test_dist_discovery_rejects_every_extra_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / smoke_wheel.EXPECTED_WHEEL).touch()
    expected_sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    expected_sdist.touch()
    (tmp_path / "unexpected.zip").touch()
    monkeypatch.setattr(smoke_wheel, "DIST", tmp_path)

    with pytest.raises(RuntimeError, match="exactly the expected"):
        smoke_wheel._archives()


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

    with pytest.raises(RuntimeError, match="exactly the expected"):
        smoke_wheel._archives()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            METADATA.replace(b"Name: stableboundary\n", b"Name: substitute\n"),
            "unexpected Name",
        ),
        (
            METADATA.replace(b"Version: 0.1.0\n", b"Version: 9.9.9\n"),
            "unexpected Version",
        ),
        (
            METADATA.replace(
                b"Name: stableboundary\n",
                b"Name: stableboundary\nName: substitute\n",
            ),
            "unexpected Name",
        ),
        (
            METADATA.replace(
                b"Version: 0.1.0\n",
                b"Version: 0.1.0\nVersion: 9.9.9\n",
            ),
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


@pytest.mark.parametrize(
    "member",
    [
        "sitecustomize.py",
        "payload.pth",
        "stableboundary-0.1.0.data/scripts/runner.py",
        "substitute-0.1.0.dist-info/METADATA",
        "top_level_module.py",
        "secret.env",
    ],
)
def test_wheel_rejects_every_unknown_member(tmp_path: Path, member: str) -> None:
    wheel = tmp_path / smoke_wheel.EXPECTED_WHEEL
    sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    _write_minimal_wheel(wheel, additions={member: b"hostile\n"})
    _write_minimal_sdist(sdist)

    with pytest.raises(RuntimeError, match="exact source-bound manifest"):
        smoke_wheel._inspect_archives(wheel, sdist)


@pytest.mark.parametrize(
    "member",
    [
        "secret.env",
        "dist/stale.whl",
        "sitecustomize.py",
        "tests/test_leak.py",
    ],
)
def test_sdist_rejects_every_unknown_member(tmp_path: Path, member: str) -> None:
    wheel = tmp_path / smoke_wheel.EXPECTED_WHEEL
    sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    _write_minimal_wheel(wheel)
    root = smoke_wheel.EXPECTED_SDIST_ROOT
    _write_minimal_sdist(sdist, additions={f"{root}/{member}": b"hostile\n"})

    with pytest.raises(RuntimeError):
        smoke_wheel._inspect_archives(wheel, sdist)


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_archives_reject_stale_repository_package_payload(
    tmp_path: Path, archive_kind: str
) -> None:
    wheel = tmp_path / smoke_wheel.EXPECTED_WHEEL
    sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    relative = "stableboundary/__init__.py"
    source = (smoke_wheel.REPOSITORY / "src" / relative).read_bytes()
    stale = bytes([source[0] ^ 1]) + source[1:]
    wheel_override = {relative: stale} if archive_kind == "wheel" else None
    root = smoke_wheel.EXPECTED_SDIST_ROOT
    sdist_override = (
        {f"{root}/src/{relative}": stale} if archive_kind == "sdist" else None
    )
    _write_minimal_wheel(wheel, overrides=wheel_override)
    _write_minimal_sdist(sdist, overrides=sdist_override)

    with pytest.raises(RuntimeError, match="differs from repository source"):
        smoke_wheel._inspect_archives(wheel, sdist)


def test_sdist_rejects_substituted_pyproject(tmp_path: Path) -> None:
    wheel = tmp_path / smoke_wheel.EXPECTED_WHEEL
    sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    _write_minimal_wheel(wheel)
    root = smoke_wheel.EXPECTED_SDIST_ROOT
    original = (smoke_wheel.REPOSITORY / "pyproject.toml").read_bytes()
    substituted = original.replace(
        b'name = "stableboundary"', b'name = "substitution!!"'
    )
    _write_minimal_sdist(
        sdist,
        overrides={f"{root}/pyproject.toml": substituted},
    )

    with pytest.raises(RuntimeError, match="pyproject.toml"):
        smoke_wheel._inspect_archives(wheel, sdist)


def _hostile_record(kind: str) -> bytes:
    valid = _record_bytes(_wheel_members()).decode()
    lines = valid.splitlines()
    if kind == "missing":
        lines.pop(0)
    elif kind == "duplicate":
        lines.insert(1, lines[0])
    elif kind == "digest":
        fields = lines[0].split(",")
        fields[1] = "sha256=" + "A" * 43
        lines[0] = ",".join(fields)
    elif kind == "size":
        fields = lines[0].split(",")
        fields[2] = str(int(fields[2]) + 1)
        lines[0] = ",".join(fields)
    elif kind == "self":
        lines[-1] = f"{smoke_wheel.DIST_INFO}/RECORD,sha256={'A' * 43},1"
    elif kind == "extra":
        lines.insert(-1, f"secret.env,sha256={'A' * 43},1")
    else:  # pragma: no cover - test helper guard
        raise AssertionError(kind)
    return ("\n".join(lines) + "\n").encode()


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "digest", "size", "self", "extra"]
)
def test_wheel_record_is_an_exact_content_manifest(tmp_path: Path, kind: str) -> None:
    wheel = tmp_path / smoke_wheel.EXPECTED_WHEEL
    sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    _write_minimal_wheel(wheel, record_override=_hostile_record(kind))
    _write_minimal_sdist(sdist)

    with pytest.raises(RuntimeError, match="RECORD"):
        smoke_wheel._inspect_archives(wheel, sdist)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (b"Wheel-Version: 1.0", b"Wheel-Version: 1.1"),
        (b"Generator: hatchling 1.32.0", b"Generator: hostile 9.9"),
        (b"Root-Is-Purelib: true", b"Root-Is-Purelib: false"),
        (b"Tag: py3-none-any", b"Tag: cp314-cp314-win_amd64"),
    ],
)
def test_wheel_file_requires_exact_pure_python_build_contract(
    old: bytes, new: bytes
) -> None:
    with pytest.raises(RuntimeError, match="WHEEL"):
        smoke_wheel._validate_wheel_file(
            Path(smoke_wheel.EXPECTED_WHEEL), WHEEL.replace(old, new)
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            b"Summary: Auditable Bayesian inference",
            b"Summary: Poisoned inference",
            "Summary",
        ),
        (b"Author: Moeketsi Mosia", b"Author: Substitute", "Author"),
        (b"License-File: LICENSE", b"License-File: SECRET", "License-File"),
        (b"Requires-Python: >=3.12", b"Requires-Python: >=3.9", "Requires-Python"),
        (b"Requires-Dist: numpy>=2.2", b"Requires-Dist: numpy", "Requires-Dist"),
        (b"Provides-Extra: dev", b"Provides-Extra: hostile", "Provides-Extra"),
        (
            b"Description-Content-Type: text/markdown",
            b"Description-Content-Type: text/plain",
            "Description-Content-Type",
        ),
    ],
)
def test_metadata_binds_complete_pyproject_contract(
    old: bytes, new: bytes, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        smoke_wheel._validate_metadata(
            Path(smoke_wheel.EXPECTED_WHEEL),
            METADATA.replace(old, new),
            subject="METADATA",
        )


def test_metadata_rejects_poison_header_and_readme_payload() -> None:
    artifact = Path(smoke_wheel.EXPECTED_WHEEL)
    with pytest.raises(RuntimeError, match="headers"):
        smoke_wheel._validate_metadata(
            artifact,
            METADATA.replace(
                b"Name: stableboundary\n", b"X-Poison: yes\nName: stableboundary\n"
            ),
            subject="METADATA",
        )
    with pytest.raises(RuntimeError, match="README"):
        smoke_wheel._validate_metadata(
            artifact,
            METADATA + b"poison",
            subject="METADATA",
        )


def test_wheel_metadata_and_sdist_pkg_info_require_byte_parity(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / smoke_wheel.EXPECTED_WHEEL
    sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    lf_metadata = METADATA.replace(b"\r\n", b"\n")
    crlf_metadata = lf_metadata.replace(b"\n", b"\r\n")
    _write_minimal_wheel(
        wheel,
        overrides={f"{smoke_wheel.DIST_INFO}/METADATA": lf_metadata},
    )
    root = smoke_wheel.EXPECTED_SDIST_ROOT
    _write_minimal_sdist(
        sdist,
        overrides={f"{root}/PKG-INFO": crlf_metadata},
    )

    with pytest.raises(RuntimeError, match="byte-for-byte"):
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

    assert len(calls) == 4
    assert "numpy>=2.2" in calls[0]
    assert "scipy>=1.18,<1.19" in calls[0]
    assert not any("stableboundary" in argument for argument in calls[0])
    assert "find_spec('stableboundary') is None" in calls[1][-1]
    assert "--no-deps" in calls[2]
    assert calls[2][-1] == str(artifact)
    assert calls[3][-3:] == ["pip", "check", "--disable-pip-version-check"]


def test_artifact_install_fails_closed_when_pip_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_check(command: list[str], **kwargs: object) -> str:
        del kwargs
        if "check" in command:
            raise RuntimeError("dependency graph is inconsistent")
        return ""

    monkeypatch.setattr(smoke_wheel, "_run", fail_check)
    artifact = tmp_path / smoke_wheel.EXPECTED_WHEEL
    artifact.touch()

    with pytest.raises(RuntimeError, match="dependency graph"):
        smoke_wheel._install_archive(Path("python"), artifact, cwd=tmp_path)


def _valid_distribution_probe(artifact: Path, origin: Path) -> dict[str, object]:
    digest = sha256(artifact.read_bytes()).hexdigest()
    source_digest = sha256(origin.read_bytes()).hexdigest()
    return {
        "import_origin": str(origin),
        "metadata_version": smoke_wheel.PROJECT_VERSION,
        "package_version": smoke_wheel.PROJECT_VERSION,
        "versions": {
            "python": "3.14.7",
            "numpy": "2.5.2",
            "scipy": "1.18.1",
            "stableboundary": smoke_wheel.PROJECT_VERSION,
        },
        "package_files": {
            "__init__.py": {"size": origin.stat().st_size, "sha256": source_digest}
        },
        "direct_url": {
            "url": artifact.resolve().as_uri(),
            "archive_info": {
                "hash": f"sha256={digest}",
                "hashes": {"sha256": digest},
            },
        },
    }


def _test_package_manifest(origin: Path) -> dict[str, smoke_wheel.FileIdentity]:
    content = origin.read_bytes()
    return {"__init__.py": smoke_wheel._file_identity(content)}


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
        expected_digest=sha256(artifact.read_bytes()).hexdigest(),
        package_files=_test_package_manifest(origin),
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
        expected_digest=sha256(artifact.read_bytes()).hexdigest(),
        package_files=_test_package_manifest(origin),
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
            expected_digest=sha256(artifact.read_bytes()).hexdigest(),
            package_files=_test_package_manifest(origin),
        )


@pytest.mark.parametrize(
    ("container", "field", "bad_value"),
    [
        ("versions", "stableboundary", "0.0.0"),
        ("versions", "python", "3.11.9"),
        ("versions", "scipy", "1.19.0"),
        ("package_files", "__init__.py", {"size": 0, "sha256": "0" * 64}),
    ],
)
def test_installed_distribution_rejects_stale_runtime_or_source(
    tmp_path: Path,
    container: str,
    field: str,
    bad_value: object,
) -> None:
    environment = tmp_path / "venv"
    origin = environment / "site-packages" / "stableboundary" / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_text("trusted = True\n", encoding="utf-8")
    artifact = tmp_path / smoke_wheel.EXPECTED_WHEEL
    artifact.write_bytes(b"artifact")
    probe = _valid_distribution_probe(artifact, origin)
    selected = probe[container]
    assert isinstance(selected, dict)
    selected[field] = bad_value

    with pytest.raises(RuntimeError):
        smoke_wheel._validate_installed_distribution(
            probe,
            artifact=artifact,
            environment=environment,
            expected_digest=sha256(artifact.read_bytes()).hexdigest(),
            package_files=_test_package_manifest(origin),
        )


def test_artifact_snapshot_isolated_from_original_path_replacement(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / smoke_wheel.EXPECTED_WHEEL
    sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    wheel.write_bytes(b"trusted wheel")
    sdist.write_bytes(b"trusted sdist")

    with smoke_wheel._artifact_snapshots(wheel, sdist) as snapshots:
        wheel_snapshot, sdist_snapshot = snapshots
        wheel.write_bytes(b"replacement wheel")
        sdist.write_bytes(b"replacement sdist")

        smoke_wheel._assert_snapshot(wheel_snapshot)
        smoke_wheel._assert_snapshot(sdist_snapshot)
        assert wheel_snapshot.path.read_bytes() == b"trusted wheel"
        assert sdist_snapshot.path.read_bytes() == b"trusted sdist"
        assert wheel_snapshot.path.resolve() != wheel.resolve()


def test_artifact_snapshot_mutation_fails_before_execution(tmp_path: Path) -> None:
    wheel = tmp_path / smoke_wheel.EXPECTED_WHEEL
    sdist = tmp_path / smoke_wheel.EXPECTED_SDIST
    wheel.write_bytes(b"trusted wheel")
    sdist.write_bytes(b"trusted sdist")

    with smoke_wheel._artifact_snapshots(wheel, sdist) as snapshots:
        wheel_snapshot, _ = snapshots
        wheel_snapshot.path.chmod(stat.S_IREAD | stat.S_IWRITE)
        wheel_snapshot.path.write_bytes(b"hostile wheel")

        with pytest.raises(RuntimeError, match="snapshot"):
            smoke_wheel._assert_snapshot(wheel_snapshot)


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
