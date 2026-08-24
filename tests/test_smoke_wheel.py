"""Focused regression tests for the distribution-artifact smoke runner."""

from __future__ import annotations

import ast
import copy
import csv
import io
import json
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


def _valid_payload() -> dict[str, object]:
    oracle = smoke_wheel._oracle_document()
    design = copy.deepcopy(oracle["design"])
    r_value = design["r"]
    assert isinstance(r_value, float)
    contract = oracle["simulation_contract"]
    assert isinstance(contract, dict)
    approved = contract["approved_environments"]
    assert isinstance(approved, dict)
    environment = approved["system=Windows|machine=AMD64|numpy=2.5.2|scipy=1.18.1"]
    assert isinstance(environment, dict)
    reference_parameters = oracle["parameters"]
    assert isinstance(reference_parameters, dict)
    parameters: dict[str, object] = {}
    for quantity, reference in reference_parameters.items():
        assert isinstance(reference, dict)
        parameters[quantity] = {
            "mean": reference["mean"],
            "median": reference["median"],
            "credible_interval": {
                "lower": reference["lower"],
                "upper": reference["upper"],
                "mass": 0.9,
            },
        }
    return {
        "schema_version": 1,
        "package_version": "0.1.0",
        "status": "research_uncertified",
        "method": "exact_finite_three_cell",
        "parameterization": "S0",
        "known_nuisance": copy.deepcopy(oracle["known_nuisance"]),
        "seed": 20_260_824,
        "truth": {
            "alpha": 2.0 - 1.5 * r_value,
            "beta": 0.35,
            "loc": 0.0,
            "scale": 1.0,
        },
        "inference_fixture": copy.deepcopy(oracle["fixture"]),
        "simulation": {
            "dtype": environment["dtype"],
            "rng_algorithm": contract["rng_algorithm"],
            "simulator_algorithm": contract["simulator_algorithm"],
            "platform_system": "Windows",
            "platform_machine": "AMD64",
            "python_version": "3.14.7",
            "numpy_version": "2.5.2",
            "scipy_version": "1.18.1",
            "sample_sha256": environment["observed_raw_sha256"][0],
            "quantized_sample_sha256": copy.deepcopy(
                environment["quantized_sample_sha256"]
            ),
            "counts": copy.deepcopy(environment["counts"]),
            "minimum": environment["minimum"],
            "maximum": environment["maximum"],
            "diagnostics": copy.deepcopy(environment["observed_diagnostics"]),
        },
        "design": design,
        "prior": copy.deepcopy(oracle["prior"]),
        "counts": copy.deepcopy(oracle["counts"]),
        "quadrature": copy.deepcopy(oracle["quadrature"]),
        "parameters": parameters,
        "posterior_mass": oracle["posterior_mass"],
        "refinement": copy.deepcopy(oracle["refinement"]),
        "identification": copy.deepcopy(oracle["identification"]),
        "backend": {
            "method": "scipy-piecewise-s0-direct-log-tails",
            "tolerance": 1.2e-14,
            "origin": "canonical_scipy_s0",
            "parameterization": "S0",
            "library": "scipy",
            "library_version": "1.18.1",
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
        "warnings": copy.deepcopy(oracle["warnings"]),
    }


def _valid_runtime_versions() -> dict[str, str]:
    return {
        "python": "3.14.7",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "numpy": "2.5.2",
        "scipy": "1.18.1",
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


def test_oracle_backed_payload_passes_full_scientific_validation() -> None:
    smoke_wheel._validate_example(
        _valid_payload(),
        runtime_versions=_valid_runtime_versions(),
    )


def test_source_tree_simulation_fingerprint_is_explicitly_approved() -> None:
    """Collect the live fingerprint in every ordinary dependency/OS CI job."""
    payload = _valid_payload()
    python = Path(sys.executable)
    public_probe = smoke_wheel._fresh_simulation_probe(
        python,
        cwd=smoke_wheel.REPOSITORY,
        stage="fresh public source-tree simulation probe",
    )
    independent_probe = smoke_wheel._fresh_simulation_probe(
        python,
        cwd=smoke_wheel.REPOSITORY,
        stage="fresh independent source-tree simulation probe",
    )
    smoke_wheel._assert_simulation_probe_parity(public_probe, independent_probe)
    payload["simulation"] = public_probe
    backend = payload["backend"]
    assert isinstance(backend, dict)
    backend["library_version"] = public_probe["scipy_version"]
    runtime_versions = {
        "python": public_probe["python_version"],
        "platform_system": public_probe["platform_system"],
        "platform_machine": public_probe["platform_machine"],
        "numpy": public_probe["numpy_version"],
        "scipy": public_probe["scipy_version"],
    }

    smoke_wheel._validate_example(payload, runtime_versions=runtime_versions)


def test_source_tree_posterior_regression_is_explicitly_approved() -> None:
    """Exercise primary and reflected inference in every ordinary CI environment."""
    science_probe = smoke_wheel._installed_science_probe(
        Path(sys.executable),
        cwd=smoke_wheel.REPOSITORY,
        artifact=Path("source-tree"),
    )
    assert set(science_probe) == {"primary", "reflection"}
    primary = science_probe["primary"]
    reflection = science_probe["reflection"]
    assert isinstance(primary, dict)
    simulation = primary["simulation"]
    assert isinstance(simulation, dict)
    runtime_versions = {
        "python": simulation["python_version"],
        "platform_system": simulation["platform_system"],
        "platform_machine": simulation["platform_machine"],
        "numpy": simulation["numpy_version"],
        "scipy": simulation["scipy_version"],
    }

    smoke_wheel._validate_science_probe(
        primary,
        reflection,
        runtime_versions=runtime_versions,
    )


@pytest.mark.parametrize(
    ("container", "quantity", "field", "fake_value"),
    [
        ("quadrature", "", "log_normalizer", -17.0),
        ("parameters", "h", "mean", 2.0),
        ("parameters", "p", "mean", 0.5),
    ],
)
def test_prior_invented_valid_payload_values_are_rejected(
    container: str, quantity: str, field: str, fake_value: float
) -> None:
    """The former shape-only fixture must never authenticate numerical output."""
    payload = _valid_payload()
    selected = payload[container]
    assert isinstance(selected, dict)
    if quantity:
        selected = selected[quantity]
        assert isinstance(selected, dict)
    selected[field] = fake_value

    with pytest.raises(RuntimeError, match="trusted numerical reference"):
        smoke_wheel._validate_example(payload)


def test_numeric_reference_mismatch_reports_actual_expected_and_tolerance() -> None:
    with pytest.raises(RuntimeError, match="trusted numerical reference") as captured:
        smoke_wheel._reference_close(
            "diagnostic quantity",
            1.25,
            1.0,
            tolerance=0.01,
        )

    message = str(captured.value)
    assert "actual=1.25" in message
    assert "expected=1.0" in message
    assert "absolute_tolerance=0.01" in message
    assert "absolute_error=0.25" in message


def test_numerical_regression_fingerprint_contains_all_observed_evidence() -> None:
    primary = _valid_payload()
    fingerprint = smoke_wheel._numerical_regression_fingerprint(
        primary,
        {
            "status": primary["status"],
            "method": primary["method"],
            "parameterization": primary["parameterization"],
            "counts": primary["counts"],
            "log_normalizer": primary["quadrature"]["log_normalizer"],
            "parameters": primary["parameters"],
            "identification": primary["identification"],
            "warnings": primary["warnings"],
        },
        runtime_versions=_valid_runtime_versions(),
    )

    observed = fingerprint["primary"]
    assert isinstance(observed, dict)
    refinement = observed["refinement"]
    assert isinstance(refinement, dict)
    components = refinement["components"]
    assert isinstance(components, dict)
    assert refinement["component_count"] == 28
    assert len(components) == 28
    assert "summary.h.mean" in components
    assert "summary.tau_minus.interval_upper" in components
    assert "predictive.negative" in components
    assert observed["quadrature"] == primary["quadrature"]
    assert observed["parameters"] == primary["parameters"]
    assert observed["posterior_mass"] == primary["posterior_mass"]
    assert observed["identification"] == primary["identification"]
    assert observed["warnings"] == primary["warnings"]
    assert observed["known_nuisance"] == primary["known_nuisance"]
    assert observed["backend"] == primary["backend"]


def test_science_probe_failure_emits_complete_regression_fingerprint() -> None:
    primary = _valid_payload()
    refinement = primary["refinement"]
    assert isinstance(refinement, dict)
    refinement["log_normalizer_change"] = 0.001

    with pytest.raises(
        RuntimeError, match="observed numerical regression fingerprint"
    ) as captured:
        smoke_wheel._validate_science_probe(
            primary,
            {"status": primary["status"]},
            runtime_versions=_valid_runtime_versions(),
        )

    message = str(captured.value)
    assert "actual=0.001" in message
    assert "expected=5.329070518200751e-15" in message
    assert "allowed_interval=(0.0, 5e-14]" in message
    assert '"component_count": 28' in message
    assert '"summary.alpha.interval_lower"' in message
    assert '"summary.tau_plus.median"' in message
    assert '"predictive.positive"' in message
    assert '"posterior_mass": 0.9999999999999999' in message
    assert '"known_nuisance": {' in message
    assert '"warnings": [' in message
    assert '"reflection": {' in message


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


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("numpy_version", "2.4.0"),
        ("scipy_version", "1.18.2"),
        ("platform_system", "Darwin"),
        ("platform_machine", "arm64"),
    ],
)
def test_installed_payload_rejects_unknown_platform_dependency_combinations(
    field: str, bad_value: str
) -> None:
    payload = _valid_payload()
    simulation = payload["simulation"]
    assert isinstance(simulation, dict)
    simulation[field] = bad_value

    with pytest.raises(RuntimeError, match="environment is not approved"):
        smoke_wheel._validate_example(payload)


def test_simulation_mismatch_reports_complete_observed_fingerprint() -> None:
    payload = _valid_payload()
    simulation = payload["simulation"]
    assert isinstance(simulation, dict)
    quantized = simulation["quantized_sample_sha256"]
    assert isinstance(quantized, dict)
    quantized["1e-12"] = "f" * 64

    with pytest.raises(RuntimeError, match="approved reference") as captured:
        smoke_wheel._validate_example(payload)

    message = str(captured.value)
    for required in (
        '"sample_sha256": "a39752ae',
        '"1e-12": "ffffffff',
        '"counts": {',
        '"diagnostics": {',
        '"minimum": -12.046204723023758',
        '"maximum": 9.384903377918656',
        '"system": "Windows"',
        '"machine": "AMD64"',
        '"python": "3.14.7"',
        '"numpy": "2.5.2"',
        '"scipy": "1.18.1"',
        '"rng_algorithm": "numpy.random.PCG64"',
        '"simulator_algorithm": "scipy.stats.levy_stable.rvs:S0:private-generator:v1"',
        '"seed": 20260824',
        '"truth": {',
    ):
        assert required in message


def test_raw_sample_hash_is_diagnostic_not_a_portability_contract() -> None:
    payload = _valid_payload()
    simulation = payload["simulation"]
    assert isinstance(simulation, dict)
    simulation["sample_sha256"] = "0" * 64

    smoke_wheel._validate_example(payload)


def test_finer_sample_hash_is_diagnostic_not_a_portability_contract() -> None:
    payload = _valid_payload()
    simulation = payload["simulation"]
    assert isinstance(simulation, dict)
    quantized = simulation["quantized_sample_sha256"]
    assert isinstance(quantized, dict)
    quantized["1e-14"] = "0" * 64

    smoke_wheel._validate_example(payload)


def test_changed_simulation_summary_diagnostics_are_rejected() -> None:
    payload = _valid_payload()
    simulation = payload["simulation"]
    assert isinstance(simulation, dict)
    diagnostics = simulation["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["mean"] = float(diagnostics["mean"]) + 1e-6

    with pytest.raises(RuntimeError, match=r"diagnostics\[mean\]"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_rejects_three_point_fake_simulator() -> None:
    payload = _valid_payload()
    simulation = payload["simulation"]
    assert isinstance(simulation, dict)
    simulation["sample_sha256"] = sha256(
        b"three hand-selected floating-point observations"
    ).hexdigest()
    simulation["counts"] = {"n_minus": 1, "n_zero": 1, "n_plus": 1}

    with pytest.raises(RuntimeError, match="wrong sample size"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_rejects_non_stable_simulator_algorithm() -> None:
    payload = _valid_payload()
    simulation = payload["simulation"]
    assert isinstance(simulation, dict)
    simulation["simulator_algorithm"] = "hand-coded-three-point-generator"

    with pytest.raises(RuntimeError, match="algorithm contract"):
        smoke_wheel._validate_example(payload)


def test_installed_payload_rejects_stale_independently_imported_versions() -> None:
    stale_versions = _valid_runtime_versions()
    stale_versions["scipy"] = "1.18.0"
    with pytest.raises(RuntimeError, match="contradicts independent imports"):
        smoke_wheel._validate_example(
            _valid_payload(),
            runtime_versions=stale_versions,
        )


def test_installed_payload_rejects_stale_backend_library_version() -> None:
    payload = _valid_payload()
    backend = payload["backend"]
    assert isinstance(backend, dict)
    backend["library_version"] = "1.18.0"

    with pytest.raises(RuntimeError, match="contradicts simulation"):
        smoke_wheel._validate_example(payload)


def test_fixture_identity_is_reconstructed_outside_the_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    fixture = payload["inference_fixture"]
    assert isinstance(fixture, dict)
    fixture["sha256"] = "0" * 64
    poisoned_oracle = copy.deepcopy(smoke_wheel._oracle_document())
    oracle_fixture = poisoned_oracle["fixture"]
    assert isinstance(oracle_fixture, dict)
    oracle_fixture["sha256"] = "0" * 64
    monkeypatch.setattr(smoke_wheel, "_oracle_document", lambda: poisoned_oracle)

    with pytest.raises(RuntimeError, match="fixed inference fixture"):
        smoke_wheel._validate_example(payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("provenance", "fixed cell-count witness"), ("mode", "same_sample_plugin")],
)
def test_nuisance_provenance_requires_exact_values(field: str, bad_value: str) -> None:
    payload = _valid_payload()
    nuisance = payload["known_nuisance"]
    assert isinstance(nuisance, dict)
    nuisance[field] = bad_value

    with pytest.raises(RuntimeError, match="exact nuisance provenance"):
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
    interval["lower"] = 0.8

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


NUMERICAL_ORACLE_PATHS = [
    ("quadrature", "log_normalizer"),
    ("posterior_mass",),
    ("identification", "p_kl_divergence"),
    ("identification", "p_interval_width_contraction"),
    *[
        ("parameters", quantity, component)
        for quantity in smoke_wheel.QUANTITIES
        for component in ("mean", "median")
    ],
    *[
        ("parameters", quantity, "credible_interval", endpoint)
        for quantity in smoke_wheel.QUANTITIES
        for endpoint in ("lower", "upper")
    ],
]


@pytest.mark.parametrize("path", NUMERICAL_ORACLE_PATHS)
def test_every_posterior_quantity_is_bound_to_the_numerical_oracle(
    path: tuple[str, ...],
) -> None:
    payload = _valid_payload()
    selected: dict[str, object] = payload
    for name in path[:-1]:
        nested = selected[name]
        assert isinstance(nested, dict)
        selected = nested
    value = selected[path[-1]]
    assert isinstance(value, float)
    selected[path[-1]] = value + (1e-6 if "tau_" in ".".join(path) else 1e-4)

    with pytest.raises(
        RuntimeError,
        match="trusted numerical reference|inconsistent with its interval",
    ):
        smoke_wheel._validate_example(payload)


def test_reported_contraction_is_derived_from_the_reported_p_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    identification = payload["identification"]
    assert isinstance(identification, dict)
    identification["p_interval_width_contraction"] = 0.3
    poisoned_oracle = copy.deepcopy(smoke_wheel._oracle_document())
    oracle_identification = poisoned_oracle["identification"]
    assert isinstance(oracle_identification, dict)
    oracle_identification["p_interval_width_contraction"] = 0.3
    monkeypatch.setattr(smoke_wheel, "_oracle_document", lambda: poisoned_oracle)

    with pytest.raises(RuntimeError, match="inconsistent with its interval"):
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


NONZERO_REFINEMENT_PATHS = [
    ("joint_total_variation",),
    ("log_normalizer_change",),
    *[
        ("summary_changes", quantity, component)
        for quantity in smoke_wheel.QUANTITIES
        for component in ("mean", "median", "interval_lower", "interval_upper")
    ],
    ("predictive_tail", "negative"),
    ("predictive_tail", "positive"),
]

NOISE_SCALE_REFINEMENT_PATHS = [
    ("log_normalizer_change",),
    *[("summary_changes", quantity, "mean") for quantity in smoke_wheel.QUANTITIES],
    ("predictive_tail", "negative"),
    ("predictive_tail", "positive"),
]

SUBSTANTIVE_REFINEMENT_PATHS = [
    ("joint_total_variation",),
    *[
        ("summary_changes", quantity, component)
        for quantity in smoke_wheel.QUANTITIES
        for component in ("median", "interval_lower", "interval_upper")
    ],
]


def _replace_refinement_component(
    payload: dict[str, object], path: tuple[str, ...], value: float
) -> None:
    refinement = payload["refinement"]
    assert isinstance(refinement, dict)
    selected = refinement
    for name in path[:-1]:
        nested = selected[name]
        assert isinstance(nested, dict)
        selected = nested
    selected[path[-1]] = value


@pytest.mark.parametrize("path", NONZERO_REFINEMENT_PATHS)
def test_every_nonzero_refinement_component_is_reference_bound(
    path: tuple[str, ...],
) -> None:
    payload = _valid_payload()
    refinement = payload["refinement"]
    assert isinstance(refinement, dict)
    selected = refinement
    for name in path[:-1]:
        nested = selected[name]
        assert isinstance(nested, dict)
        selected = nested
    selected[path[-1]] = 0.0

    with pytest.raises(RuntimeError, match="lost nonzero refinement evidence"):
        smoke_wheel._validate_example(payload)


@pytest.mark.parametrize("path", NOISE_SCALE_REFINEMENT_PATHS)
@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (0.0, "lost nonzero refinement evidence"),
        (5.000000000000001e-14, "5e-14 bound"),
    ],
)
def test_noise_scale_refinement_band_rejects_zero_and_overshoot(
    path: tuple[str, ...], bad_value: float, message: str
) -> None:
    payload = _valid_payload()
    _replace_refinement_component(payload, path, bad_value)

    with pytest.raises(RuntimeError, match=message):
        smoke_wheel._validate_example(payload)


@pytest.mark.parametrize("path", SUBSTANTIVE_REFINEMENT_PATHS)
def test_substantive_refinement_components_remain_reference_bound(
    path: tuple[str, ...],
) -> None:
    payload = _valid_payload()
    refinement = payload["refinement"]
    assert isinstance(refinement, dict)
    selected = refinement
    for name in path[:-1]:
        nested = selected[name]
        assert isinstance(nested, dict)
        selected = nested
    original = selected[path[-1]]
    assert isinstance(original, float)
    selected[path[-1]] = original + 1e-8

    with pytest.raises(RuntimeError, match="trusted numerical reference"):
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

    with pytest.raises(RuntimeError, match="exact warning"):
        smoke_wheel._validate_example(payload)


def test_warning_substrings_cannot_hide_a_contradictory_claim() -> None:
    payload = _valid_payload()
    warnings = payload["warnings"]
    assert isinstance(warnings, list)
    warnings[0] = f"{warnings[0]} This result is certified."

    with pytest.raises(RuntimeError, match="exact warning"):
        smoke_wheel._validate_example(payload)


def test_independent_accuracy_reference_rejects_matching_regression_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    parameters = payload["parameters"]
    assert isinstance(parameters, dict)
    h_summary = parameters["h"]
    assert isinstance(h_summary, dict)
    h_summary["median"] = 2.4
    poisoned_oracle = copy.deepcopy(smoke_wheel._oracle_document())
    oracle_parameters = poisoned_oracle["parameters"]
    assert isinstance(oracle_parameters, dict)
    oracle_h = oracle_parameters["h"]
    assert isinstance(oracle_h, dict)
    oracle_h["median"] = 2.4
    monkeypatch.setattr(smoke_wheel, "_oracle_document", lambda: poisoned_oracle)

    with pytest.raises(RuntimeError, match="against independent quadrature"):
        smoke_wheel._validate_example(payload)


def test_public_example_must_match_independent_installed_estimator() -> None:
    independent = _valid_payload()
    hardcoded = copy.deepcopy(independent)
    parameters = hardcoded["parameters"]
    assert isinstance(parameters, dict)
    h_summary = parameters["h"]
    assert isinstance(h_summary, dict)
    h_summary["mean"] = 2.0

    with pytest.raises(
        RuntimeError, match="independently executed installed estimator"
    ):
        smoke_wheel._assert_science_payload_parity(hardcoded, independent)


def test_fresh_simulator_probe_disagreement_reports_both_observations() -> None:
    payload = _valid_payload()
    public = payload["simulation"]
    assert isinstance(public, dict)
    independent = copy.deepcopy(public)
    independent["sample_sha256"] = "f" * 64

    with pytest.raises(RuntimeError, match="different observations") as captured:
        smoke_wheel._assert_simulation_probe_parity(public, independent)

    message = str(captured.value)
    assert "public=" in message
    assert "independent=" in message
    assert "a39752ae" in message
    assert "ffffffff" in message


def test_installed_estimator_cannot_hardcode_the_primary_posterior() -> None:
    primary = _valid_payload()
    counts = primary["counts"]
    assert isinstance(counts, dict)
    quadrature = primary["quadrature"]
    assert isinstance(quadrature, dict)
    reflection = {
        "status": primary["status"],
        "method": primary["method"],
        "parameterization": primary["parameterization"],
        "counts": {
            "n_minus": counts["n_plus"],
            "n_zero": counts["n_zero"],
            "n_plus": counts["n_minus"],
            "n": counts["n"],
            "threshold": counts["threshold"],
        },
        "log_normalizer": quadrature["log_normalizer"],
        "parameters": copy.deepcopy(primary["parameters"]),
        "identification": copy.deepcopy(primary["identification"]),
        "warnings": copy.deepcopy(primary["warnings"]),
    }

    with pytest.raises(RuntimeError, match="reflected p"):
        smoke_wheel._validate_reflection_probe(primary, reflection)


def test_independent_oracle_generator_never_imports_stableboundary() -> None:
    generator = smoke_wheel.REPOSITORY / "scripts" / "generate_artifact_oracle.py"
    source = generator.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "stableboundary" not in imported_modules


def test_oracle_records_measured_positive_refinement_envelopes() -> None:
    document = json.loads(smoke_wheel.ORACLE.read_text(encoding="utf-8"))
    tolerances = document["tolerances"]
    assert tolerances["refinement_noise_scale_upper"] == 5e-14
    evidence = tolerances["refinement_noise_scale_evidence"]
    assert evidence["ci_run_id"] == 32763920150
    assert evidence["ci_run_url"] == (
        "https://github.com/moeketsims/stableboundary/actions/runs/32763920150"
    )
    assert evidence["observed_positive_envelopes"] == {
        "log-normalizer change": {
            "minimum": 3.552713678800501e-15,
            "maximum": 5.329070518200751e-15,
        },
        "predictive negative": {
            "minimum": 9.485409843146625e-15,
            "maximum": 1.1951616402364784e-14,
        },
        "predictive positive": {
            "minimum": 3.0241327034972768e-15,
            "maximum": 5.645047713194905e-15,
        },
        "alpha mean": {
            "minimum": 1.5237943101242703e-14,
            "maximum": 3.809485775310675e-14,
        },
        "beta mean": {
            "minimum": 2.2204460492503135e-15,
            "maximum": 4.163336342344338e-15,
        },
        "h mean": {
            "minimum": 1.6579330501069005e-15,
            "maximum": 2.6053233644537005e-15,
        },
        "p mean": {
            "minimum": 2.2204460492503135e-15,
            "maximum": 3.700743415417189e-15,
        },
        "tau_plus mean": {
            "minimum": 2.1216195530814406e-15,
            "maximum": 3.536032588469068e-15,
        },
        "tau_minus mean": {
            "minimum": 4.420040735586335e-16,
            "maximum": 1.119743653015205e-15,
        },
    }


def test_oracle_approves_only_measured_platform_dependency_combinations() -> None:
    document = json.loads(smoke_wheel.ORACLE.read_text(encoding="utf-8"))
    contract = document["simulation_contract"]
    approved = contract["approved_environments"]

    assert contract["raw_hash_policy"] == "diagnostic_only"
    assert contract["quantization_steps"] == [
        "1e-10",
        "1e-11",
        "1e-12",
        "1e-13",
        "1e-14",
    ]
    assert contract["approval_evidence"] == {
        "ci_run_id": 32761069162,
        "ci_run_url": (
            "https://github.com/moeketsims/stableboundary/actions/runs/32761069162"
        ),
        "normative_selection": (
            "1e-12 was the finest full-sample quantization grid identical across "
            "all observed Windows, Linux, and Darwin jobs; 1e-13 and 1e-14 "
            "diverged and remain diagnostic only."
        ),
    }
    assert set(approved) == {
        "system=Darwin|machine=arm64|numpy=2.5.2|scipy=1.18.1",
        "system=Linux|machine=x86_64|numpy=2.2.0|scipy=1.18.0",
        "system=Linux|machine=x86_64|numpy=2.5.2|scipy=1.18.1",
        "system=Windows|machine=AMD64|numpy=2.2.0|scipy=1.18.0",
        "system=Windows|machine=AMD64|numpy=2.2.0|scipy=1.18.1",
        "system=Windows|machine=AMD64|numpy=2.5.2|scipy=1.18.0",
        "system=Windows|machine=AMD64|numpy=2.5.2|scipy=1.18.1",
    }
    assert all(
        evidence["normative_quantization_step"] == "1e-12"
        for evidence in approved.values()
    )
    assert {
        evidence["quantized_sample_sha256"]["1e-12"] for evidence in approved.values()
    } == {"ef77eb05096ae0713fd78bc4691206a7a0efa40c48077773c0b57e570657c467"}


def test_oracle_preserves_observed_raw_and_fine_grid_diagnostics() -> None:
    document = json.loads(smoke_wheel.ORACLE.read_text(encoding="utf-8"))
    approved = document["simulation_contract"]["approved_environments"]
    linux = approved["system=Linux|machine=x86_64|numpy=2.5.2|scipy=1.18.1"]

    assert linux["observed_raw_sha256"] == [
        "0587a2b545297ad8dbb6f7c033ce58dd6c7f7c8fce15973d4c41f142b4f94511",
        "64db08b6e7d395a9ed3ff620affff5797d9cdd1fd67ae126df9693c8aebfae79",
    ]
    assert linux["observed_non_normative_quantized_sha256"]["1e-14"] == [
        "7090515105d587324069c5467fd2d36dfbea2ea01d22094a95e4df77fa0d0cda",
        "77569e1202b2fe4b2c42e806e245816067070bc6f8d9f4c0a0be404a11838498",
    ]


def test_invalid_non_normative_oracle_diagnostics_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    poisoned_oracle = copy.deepcopy(smoke_wheel._oracle_document())
    contract = poisoned_oracle["simulation_contract"]
    assert isinstance(contract, dict)
    approved = contract["approved_environments"]
    assert isinstance(approved, dict)
    environment = approved["system=Windows|machine=AMD64|numpy=2.5.2|scipy=1.18.1"]
    assert isinstance(environment, dict)
    diagnostics = environment["observed_non_normative_quantized_sha256"]
    assert isinstance(diagnostics, dict)
    diagnostics["1e-14"] = ["0" * 64]
    monkeypatch.setattr(smoke_wheel, "_oracle_document", lambda: poisoned_oracle)

    with pytest.raises(RuntimeError, match="invalid non-normative hashes"):
        smoke_wheel._validate_example(payload)


def test_oracle_cannot_restore_a_finer_unportable_normative_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    poisoned_oracle = copy.deepcopy(smoke_wheel._oracle_document())
    contract = poisoned_oracle["simulation_contract"]
    assert isinstance(contract, dict)
    approved = contract["approved_environments"]
    assert isinstance(approved, dict)
    environment = approved["system=Windows|machine=AMD64|numpy=2.5.2|scipy=1.18.1"]
    assert isinstance(environment, dict)
    environment["normative_quantization_step"] = "1e-14"
    monkeypatch.setattr(smoke_wheel, "_oracle_document", lambda: poisoned_oracle)

    with pytest.raises(RuntimeError, match="approved normative grid"):
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
    assert "--isolated" in calls[0]
    assert "--no-input" in calls[0]
    assert "--no-compile" in calls[0]
    assert "find_spec('stableboundary') is None" in calls[1][-1]
    assert "--no-deps" in calls[2]
    assert "--isolated" in calls[2]
    assert "--no-input" in calls[2]
    assert "--no-compile" in calls[2]
    assert calls[2][-1] == str(artifact)
    assert calls[3][-5:] == [
        "pip",
        "--isolated",
        "--no-input",
        "check",
        "--disable-pip-version-check",
    ]


def test_stage_subprocess_uses_allowlisted_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, str] = {}

    def record(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args
        raw_environment = kwargs["env"]
        assert isinstance(raw_environment, dict)
        observed.update(raw_environment)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="")

    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("PIP_TARGET", str(tmp_path / "attacker-target"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "startup-hook"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "fake-home"))
    monkeypatch.setattr(smoke_wheel.subprocess, "run", record)

    smoke_wheel._run(
        ["python"],
        cwd=tmp_path,
        stage="allowlist proof",
        timeout_seconds=17.0,
    )

    assert "PIP_INDEX_URL" not in observed
    assert "PIP_TARGET" not in observed
    assert "PYTHONPATH" not in observed
    assert "PYTHONHOME" not in observed
    assert observed["PIP_CONFIG_FILE"] == os.devnull
    assert observed["PYTHONDONTWRITEBYTECODE"] == "1"


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
            "platform_system": "Windows",
            "platform_machine": "AMD64",
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
