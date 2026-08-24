"""Inspect and exercise both built stableboundary distribution archives."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unicodedata
import zipfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

REPOSITORY = Path(__file__).resolve().parents[1]
DIST = REPOSITORY / "dist"
EXAMPLE = REPOSITORY / "examples" / "known_nuisance_fit.py"
ORACLE = REPOSITORY / "scripts" / "artifact_oracle.json"
PROJECT_NAME = "stableboundary"
PROJECT_VERSION = "0.1.0"
EXPECTED_WHEEL = f"{PROJECT_NAME}-{PROJECT_VERSION}-py3-none-any.whl"
EXPECTED_SDIST = f"{PROJECT_NAME}-{PROJECT_VERSION}.tar.gz"
EXPECTED_SDIST_ROOT = f"{PROJECT_NAME}-{PROJECT_VERSION}"
DIST_INFO = f"{PROJECT_NAME}-{PROJECT_VERSION}.dist-info"
QUANTITIES = {"h", "p", "alpha", "beta", "tau_plus", "tau_minus"}
REFINEMENT_NOISE_COMPONENTS = frozenset(
    {
        "log-normalizer change",
        "predictive negative",
        "predictive positive",
        "h mean",
        "p mean",
        "tau_plus mean",
        "tau_minus mean",
    }
)
VENV_TIMEOUT_SECONDS = 180.0
INSTALL_TIMEOUT_SECONDS = 600.0
IMPORT_TIMEOUT_SECONDS = 60.0
EXAMPLE_TIMEOUT_SECONDS = 180.0
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_WHEEL_BYTES = 4 * 1024
EXPECTED_WHEEL_HEADERS = {
    "Wheel-Version": ["1.0"],
    "Generator": ["hatchling 1.32.0"],
    "Root-Is-Purelib": ["true"],
    "Tag": ["py3-none-any"],
}
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
WINDOWS_RESERVED_STEMS = {
    "aux",
    "clock$",
    "con",
    "conin$",
    "conout$",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
SYSTEM_ENVIRONMENT_KEYS = frozenset({"SYSTEMROOT", "WINDIR"})


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Trusted byte identity for one source or installed package file."""

    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArchiveInspection:
    """Source identities retained after exact archive inspection."""

    package_files: dict[str, FileIdentity]
    wheel_files: dict[str, bytes]
    sdist_files: dict[str, bytes]
    wheel_sha256: str
    sdist_sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    """Private artifact copy authenticated before every execution stage."""

    source_name: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TreeEntryIdentity:
    """Type and byte identity for one entry in a private environment."""

    kind: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TreeInventory:
    """Complete non-following inventory for a virtual environment."""

    entries: dict[str, TreeEntryIdentity]
    directories: frozenset[str]


@dataclass(frozen=True, slots=True)
class InstalledDistribution:
    """Raw-path proof completed before any installed package import."""

    site_packages: Path
    import_origin: Path


def _archives() -> tuple[Path, Path]:
    if not DIST.is_dir():
        raise RuntimeError(f"distribution directory does not exist: {DIST}")
    entries = sorted(DIST.iterdir(), key=lambda path: path.name)
    expected_names = {EXPECTED_WHEEL, EXPECTED_SDIST}
    if {path.name for path in entries} != expected_names or any(
        not path.is_file() for path in entries
    ):
        raise RuntimeError(
            "dist must contain exactly the expected wheel and sdist; "
            f"found {[path.name for path in entries]!r}"
        )
    by_name = {path.name: path.resolve() for path in entries}
    wheel = by_name[EXPECTED_WHEEL]
    sdist = by_name[EXPECTED_SDIST]
    for artifact in (wheel, sdist):
        if not artifact.is_relative_to(DIST.resolve()):
            raise RuntimeError(
                f"artifact escaped the repository dist directory: {artifact}"
            )
        if artifact.stat().st_size > MAX_ARCHIVE_BYTES:
            raise RuntimeError(f"artifact is unreasonably large: {artifact.name}")
    return wheel, sdist


def _once_read_artifact(source: Path, destination: Path) -> ArtifactSnapshot:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"artifact source is not a regular file: {source}")
    with source.open("rb") as stream:
        content = stream.read(MAX_ARCHIVE_BYTES + 1)
    if len(content) > MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"artifact is unreasonably large: {source.name}")
    digest = hashlib.sha256(content).hexdigest()
    addressed_directory = destination.parent / digest
    addressed_directory.mkdir(mode=0o700)
    addressed_destination = addressed_directory / destination.name
    addressed_destination.write_bytes(content)
    addressed_destination.chmod(stat.S_IREAD)
    return ArtifactSnapshot(
        source_name=source.name,
        path=addressed_destination,
        size=len(content),
        sha256=digest,
    )


def _assert_snapshot(snapshot: ArtifactSnapshot) -> None:
    path = snapshot.path
    if path.name != snapshot.source_name or not path.is_file() or path.is_symlink():
        raise RuntimeError(
            f"artifact snapshot identity changed: {snapshot.source_name}"
        )
    if path.stat().st_size != snapshot.size:
        raise RuntimeError(f"artifact snapshot size changed: {snapshot.source_name}")
    with path.open("rb") as stream:
        content = stream.read(MAX_ARCHIVE_BYTES + 1)
    if (
        len(content) != snapshot.size
        or hashlib.sha256(content).hexdigest() != snapshot.sha256
    ):
        raise RuntimeError(f"artifact snapshot bytes changed: {snapshot.source_name}")


@contextmanager
def _artifact_snapshots(
    wheel: Path, sdist: Path
) -> Iterator[tuple[ArtifactSnapshot, ArtifactSnapshot]]:
    """Yield private, read-only copies made from one bounded source read each."""
    with tempfile.TemporaryDirectory(prefix="stableboundary-artifacts-") as temporary:
        root = Path(temporary).resolve()
        if root.is_relative_to(REPOSITORY.resolve()):
            raise RuntimeError("artifact snapshot directory is inside the repository")
        root.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        wheel_snapshot = _once_read_artifact(wheel, root / EXPECTED_WHEEL)
        sdist_snapshot = _once_read_artifact(sdist, root / EXPECTED_SDIST)
        _assert_snapshot(wheel_snapshot)
        _assert_snapshot(sdist_snapshot)
        yield wheel_snapshot, sdist_snapshot


@contextmanager
def _artifact_snapshot(
    source: Path, *, expected_digest: str
) -> Iterator[ArtifactSnapshot]:
    """Yield one final-boundary, content-addressed authenticated copy."""
    with tempfile.TemporaryDirectory(prefix="stableboundary-artifact-") as temporary:
        root = Path(temporary).resolve()
        if root.is_relative_to(REPOSITORY.resolve()):
            raise RuntimeError("artifact snapshot directory is inside the repository")
        snapshot = _once_read_artifact(source, root / source.name)
        if snapshot.sha256 != expected_digest:
            raise RuntimeError(
                f"artifact source changed after inspection: {source.name}"
            )
        _assert_snapshot(snapshot)
        try:
            yield snapshot
        finally:
            _assert_snapshot(snapshot)


def _validated_archive_path(artifact: Path, name: str, *, subject: str) -> str:
    """Return a canonical portable path after rejecting extraction hazards."""
    if not name or any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in name
    ):
        raise RuntimeError(f"invalid {subject} in {artifact.name}: {name!r}")
    if "\\" in name:
        raise RuntimeError(
            f"backslash is forbidden in {subject} in {artifact.name}: {name}"
        )
    if name.startswith("/") or PureWindowsPath(name).drive:
        raise RuntimeError(f"absolute {subject} in {artifact.name}: {name}")

    raw_parts = name.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise RuntimeError(
            f"noncanonical component in {subject} in {artifact.name}: {name}"
        )
    for component in raw_parts:
        if ":" in component:
            raise RuntimeError(
                f"colon is forbidden in {subject} in {artifact.name}: {name}"
            )
        if component.rstrip(" .") != component:
            raise RuntimeError(
                f"trailing dot or space in {subject} in {artifact.name}: {name}"
            )
        if unicodedata.normalize("NFKC", component) != component:
            raise RuntimeError(
                f"noncanonical Unicode in {subject} in {artifact.name}: {name}"
            )
        reserved_stem = component.split(".", maxsplit=1)[0].casefold()
        if reserved_stem in WINDOWS_RESERVED_STEMS:
            raise RuntimeError(
                f"Windows-reserved component in {subject} in {artifact.name}: {name}"
            )

    path = PurePosixPath(*raw_parts)
    if path.is_absolute() or path.as_posix() != name:
        raise RuntimeError(f"noncanonical {subject} in {artifact.name}: {name}")
    return name


def _portable_collision_key(name: str) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFKC", component).casefold()
        for component in name.split("/")
    )


def _assert_members_safe(
    artifact: Path, members: Iterable[str], *, wheel: bool
) -> tuple[str, ...]:
    normalized = [
        _validated_archive_path(artifact, name, subject="archive member")
        for name in members
    ]
    seen: dict[tuple[str, ...], str] = {}
    for name in normalized:
        key = _portable_collision_key(name)
        previous = seen.get(key)
        if previous is not None:
            raise RuntimeError(
                f"portable path collision in {artifact.name}: {previous!r} and {name!r}"
            )
        for prefix_length in range(1, len(key)):
            prefix = key[:prefix_length]
            previous = seen.get(prefix)
            if previous is not None:
                raise RuntimeError(
                    f"file/directory collision in {artifact.name}: "
                    f"{previous!r} and {name!r}"
                )
        if any(existing[: len(key)] == key for existing in seen):
            previous = next(
                seen[existing] for existing in seen if existing[: len(key)] == key
            )
            raise RuntimeError(
                f"file/directory collision in {artifact.name}: "
                f"{name!r} and {previous!r}"
            )
        seen[key] = name
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
    return tuple(normalized)


def _file_identity(content: bytes) -> FileIdentity:
    return FileIdentity(size=len(content), sha256=hashlib.sha256(content).hexdigest())


def _repository_file(path: Path) -> bytes:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPOSITORY.resolve()) or path.is_symlink():
        raise RuntimeError(f"repository manifest contains an unsafe file: {path}")
    if not path.is_file():
        raise RuntimeError(f"repository manifest file is missing: {path}")
    content = path.read_bytes()
    if len(content) > MAX_METADATA_BYTES:
        raise RuntimeError(f"repository manifest file is too large: {path}")
    return content


def _repository_package_payload() -> dict[str, bytes]:
    package_root = REPOSITORY / "src" / PROJECT_NAME
    payload: dict[str, bytes] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        name = path.relative_to(package_root).as_posix()
        _validated_archive_path(path, name, subject="repository package path")
        payload[name] = _repository_file(path)
    if not payload or "py.typed" not in payload or "__init__.py" not in payload:
        raise RuntimeError("repository package manifest is incomplete")
    return payload


def _bounded_zip_read(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    maximum: int,
    expected_size: int | None = None,
) -> bytes:
    if member.file_size < 0 or member.file_size > maximum:
        raise RuntimeError(
            f"wheel member has an invalid size: {member.filename}={member.file_size}"
        )
    if expected_size is not None and member.file_size != expected_size:
        raise RuntimeError(
            f"wheel member size differs from repository source: {member.filename}"
        )
    content = archive.read(member)
    if len(content) != member.file_size:
        raise RuntimeError(f"wheel member was truncated: {member.filename}")
    return content


def _bounded_tar_read(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    maximum: int,
    expected_size: int | None = None,
) -> bytes:
    if member.size < 0 or member.size > maximum:
        raise RuntimeError(
            f"sdist member has an invalid size: {member.name}={member.size}"
        )
    if expected_size is not None and member.size != expected_size:
        raise RuntimeError(
            f"sdist member size differs from repository source: {member.name}"
        )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"could not read sdist member: {member.name}")
    content = extracted.read(member.size + 1)
    if len(content) != member.size:
        raise RuntimeError(f"sdist member was truncated: {member.name}")
    return content


def _metadata_body(content: bytes, *, artifact: Path, subject: str) -> bytes:
    candidates = [
        (content.find(separator), separator)
        for separator in (b"\r\n\r\n", b"\n\n")
        if content.find(separator) >= 0
    ]
    if not candidates:
        raise RuntimeError(f"invalid {subject} body in {artifact.name}")
    index, separator = min(candidates, key=lambda item: item[0])
    headers = content[:index]
    body = content[index + len(separator) :]
    if not headers or not body:
        raise RuntimeError(f"invalid {subject} body in {artifact.name}")
    return body


def _canonical_requirement(requirement: str, *, extra: str | None = None) -> str:
    match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)(.*)", requirement.strip())
    if match is None or ";" in requirement or "[" in requirement:
        raise RuntimeError(
            f"unsupported dependency syntax in pyproject.toml: {requirement!r}"
        )
    name, specifiers = match.groups()
    clauses = [clause.strip() for clause in specifiers.split(",") if clause.strip()]
    normalized = name + ",".join(sorted(clauses))
    if extra is not None:
        normalized += f"; extra == '{extra}'"
    return normalized


def _metadata_expectations() -> tuple[dict[str, list[str]], bytes, bytes]:
    pyproject_bytes = _repository_file(REPOSITORY / "pyproject.toml")
    try:
        document = tomllib.loads(pyproject_bytes.decode("utf-8"))
        project = document["project"]
        authors = project["authors"]
        optional = project.get("optional-dependencies", {})
    except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(
            "repository pyproject.toml has invalid project metadata"
        ) from error
    if project.get("name") != PROJECT_NAME or project.get("version") != PROJECT_VERSION:
        raise RuntimeError("repository pyproject.toml identity is unexpected")
    if not isinstance(authors, list) or authors != [{"name": "Moeketsi Mosia"}]:
        raise RuntimeError("repository pyproject.toml authors are unexpected")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise RuntimeError("repository pyproject.toml dependencies are invalid")
    if not isinstance(optional, dict) or not all(
        isinstance(name, str)
        and isinstance(items, list)
        and all(isinstance(item, str) for item in items)
        for name, items in optional.items()
    ):
        raise RuntimeError(
            "repository pyproject.toml optional dependencies are invalid"
        )
    requirements = [_canonical_requirement(item) for item in dependencies]
    for extra, items in optional.items():
        requirements.extend(_canonical_requirement(item, extra=extra) for item in items)
    license_bytes = _repository_file(REPOSITORY / "LICENSE")
    expected = {
        "Metadata-Version": ["2.5"],
        "Name": [PROJECT_NAME],
        "Version": [PROJECT_VERSION],
        "Summary": [str(project["description"])],
        "Author": ["Moeketsi Mosia"],
        "License-File": ["LICENSE"],
        "Requires-Python": [str(project["requires-python"])],
        "Requires-Dist": requirements,
        "Provides-Extra": list(optional),
        "Description-Content-Type": ["text/markdown"],
    }
    return expected, _repository_file(REPOSITORY / "README.md"), license_bytes


def _validate_metadata(artifact: Path, content: bytes, *, subject: str) -> None:
    if len(content) > MAX_METADATA_BYTES:
        raise RuntimeError(f"oversized {subject} in {artifact.name}")
    try:
        metadata = BytesParser(policy=policy.default).parsebytes(content)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid {subject} in {artifact.name}") from error
    expected, readme_bytes, license_bytes = _metadata_expectations()
    expected_headers = set(expected) | {"License"}
    if set(metadata.keys()) != expected_headers:
        raise RuntimeError(
            f"{subject} in {artifact.name} has unexpected metadata headers"
        )
    for name, values in expected.items():
        actual = [str(value) for value in metadata.get_all(name, failobj=[])]
        if actual != values:
            raise RuntimeError(
                f"{subject} in {artifact.name} has unexpected {name}: {actual!r}"
            )
    licenses = [str(value) for value in metadata.get_all("License", failobj=[])]
    if len(licenses) != 1 or " ".join(licenses[0].split()) != " ".join(
        license_bytes.decode("utf-8").split()
    ):
        raise RuntimeError(f"{subject} in {artifact.name} has unexpected License")
    metadata_description = _metadata_body(
        content, artifact=artifact, subject=subject
    ).replace(b"\r\n", b"\n")
    repository_description = readme_bytes.replace(b"\r\n", b"\n")
    if b"\r" in metadata_description or metadata_description != repository_description:
        raise RuntimeError(f"{subject} in {artifact.name} does not embed README.md")


def _validate_wheel_file(artifact: Path, content: bytes) -> None:
    if len(content) > MAX_WHEEL_BYTES:
        raise RuntimeError(f"oversized WHEEL in {artifact.name}")
    try:
        wheel_metadata = BytesParser(policy=policy.default).parsebytes(content)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid WHEEL in {artifact.name}") from error
    if set(wheel_metadata.keys()) != set(EXPECTED_WHEEL_HEADERS):
        raise RuntimeError(f"WHEEL in {artifact.name} has unexpected headers")
    for name, expected in EXPECTED_WHEEL_HEADERS.items():
        actual = [str(value) for value in wheel_metadata.get_all(name, failobj=[])]
        if actual != expected:
            raise RuntimeError(
                f"WHEEL in {artifact.name} has unexpected {name}: {actual!r}"
            )
    if wheel_metadata.get_payload() not in {"", None}:
        raise RuntimeError(f"WHEEL in {artifact.name} has an unexpected body")


def _record_digest(content: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return encoded.rstrip(b"=").decode("ascii")


def _validate_record(
    artifact: Path,
    content: bytes,
    members: dict[str, bytes],
) -> None:
    if len(content) > MAX_RECORD_BYTES:
        raise RuntimeError(f"oversized RECORD in {artifact.name}")
    record_path = f"{DIST_INFO}/RECORD"
    try:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8"), newline="")))
    except (csv.Error, UnicodeDecodeError) as error:
        raise RuntimeError(f"invalid RECORD in {artifact.name}") from error
    if any(len(row) != 3 for row in rows):
        raise RuntimeError(f"RECORD in {artifact.name} has malformed rows")
    names = [row[0] for row in rows]
    try:
        _assert_members_safe(artifact, names, wheel=False)
    except RuntimeError as error:
        raise RuntimeError(f"RECORD in {artifact.name} has unsafe paths") from error
    if len(names) != len(set(names)) or set(names) != set(members):
        raise RuntimeError(f"RECORD in {artifact.name} does not cover exact members")
    for name, digest, size_text in rows:
        if name == record_path:
            if digest or size_text:
                raise RuntimeError(f"RECORD self-entry in {artifact.name} is not empty")
            continue
        if not digest.startswith("sha256=") or not size_text.isascii():
            raise RuntimeError(
                f"RECORD entry in {artifact.name} is not canonical: {name}"
            )
        encoded = digest.removeprefix("sha256=")
        try:
            decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except (ValueError, base64.binascii.Error) as error:
            raise RuntimeError(
                f"invalid RECORD digest in {artifact.name}: {name}"
            ) from error
        if (
            len(decoded) != hashlib.sha256().digest_size
            or base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != encoded
            or encoded != _record_digest(members[name])
        ):
            raise RuntimeError(f"RECORD digest mismatch in {artifact.name}: {name}")
        expected_size = str(len(members[name]))
        if size_text != expected_size:
            raise RuntimeError(f"RECORD size mismatch in {artifact.name}: {name}")


def _scientific_payloads_match(
    wheel_payload: dict[str, bytes], sdist_payload: dict[str, bytes]
) -> None:
    if wheel_payload.keys() != sdist_payload.keys():
        wheel_only = sorted(wheel_payload.keys() - sdist_payload.keys())
        sdist_only = sorted(sdist_payload.keys() - wheel_payload.keys())
        raise RuntimeError(
            "wheel/sdist package payloads differ: "
            f"wheel-only={wheel_only!r}, sdist-only={sdist_only!r}"
        )
    mismatches = [
        name
        for name in wheel_payload
        if hashlib.sha256(wheel_payload[name]).digest()
        != hashlib.sha256(sdist_payload[name]).digest()
    ]
    if mismatches:
        raise RuntimeError(
            f"wheel/sdist scientific payload bytes differ: {sorted(mismatches)!r}"
        )


def _inspect_archives(wheel: Path, sdist: Path) -> ArchiveInspection:
    for artifact in (wheel, sdist):
        if not artifact.is_file() or artifact.stat().st_size > MAX_ARCHIVE_BYTES:
            raise RuntimeError(f"artifact is missing or unreasonably large: {artifact}")
    repository_payload = _repository_package_payload()
    expected_metadata, readme_bytes, license_bytes = _metadata_expectations()
    del expected_metadata
    repository_sdist_payload = {
        ".gitignore": _repository_file(REPOSITORY / ".gitignore"),
        "LICENSE": license_bytes,
        "README.md": readme_bytes,
        "pyproject.toml": _repository_file(REPOSITORY / "pyproject.toml"),
        "uv.lock": _repository_file(REPOSITORY / "uv.lock"),
        f"examples/{EXAMPLE.name}": _repository_file(EXAMPLE),
    }
    wheel_package_names = {
        f"{PROJECT_NAME}/{name}": content
        for name, content in repository_payload.items()
    }
    metadata_path = f"{DIST_INFO}/METADATA"
    wheel_path = f"{DIST_INFO}/WHEEL"
    license_path = f"{DIST_INFO}/licenses/LICENSE"
    record_path = f"{DIST_INFO}/RECORD"
    expected_wheel_names = set(wheel_package_names) | {
        metadata_path,
        wheel_path,
        license_path,
        record_path,
    }

    with zipfile.ZipFile(wheel) as archive:
        wheel_infos = archive.infolist()
        wheel_names = _assert_members_safe(
            wheel, (member.filename for member in wheel_infos), wheel=True
        )
        if set(wheel_names) != expected_wheel_names:
            raise RuntimeError(
                "wheel members do not match the exact source-bound manifest: "
                f"missing={sorted(expected_wheel_names - set(wheel_names))!r}, "
                f"unexpected={sorted(set(wheel_names) - expected_wheel_names)!r}"
            )
        for member in wheel_infos:
            unix_type = (member.external_attr >> 16) & 0o170000
            if member.flag_bits & 0x1:
                raise RuntimeError(
                    f"encrypted wheel member in {wheel.name}: {member.filename}"
                )
            if unix_type == stat.S_IFLNK:
                raise RuntimeError(
                    f"symbolic links are forbidden in {wheel.name}: {member.filename}"
                )
            if unix_type not in {0, stat.S_IFREG}:
                raise RuntimeError(
                    f"non-regular wheel member in {wheel.name}: {member.filename}"
                )
        info_by_name = {member.filename: member for member in wheel_infos}
        wheel_members: dict[str, bytes] = {}
        for name, member in info_by_name.items():
            if name in wheel_package_names:
                expected = wheel_package_names[name]
                content = _bounded_zip_read(
                    archive,
                    member,
                    maximum=MAX_METADATA_BYTES,
                    expected_size=len(expected),
                )
                if content != expected:
                    raise RuntimeError(
                        f"wheel package member differs from repository source: {name}"
                    )
            elif name == license_path:
                content = _bounded_zip_read(
                    archive,
                    member,
                    maximum=MAX_METADATA_BYTES,
                    expected_size=len(license_bytes),
                )
                if content != license_bytes:
                    raise RuntimeError("wheel license differs from repository LICENSE")
            else:
                maximum = (
                    MAX_WHEEL_BYTES
                    if name == wheel_path
                    else MAX_RECORD_BYTES
                    if name == record_path
                    else MAX_METADATA_BYTES
                )
                content = _bounded_zip_read(archive, member, maximum=maximum)
            wheel_members[name] = content
        _validate_metadata(
            wheel, wheel_members[metadata_path], subject="wheel METADATA"
        )
        _validate_wheel_file(wheel, wheel_members[wheel_path])
        _validate_record(wheel, wheel_members[record_path], wheel_members)
        wheel_payload = {
            name.removeprefix(f"{PROJECT_NAME}/"): content
            for name, content in wheel_members.items()
            if name.startswith(f"{PROJECT_NAME}/")
        }

    with tarfile.open(sdist, mode="r:*") as archive:
        members = archive.getmembers()
        sdist_names = _assert_members_safe(
            sdist, (member.name for member in members), wheel=False
        )
        for member in members:
            if member.issym() or member.islnk():
                _validated_archive_path(
                    sdist,
                    member.linkname,
                    subject=f"link target for {member.name}",
                )
                raise RuntimeError(
                    f"links are forbidden in {sdist.name}: {member.name}"
                )
            if member.ischr() or member.isblk() or member.isfifo():
                raise RuntimeError(
                    f"special archive member in {sdist.name}: {member.name}"
                )
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    f"unsupported archive member in {sdist.name}: {member.name}"
                )
        expected_prefix = f"{EXPECTED_SDIST_ROOT}/"
        if any(not name.startswith(expected_prefix) for name in sdist_names):
            raise RuntimeError(
                f"{sdist.name} contains a member outside {EXPECTED_SDIST_ROOT!r}"
            )
        package_prefix = f"{EXPECTED_SDIST_ROOT}/src/{PROJECT_NAME}/"
        sdist_package_names = {
            f"{package_prefix}{name}": content
            for name, content in repository_payload.items()
        }
        sdist_bound_names = {
            f"{EXPECTED_SDIST_ROOT}/{name}": content
            for name, content in repository_sdist_payload.items()
        }
        pkg_info_path = f"{EXPECTED_SDIST_ROOT}/PKG-INFO"
        expected_sdist_names = (
            set(sdist_package_names) | set(sdist_bound_names) | {pkg_info_path}
        )
        if set(sdist_names) != expected_sdist_names or any(
            not member.isfile() for member in members
        ):
            raise RuntimeError(
                "sdist members do not match the exact source-bound manifest: "
                f"missing={sorted(expected_sdist_names - set(sdist_names))!r}, "
                f"unexpected={sorted(set(sdist_names) - expected_sdist_names)!r}"
            )
        member_by_name = {member.name: member for member in members}
        sdist_members: dict[str, bytes] = {}
        for name, member in member_by_name.items():
            expected = sdist_package_names.get(name, sdist_bound_names.get(name))
            content = _bounded_tar_read(
                archive,
                member,
                maximum=MAX_METADATA_BYTES,
                expected_size=None if expected is None else len(expected),
            )
            if expected is not None and content != expected:
                raise RuntimeError(
                    f"sdist member differs from repository source: {name}"
                )
            sdist_members[name] = content
        _validate_metadata(
            sdist,
            sdist_members[pkg_info_path],
            subject="sdist PKG-INFO",
        )
        if sdist_members[pkg_info_path] != wheel_members[metadata_path]:
            raise RuntimeError("wheel METADATA and sdist PKG-INFO differ byte-for-byte")
        sdist_payload = {
            name.removeprefix(package_prefix): content
            for name, content in sdist_members.items()
            if name.startswith(package_prefix)
        }

    _scientific_payloads_match(wheel_payload, sdist_payload)
    return ArchiveInspection(
        package_files={
            name: _file_identity(content)
            for name, content in repository_payload.items()
        },
        wheel_files=wheel_members,
        sdist_files=sdist_members,
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        sdist_sha256=hashlib.sha256(sdist.read_bytes()).hexdigest(),
    )


def _venv_python(environment: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    # Do not resolve the POSIX venv launcher: it is normally a symlink to the
    # base interpreter, and resolving it bypasses the adjacent pyvenv.cfg.
    executable = environment / relative
    if not executable.is_file():
        raise RuntimeError(f"virtual-environment interpreter is missing: {executable}")
    return executable


def _subprocess_environment(cwd: Path) -> dict[str, str]:
    """Return a small, deterministic environment for every child process."""
    environment = {
        name: os.environ[name] for name in SYSTEM_ENVIRONMENT_KEYS if name in os.environ
    }
    executable_directory = str(Path(sys.executable).resolve().parent)
    if os.name == "nt" and "SYSTEMROOT" in environment:
        system32 = str(Path(environment["SYSTEMROOT"]) / "System32")
        environment["PATH"] = os.pathsep.join((executable_directory, system32))
    else:
        environment["PATH"] = executable_directory
    controlled_root = cwd.resolve()
    environment.update(
        {
            "HOME": str(controlled_root),
            "USERPROFILE": str(controlled_root),
            "TEMP": str(controlled_root),
            "TMP": str(controlled_root),
            "TMPDIR": str(controlled_root),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONUTF8": "1",
        }
    )
    return environment


def _run(
    command: list[str],
    *,
    cwd: Path,
    stage: str,
    timeout_seconds: float,
    capture: bool = False,
) -> str:
    environment = _subprocess_environment(cwd)
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


def _prepare_environment(python: Path, *, cwd: Path) -> None:
    """Install dependencies and the pinned build backend before proof begins."""
    _run(
        [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "--no-input",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            "--only-binary=:all:",
            "build>=1.5,<2",
            "hatchling==1.32.0",
            "numpy>=2.2",
            "scipy>=1.18,<1.19",
        ],
        cwd=cwd,
        stage="runtime and build dependency installation",
        timeout_seconds=INSTALL_TIMEOUT_SECONDS,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "--no-input",
            "check",
            "--disable-pip-version-check",
        ],
        cwd=cwd,
        stage="dependency consistency check before artifact proof",
        timeout_seconds=IMPORT_TIMEOUT_SECONDS,
    )


def _install_verified_wheel(
    python: Path, snapshot: ArtifactSnapshot, *, cwd: Path
) -> None:
    """Consume one inspected wheel with no intervening child process."""
    if snapshot.source_name != EXPECTED_WHEEL or snapshot.path.suffix != ".whl":
        raise RuntimeError("only the inspected wheel may cross the install boundary")
    _assert_snapshot(snapshot)
    authenticated_url = f"{snapshot.path.resolve().as_uri()}#sha256={snapshot.sha256}"
    _run(
        [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "--no-input",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            "--no-deps",
            authenticated_url,
        ],
        cwd=cwd,
        stage=f"installation of {snapshot.source_name}",
        timeout_seconds=INSTALL_TIMEOUT_SECONDS,
    )
    _assert_snapshot(snapshot)


def _extract_inspected_sdist(
    inspection: ArchiveInspection, destination: Path, *, artifact: Path
) -> Path:
    """Materialize only the already-inspected, repository-bound sdist bytes."""
    destination.mkdir()
    expected_prefix = f"{EXPECTED_SDIST_ROOT}/"
    for name, content in inspection.sdist_files.items():
        canonical = _validated_archive_path(
            artifact, name, subject="sdist build member"
        )
        if not canonical.startswith(expected_prefix):
            raise RuntimeError("inspected sdist member escaped its source root")
        relative = PurePosixPath(canonical)
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    source_root = destination / EXPECTED_SDIST_ROOT
    if not source_root.is_dir():
        raise RuntimeError("inspected sdist did not materialize its source root")
    return source_root


def _build_inspected_sdist_wheel(
    python: Path,
    snapshot: ArtifactSnapshot,
    inspection: ArchiveInspection,
    *,
    cwd: Path,
) -> tuple[Path, ArchiveInspection]:
    """Build an inspected sdist tree, then inspect the exact resulting wheel."""
    _assert_snapshot(snapshot)
    source_root = _extract_inspected_sdist(
        inspection, cwd / "sdist-source", artifact=snapshot.path
    )
    source_before = _tree_inventory(source_root)
    _assert_snapshot(snapshot)
    output = cwd / "sdist-wheel"
    output.mkdir()
    _run(
        [
            str(python),
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
            str(source_root),
        ],
        cwd=cwd,
        stage=f"wheel build from inspected {snapshot.source_name}",
        timeout_seconds=INSTALL_TIMEOUT_SECONDS,
    )
    _assert_snapshot(snapshot)
    if _tree_inventory(source_root) != source_before:
        raise RuntimeError("sdist source tree changed while its wheel was being built")
    outputs = sorted(output.iterdir(), key=lambda path: path.name)
    if len(outputs) != 1 or outputs[0].name != EXPECTED_WHEEL:
        raise RuntimeError(
            "sdist build did not produce exactly the expected wheel: "
            f"{[path.name for path in outputs]!r}"
        )
    built_wheel = outputs[0].resolve()
    built_inspection = _inspect_archives(built_wheel, snapshot.path)
    if built_inspection.sdist_sha256 != snapshot.sha256:
        raise RuntimeError("sdist changed while its wheel was being built")
    return built_wheel, built_inspection


def _tree_entry_identity(path: Path) -> TreeEntryIdentity:
    if _is_windows_reparse_point(path):
        raise RuntimeError(
            f"virtual environment contains a Windows reparse point: {path}"
        )
    if path.is_symlink():
        content = os.readlink(path).encode("utf-8", errors="surrogatepass")
        return TreeEntryIdentity(
            kind="symlink",
            size=len(content),
            sha256=hashlib.sha256(b"symlink\0" + content).hexdigest(),
        )
    if not path.is_file():
        raise RuntimeError(f"virtual environment contains a special file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return TreeEntryIdentity(kind="file", size=size, sha256=digest.hexdigest())


def _is_windows_reparse_point(path: Path) -> bool:
    """Return whether *path* has Windows link-like reparse semantics."""
    if os.name != "nt":
        return False
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _tree_inventory(root: Path) -> TreeInventory:
    if not root.is_dir() or root.is_symlink() or _is_windows_reparse_point(root):
        raise RuntimeError(f"virtual environment root is invalid: {root}")
    entries: dict[str, TreeEntryIdentity] = {}
    directories: set[str] = set()
    for current, raw_directories, raw_files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(raw_directories):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if _is_windows_reparse_point(path):
                raise RuntimeError(
                    f"virtual environment contains a Windows reparse point: {path}"
                )
            if path.is_symlink():
                entries[relative] = _tree_entry_identity(path)
                raw_directories.remove(name)
            elif path.is_dir():
                directories.add(relative)
            else:
                raise RuntimeError(f"virtual environment has an invalid entry: {path}")
        for name in sorted(raw_files):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            entries[relative] = _tree_entry_identity(path)
    return TreeInventory(entries=entries, directories=frozenset(directories))


def _site_packages(environment: Path) -> Path:
    if os.name == "nt":
        candidates = [environment / "Lib" / "site-packages"]
    else:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates = [environment / "lib" / version / "site-packages"]
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if len(existing) != 1:
        raise RuntimeError(
            f"virtual environment has an unexpected site-packages layout: {existing!r}"
        )
    candidate = existing[0]
    if candidate.is_symlink() or _is_windows_reparse_point(candidate):
        raise RuntimeError(f"site-packages uses link or reparse semantics: {candidate}")
    selected = candidate.resolve()
    if (
        selected.is_symlink()
        or not selected.is_relative_to(environment.resolve())
        or selected.is_relative_to(REPOSITORY.resolve())
    ):
        raise RuntimeError(
            f"site-packages is outside the private environment: {selected}"
        )
    return selected


def _expected_environment_delta(
    environment: Path, site_packages: Path, installed_names: set[str]
) -> tuple[set[str], set[str]]:
    site_prefix = site_packages.relative_to(environment).as_posix()
    files = {f"{site_prefix}/{name}" for name in installed_names}
    directories: set[str] = set()
    for name in files:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return files, directories


def _validate_environment_delta(
    before: TreeInventory,
    after: TreeInventory,
    *,
    expected_files: set[str],
    expected_directories: set[str],
) -> None:
    removed = set(before.entries) - set(after.entries)
    changed = {
        name
        for name, identity in before.entries.items()
        if name in after.entries and after.entries[name] != identity
    }
    added = set(after.entries) - set(before.entries)
    expected_added_directories = expected_directories - set(before.directories)
    added_directories = set(after.directories) - set(before.directories)
    removed_directories = set(before.directories) - set(after.directories)
    if (
        removed
        or changed
        or added != expected_files
        or added_directories != expected_added_directories
        or removed_directories
    ):
        raise RuntimeError(
            "artifact installation changed files outside the exact distribution "
            f"manifest: removed={sorted(removed)!r}, changed={sorted(changed)!r}, "
            f"unexpected={sorted(added - expected_files)!r}, "
            f"missing={sorted(expected_files - added)!r}, "
            "unexpected_directories="
            f"{sorted(added_directories - expected_added_directories)!r}, "
            f"removed_directories={sorted(removed_directories)!r}"
        )


def _validate_direct_url(
    value: object, *, artifact: Path, expected_digest: str
) -> None:
    direct_url = _require_keys("direct_url.json", value, {"archive_info", "url"})
    raw_url = direct_url["url"]
    if not isinstance(raw_url, str):
        raise RuntimeError("installed direct_url.json has a non-string URL")
    parsed = urlparse(raw_url)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("installed direct_url.json is not a local artifact URL")
    installed_from = Path(url2pathname(parsed.path)).resolve()
    if installed_from != artifact.resolve():
        raise RuntimeError(
            "installed direct_url.json does not point to the selected artifact: "
            f"{installed_from} != {artifact.resolve()}"
        )

    archive_info = _require_keys(
        "direct_url.json archive_info",
        direct_url["archive_info"],
        {"hash", "hashes"},
    )
    hashes = _require_keys(
        "direct_url.json archive hashes", archive_info["hashes"], {"sha256"}
    )
    if hashes["sha256"] != expected_digest:
        raise RuntimeError(
            "installed direct_url.json does not retain the artifact hash"
        )
    if archive_info["hash"] != f"sha256={expected_digest}":
        raise RuntimeError("installed direct_url.json reports a contradictory hash")


def _validate_installed_distribution(
    *,
    artifact: Path,
    environment: Path,
    expected_digest: str,
    inspection: ArchiveInspection,
    before: TreeInventory,
) -> InstalledDistribution:
    """Authenticate the complete install from raw paths before any import."""
    site_packages = _site_packages(environment)
    record_path = f"{DIST_INFO}/RECORD"
    generated_names = {
        f"{DIST_INFO}/INSTALLER",
        f"{DIST_INFO}/REQUESTED",
        f"{DIST_INFO}/direct_url.json",
    }
    installed_names = set(inspection.wheel_files) | generated_names
    expected_files, expected_directories = _expected_environment_delta(
        environment, site_packages, installed_names
    )
    after = _tree_inventory(environment)
    _validate_environment_delta(
        before,
        after,
        expected_files=expected_files,
        expected_directories=expected_directories,
    )
    startup_files = sorted(
        path.name
        for path in site_packages.iterdir()
        if path.name in {"sitecustomize.py", "usercustomize.py"}
        or path.suffix.casefold() == ".pth"
    )
    if startup_files:
        raise RuntimeError(
            f"site-packages contains forbidden startup files: {startup_files!r}"
        )

    installed_files: dict[str, bytes] = {}
    site_root = site_packages.resolve()
    environment_root = environment.resolve()
    for name in sorted(installed_names):
        path = site_packages.joinpath(*PurePosixPath(name).parts)
        resolved = path.resolve(strict=True)
        if (
            not path.is_file()
            or path.is_symlink()
            or _is_windows_reparse_point(path)
            or not resolved.is_relative_to(site_root)
            or not resolved.is_relative_to(environment_root)
        ):
            raise RuntimeError(f"installed distribution file is invalid: {name}")
        installed_files[name] = path.read_bytes()
    for name, expected in inspection.wheel_files.items():
        if name != record_path and installed_files[name] != expected:
            raise RuntimeError(
                f"installed distribution file differs from inspected wheel: {name}"
            )
    if installed_files[f"{DIST_INFO}/INSTALLER"] != b"pip\n":
        raise RuntimeError("installed distribution has an unexpected INSTALLER")
    if installed_files[f"{DIST_INFO}/REQUESTED"] != b"":
        raise RuntimeError("installed distribution has an unexpected REQUESTED marker")
    try:
        direct_url = json.loads(
            installed_files[f"{DIST_INFO}/direct_url.json"].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("installed direct_url.json is invalid") from error
    _validate_direct_url(direct_url, artifact=artifact, expected_digest=expected_digest)
    _validate_record(artifact, installed_files[record_path], installed_files)
    origin = (site_packages / PROJECT_NAME / "__init__.py").resolve()
    if (
        not origin.is_file()
        or origin.is_symlink()
        or not origin.is_relative_to(environment.resolve())
        or origin.is_relative_to(REPOSITORY.resolve())
    ):
        raise RuntimeError(f"installed package origin is invalid: {origin}")
    return InstalledDistribution(site_packages=site_packages, import_origin=origin)


def _installed_probe(
    python: Path,
    *,
    cwd: Path,
    artifact: Path,
    installed: InstalledDistribution,
) -> dict[str, Any]:
    source = """
import importlib.metadata as metadata
import json
import platform
import sys

sys.path.insert(0, __SITE_PACKAGES__)

import numpy
import scipy
import stableboundary

distribution = metadata.distribution("stableboundary")
print(json.dumps({
    "import_origin": stableboundary.__file__,
    "metadata_version": distribution.version,
    "package_version": stableboundary.__version__,
    "versions": {
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "stableboundary": stableboundary.__version__,
    },
}, sort_keys=True))
"""
    source = source.replace(
        "__SITE_PACKAGES__", json.dumps(str(installed.site_packages))
    )
    decoded = json.loads(
        _run(
            [str(python), "-I", "-S", "-c", source],
            cwd=cwd,
            stage=f"installed provenance probe for {artifact.name}",
            timeout_seconds=IMPORT_TIMEOUT_SECONDS,
            capture=True,
        )
    )
    if not isinstance(decoded, dict):
        raise RuntimeError("installed provenance probe did not return a JSON object")
    return decoded


def _validate_installed_runtime(
    probe: object, *, installed: InstalledDistribution, artifact: Path
) -> dict[str, Any]:
    values = _require_keys(
        "installed runtime probe",
        probe,
        {"import_origin", "metadata_version", "package_version", "versions"},
    )
    if (
        values["metadata_version"] != PROJECT_VERSION
        or values["package_version"] != PROJECT_VERSION
    ):
        raise RuntimeError(f"installed metadata version does not match {artifact.name}")
    origin_value = values["import_origin"]
    if not isinstance(origin_value, str):
        raise RuntimeError("installed runtime returned an invalid import origin")
    if Path(origin_value).resolve() != installed.import_origin:
        raise RuntimeError("installed runtime imported unverified package bytes")
    versions = _require_keys(
        "installed runtime versions",
        values["versions"],
        {
            "python",
            "numpy",
            "scipy",
            "platform_system",
            "platform_machine",
            PROJECT_NAME,
        },
    )
    if versions[PROJECT_NAME] != PROJECT_VERSION:
        raise RuntimeError("installed package runtime version is stale")
    for name in ("python", "numpy", "scipy", "platform_system", "platform_machine"):
        if not isinstance(versions[name], str) or not versions[name].strip():
            raise RuntimeError(f"installed {name} runtime version is invalid")
    python_match = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:[A-Za-z0-9.+-]*)?", versions["python"]
    )
    if python_match is None or (
        int(python_match.group(1)),
        int(python_match.group(2)),
    ) not in {(3, 12), (3, 13), (3, 14)}:
        raise RuntimeError("installed Python runtime is unsupported")
    scipy_match = re.fullmatch(r"1\.18\.(\d+)(?:[A-Za-z0-9.+-]*)?", versions["scipy"])
    if scipy_match is None:
        raise RuntimeError("installed SciPy runtime is outside >=1.18,<1.19")
    return versions


def _check_prove_and_probe_installed_runtime(
    python: Path,
    *,
    cwd: Path,
    source_artifact: Path,
    wheel_snapshot: ArtifactSnapshot,
    environment: Path,
    inspection: ArchiveInspection,
    before: TreeInventory,
) -> tuple[InstalledDistribution, dict[str, Any]]:
    """Finish child setup, prove raw bytes, import, then prove them again."""
    _run(
        [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "--no-input",
            "check",
            "--disable-pip-version-check",
        ],
        cwd=cwd,
        stage=f"dependency check for {source_artifact.name}",
        timeout_seconds=IMPORT_TIMEOUT_SECONDS,
    )
    installed = _validate_installed_distribution(
        artifact=wheel_snapshot.path,
        environment=environment,
        expected_digest=wheel_snapshot.sha256,
        inspection=inspection,
        before=before,
    )
    probe = _installed_probe(
        python,
        cwd=cwd,
        artifact=wheel_snapshot.path,
        installed=installed,
    )
    _validate_installed_distribution(
        artifact=wheel_snapshot.path,
        environment=environment,
        expected_digest=wheel_snapshot.sha256,
        inspection=inspection,
        before=before,
    )
    return installed, probe


def _oracle_document() -> dict[str, Any]:
    try:
        decoded = json.loads(_repository_file(ORACLE))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("artifact oracle is not valid UTF-8 JSON") from error
    expected = {
        "schema_version",
        "provenance",
        "independent_reference",
        "fixture",
        "simulation_contract",
        "known_nuisance",
        "design",
        "prior",
        "counts",
        "quadrature",
        "parameters",
        "posterior_mass",
        "identification",
        "refinement",
        "warnings",
        "tolerances",
    }
    if not isinstance(decoded, dict) or set(decoded) != expected:
        raise RuntimeError("artifact oracle has an unexpected schema")
    if decoded["schema_version"] != 1:
        raise RuntimeError("artifact oracle has an unsupported schema version")
    return decoded


def _installed_science_probe(
    python: Path, *, cwd: Path, artifact: Path, import_roots: tuple[Path, ...]
) -> dict[str, Any]:
    """Recompute the public experiment without importing its example module."""
    source = r"""
import hashlib
import json
import platform
import sys

sys.path[:0] = __IMPORT_ROOTS__

import numpy as np
import stableboundary as sb

n = 5000
seed = 20260824
design = sb.LocalDesign.from_sample_size(n)
truth = sb.StableParams(alpha=2.0-design.r*1.5, beta=0.35, loc=0.0, scale=1.0)
observations = np.zeros(n, dtype=np.float64)
observations[0] = -(design.threshold + 1.0)
observations[-3:] = design.threshold + 1.0
fixture_bytes = np.ascontiguousarray(observations, dtype="<f8").tobytes(order="C")
fit = sb.fit_known_nuisance(
    observations,
    loc=0.0,
    scale=1.0,
    design=design,
    prior=sb.LocalPrior.default(design),
    provenance="fixed cell-count witness derived from the prespecified design",
    quadrature=sb.QuadratureConfig(
        base_nodes=20,
        refined_nodes=32,
        refinement_tolerance=0.002,
        common_grid_points=65,
    ),
)
reflected_fit = sb.fit_known_nuisance(
    -observations,
    loc=0.0,
    scale=1.0,
    design=design,
    prior=sb.LocalPrior.default(design),
    provenance="reflected fixed cell-count witness",
    quadrature=sb.QuadratureConfig(
        base_nodes=20,
        refined_nodes=32,
        refinement_tolerance=0.002,
        common_grid_points=65,
    ),
)
summary = fit.summary()
audit = fit.audit_record()
reflected_summary = reflected_fit.summary()
reflected_audit = reflected_fit.audit_record()
simulated = sb.simulate(truth, size=n, random_state=seed)
simulation_bytes = np.ascontiguousarray(simulated, dtype="<f8").tobytes(order="C")
quantization_steps = {f"1e-{power}": 10.0**(-power) for power in range(10, 15)}
quantized_hashes = {
    name: hashlib.sha256(
        np.ascontiguousarray(np.rint(simulated / step), dtype="<i8").tobytes(order="C")
    ).hexdigest()
    for name, step in quantization_steps.items()
}
diagnostic_quantiles = np.quantile(simulated, [0.01, 0.05, 0.5, 0.95, 0.99])
n_minus = int(np.count_nonzero(simulated <= -design.threshold))
n_plus = int(np.count_nonzero(simulated >= design.threshold))
payload = {
    "schema_version": audit["schema_version"],
    "package_version": audit["package_version"],
    "status": summary["status"],
    "method": summary["method"],
    "parameterization": summary["parameterization"],
    "known_nuisance": audit["known_nuisance"],
    "seed": seed,
    "truth": {
        "alpha": truth.alpha,
        "beta": truth.beta,
        "loc": truth.loc,
        "scale": truth.scale,
    },
    "inference_fixture": {
        "construction": "[-(threshold+1)] + [0]*4996 + [threshold+1]*3",
        "dtype": "<f8",
        "nbytes": len(fixture_bytes),
        "sha256": hashlib.sha256(fixture_bytes).hexdigest(),
    },
    "simulation": {
        "dtype": "<f8",
        "rng_algorithm": (
            "numpy.random."
            f"{np.random.default_rng(seed).bit_generator.__class__.__name__}"
        ),
        "simulator_algorithm": "scipy.stats.levy_stable.rvs:S0:private-generator:v1",
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": audit["backend"]["library_version"],
        "sample_sha256": hashlib.sha256(simulation_bytes).hexdigest(),
        "quantized_sample_sha256": quantized_hashes,
        "counts": {
            "n_minus": n_minus,
            "n_zero": int(n - n_minus - n_plus),
            "n_plus": n_plus,
        },
        "minimum": float(np.min(simulated)),
        "maximum": float(np.max(simulated)),
        "diagnostics": {
            "mean": float(np.mean(simulated)),
            "standard_deviation": float(np.std(simulated)),
            "q01": float(diagnostic_quantiles[0]),
            "q05": float(diagnostic_quantiles[1]),
            "median": float(diagnostic_quantiles[2]),
            "q95": float(diagnostic_quantiles[3]),
            "q99": float(diagnostic_quantiles[4]),
        },
    },
    "design": audit["design"],
    "prior": audit["prior"],
    "counts": audit["counts"],
    "quadrature": audit["quadrature"],
    "parameters": summary["parameters"],
    "posterior_mass": float(fit.posterior.mass.sum()),
    "identification": summary["identification"],
    "refinement": audit["refinement"],
    "backend": audit["backend"],
    "warnings": summary["warnings"],
}
print(json.dumps({
    "primary": payload,
    "reflection": {
        "status": reflected_summary["status"],
        "method": reflected_summary["method"],
        "parameterization": reflected_summary["parameterization"],
        "counts": reflected_audit["counts"],
        "log_normalizer": reflected_audit["quadrature"]["log_normalizer"],
        "parameters": reflected_summary["parameters"],
        "identification": reflected_summary["identification"],
        "warnings": reflected_summary["warnings"],
    },
}, sort_keys=True, allow_nan=False))
"""
    source = source.replace(
        "__IMPORT_ROOTS__",
        json.dumps([str(root.resolve()) for root in import_roots]),
    )
    decoded = json.loads(
        _run(
            [str(python), "-I", "-S", "-c", source],
            cwd=cwd,
            stage=f"independent installed science probe for {artifact.name}",
            timeout_seconds=EXAMPLE_TIMEOUT_SECONDS,
            capture=True,
        )
    )
    if not isinstance(decoded, dict):
        raise RuntimeError("independent installed science probe returned invalid JSON")
    return decoded


def _fresh_simulation_probe(python: Path, *, cwd: Path, stage: str) -> dict[str, Any]:
    """Run one simulator observation in an isolated interpreter process."""
    source = r"""
import hashlib
import json
import platform

import numpy as np
import scipy
import stableboundary as sb

n = 5000
seed = 20260824
design = sb.LocalDesign.from_sample_size(n)
truth = sb.StableParams(alpha=2.0-design.r*1.5, beta=0.35, loc=0.0, scale=1.0)
simulated = sb.simulate(truth, size=n, random_state=seed)
raw = np.ascontiguousarray(simulated, dtype="<f8").tobytes(order="C")
steps = {f"1e-{power}": 10.0**(-power) for power in range(10, 15)}
quantized = {
    name: hashlib.sha256(
        np.ascontiguousarray(np.rint(simulated / step), dtype="<i8").tobytes(order="C")
    ).hexdigest()
    for name, step in steps.items()
}
quantiles = np.quantile(simulated, [0.01, 0.05, 0.5, 0.95, 0.99])
n_minus = int(np.count_nonzero(simulated <= -design.threshold))
n_plus = int(np.count_nonzero(simulated >= design.threshold))
print(json.dumps({
    "dtype": "<f8",
    "rng_algorithm": (
        "numpy.random."
        f"{np.random.default_rng(seed).bit_generator.__class__.__name__}"
    ),
    "simulator_algorithm": "scipy.stats.levy_stable.rvs:S0:private-generator:v1",
    "platform_system": platform.system(),
    "platform_machine": platform.machine(),
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "sample_sha256": hashlib.sha256(raw).hexdigest(),
    "quantized_sample_sha256": quantized,
    "counts": {
        "n_minus": n_minus,
        "n_zero": int(n - n_minus - n_plus),
        "n_plus": n_plus,
    },
    "minimum": float(np.min(simulated)),
    "maximum": float(np.max(simulated)),
    "diagnostics": {
        "mean": float(np.mean(simulated)),
        "standard_deviation": float(np.std(simulated)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
    },
}, sort_keys=True, allow_nan=False))
"""
    decoded = json.loads(
        _run(
            [str(python), "-I", "-c", source],
            cwd=cwd,
            stage=stage,
            timeout_seconds=IMPORT_TIMEOUT_SECONDS,
            capture=True,
        )
    )
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{stage} returned invalid JSON")
    return decoded


def _assert_simulation_probe_parity(
    public_probe: dict[str, Any], independent_probe: dict[str, Any]
) -> None:
    if public_probe != independent_probe:
        public_json = json.dumps(public_probe, sort_keys=True, allow_nan=False)
        independent_json = json.dumps(
            independent_probe, sort_keys=True, allow_nan=False
        )
        raise RuntimeError(
            "fresh simulator subprocesses produced different observations; "
            f"public={public_json}; independent={independent_json}"
        )


def _independent_fixture_identity(threshold: float) -> dict[str, object]:
    values = (-(threshold + 1.0),) + (0.0,) * 4_996 + (threshold + 1.0,) * 3
    content = struct.pack("<5000d", *values)
    return {
        "construction": "[-(threshold+1)] + [0]*4996 + [threshold+1]*3",
        "dtype": "<f8",
        "nbytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _reference_close(
    name: str,
    actual: object,
    expected: object,
    *,
    tolerance: float,
) -> float:
    actual_value = _finite_float(name, actual)
    expected_value = _finite_float(f"oracle {name}", expected)
    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise RuntimeError(
            f"installed example {name} differs from its trusted numerical reference: "
            f"actual={actual_value!r}, expected={expected_value!r}, "
            f"absolute_tolerance={tolerance!r}, "
            f"absolute_error={abs(actual_value - expected_value)!r}"
        )
    return actual_value


def _trusted_numerical_oracle(
    oracle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    provenance = _require_keys(
        "oracle provenance",
        oracle["provenance"],
        {"method", "validated_environments"},
    )
    if provenance["method"] != (
        "Regression values were reproduced bit-for-bit in the listed environments. "
        "Independent accuracy values and Fourier cell checks are generated by "
        "scripts/generate_artifact_oracle.py, which does not import stableboundary."
    ):
        raise RuntimeError("artifact oracle has unrecognized numerical provenance")
    environments = provenance["validated_environments"]
    if not isinstance(environments, list) or not environments:
        raise RuntimeError("artifact oracle has no validated environments")

    independent = _require_keys(
        "independent oracle",
        oracle["independent_reference"],
        {
            "method",
            "orders",
            "accuracy_tolerances",
            "reference",
            "maximum_order_absolute_difference",
            "maximum_fourier_cell_absolute_difference",
        },
    )
    if independent["method"] != (
        "Independent SciPy S0 cell evaluation; 48/64 tensor Gauss-Legendre "
        "quadrature; monotone PCHIP marginal and conditional-CDF inversion"
    ) or independent["orders"] != [48, 64]:
        raise RuntimeError("artifact oracle changed its independent derivation")
    if (
        _finite_float(
            "oracle maximum order difference",
            independent["maximum_order_absolute_difference"],
        )
        > 2e-5
    ):
        raise RuntimeError("artifact oracle independent quadrature did not converge")
    if (
        _finite_float(
            "oracle maximum Fourier difference",
            independent["maximum_fourier_cell_absolute_difference"],
        )
        > 5e-11
    ):
        raise RuntimeError("artifact oracle independent Fourier cross-check failed")

    reference = _require_keys(
        "independent numerical reference",
        independent["reference"],
        {
            "order",
            "design",
            "log_normalizer",
            "posterior_mass",
            "parameters",
            "identification",
        },
    )
    if reference["order"] != 64:
        raise RuntimeError("artifact oracle changed its reference quadrature order")
    accuracy = _require_keys(
        "independent accuracy tolerances",
        independent["accuracy_tolerances"],
        {"log_normalizer", "posterior_mass", "parameters", "identification"},
    )
    tolerances = _require_keys(
        "oracle tolerances",
        oracle["tolerances"],
        {
            "main",
            "tau",
            "algebraic",
            "posterior_mass",
            "refinement_absolute",
            "refinement_noise_scale_upper",
            "refinement_noise_scale_evidence",
        },
    )
    return reference, accuracy, tolerances


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


def _require_keys(name: str, value: object, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError(f"installed example returned malformed {name}")
    return value


def _validate_parameter_summary(
    quantity: str,
    summary: object,
    *,
    lower_bound: float,
    upper_bound: float,
    interval_mass: float,
) -> dict[str, float]:
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
        if not lower_bound <= value <= upper_bound:
            raise RuntimeError(
                "installed example returned "
                f"{quantity} {label} outside its prior support"
            )
    if not lower <= median <= upper:
        raise RuntimeError(
            f"installed example returned unordered {quantity} interval/median"
        )
    if mass != interval_mass:
        raise RuntimeError(
            f"installed example returned non-common {quantity} credible mass"
        )
    return {
        "mean": mean,
        "median": median,
        "lower": lower,
        "upper": upper,
    }


def _validate_refinement(
    refinement: object,
    *,
    expected: object,
    tolerance_config: object,
) -> None:
    values = _require_keys(
        "refinement",
        refinement,
        {
            "tolerance",
            "common_grid_points",
            "joint_total_variation",
            "log_normalizer_change",
            "summary_changes",
            "predictive_tail",
            "converged",
        },
    )
    expected_values = _require_keys(
        "oracle refinement",
        expected,
        {
            "tolerance",
            "common_grid_points",
            "joint_total_variation",
            "log_normalizer_change",
            "summary_changes",
            "predictive_tail",
            "converged",
        },
    )
    configured_tolerances = _require_keys(
        "oracle tolerances",
        tolerance_config,
        {
            "main",
            "tau",
            "algebraic",
            "posterior_mass",
            "refinement_absolute",
            "refinement_noise_scale_upper",
            "refinement_noise_scale_evidence",
        },
    )
    if (
        values["converged"] is not True
        or values["converged"] is not expected_values["converged"]
    ):
        raise RuntimeError(
            "installed example did not retain passing refinement evidence"
        )
    tolerance = _finite_float("refinement tolerance", values["tolerance"])
    expected_tolerance = _finite_float(
        "oracle refinement tolerance", expected_values["tolerance"]
    )
    if tolerance != 0.002 or tolerance != expected_tolerance:
        raise RuntimeError("installed example changed the fixed refinement tolerance")
    common_grid_points = _strict_nonnegative_int(
        "refinement common grid points", values["common_grid_points"]
    )
    if (
        common_grid_points != 65
        or common_grid_points != expected_values["common_grid_points"]
    ):
        raise RuntimeError("installed example changed the fixed common grid")

    components: dict[str, float] = {
        "joint total variation": _finite_float(
            "refinement total variation", values["joint_total_variation"]
        ),
        "log-normalizer change": _finite_float(
            "refinement log-normalizer change", values["log_normalizer_change"]
        ),
    }
    expected_components: dict[str, float] = {
        "joint total variation": _finite_float(
            "oracle refinement total variation",
            expected_values["joint_total_variation"],
        ),
        "log-normalizer change": _finite_float(
            "oracle refinement log-normalizer change",
            expected_values["log_normalizer_change"],
        ),
    }
    summary_changes = _require_keys(
        "refinement summary changes", values["summary_changes"], QUANTITIES
    )
    expected_summary_changes = _require_keys(
        "oracle refinement summary changes",
        expected_values["summary_changes"],
        QUANTITIES,
    )
    summary_component_names = {
        "mean",
        "median",
        "interval_lower",
        "interval_upper",
    }
    for quantity, raw_changes in summary_changes.items():
        changes = _require_keys(
            f"{quantity} refinement changes",
            raw_changes,
            summary_component_names,
        )
        expected_changes = _require_keys(
            f"oracle {quantity} refinement changes",
            expected_summary_changes[quantity],
            summary_component_names,
        )
        for component_name, raw_value in changes.items():
            component_key = f"{quantity} {component_name}"
            components[component_key] = _finite_float(
                f"{quantity} {component_name} refinement", raw_value
            )
            expected_components[component_key] = _finite_float(
                f"oracle {quantity} {component_name} refinement",
                expected_changes[component_name],
            )

    predictive = _require_keys(
        "predictive-tail refinement",
        values["predictive_tail"],
        {"negative", "positive"},
    )
    expected_predictive = _require_keys(
        "oracle predictive-tail refinement",
        expected_values["predictive_tail"],
        {"negative", "positive"},
    )
    for side in ("negative", "positive"):
        components[f"predictive {side}"] = _finite_float(
            f"predictive {side} refinement", predictive[side]
        )
        expected_components[f"predictive {side}"] = _finite_float(
            f"oracle predictive {side} refinement", expected_predictive[side]
        )

    invalid = {
        name: value
        for name, value in components.items()
        if value < 0.0 or value > tolerance
    }
    if invalid:
        raise RuntimeError(
            f"installed example has refinement components above 0.002: {invalid!r}"
        )

    absolute_tolerance = _finite_float(
        "oracle refinement absolute tolerance",
        configured_tolerances["refinement_absolute"],
    )
    noise_scale_upper = _finite_float(
        "oracle refinement noise-scale upper bound",
        configured_tolerances["refinement_noise_scale_upper"],
    )
    if absolute_tolerance <= 0.0 or noise_scale_upper != 5e-14:
        raise RuntimeError("artifact oracle has invalid refinement tolerances")
    noise_evidence = _require_keys(
        "refinement noise-scale evidence",
        configured_tolerances["refinement_noise_scale_evidence"],
        {"ci_run_id", "ci_run_url", "scope", "observed_positive_envelopes"},
    )
    if (
        noise_evidence["ci_run_id"] != 32763920150
        or noise_evidence["ci_run_url"]
        != "https://github.com/moeketsims/stableboundary/actions/runs/32763920150"
        or noise_evidence["scope"]
        != (
            "Eight Linux and Darwin CI fingerprints plus the independently "
            "reproduced Windows reference; exact regression values remain in "
            "refinement and the envelopes are diagnostic."
        )
    ):
        raise RuntimeError("artifact oracle changed refinement evidence provenance")
    observed_envelopes = _require_keys(
        "observed refinement noise-scale envelopes",
        noise_evidence["observed_positive_envelopes"],
        set(REFINEMENT_NOISE_COMPONENTS),
    )
    for name, raw_envelope in observed_envelopes.items():
        envelope = _require_keys(
            f"observed {name} refinement envelope",
            raw_envelope,
            {"minimum", "maximum"},
        )
        minimum = _finite_float(
            f"observed {name} refinement minimum", envelope["minimum"]
        )
        maximum = _finite_float(
            f"observed {name} refinement maximum", envelope["maximum"]
        )
        expected_value = expected_components[name]
        if not 0.0 < minimum <= expected_value <= maximum <= noise_scale_upper:
            raise RuntimeError(
                f"artifact oracle has invalid observed envelope for {name}"
            )
    for affine, base in (("alpha mean", "h mean"), ("beta mean", "p mean")):
        if (
            components[affine] != components[base]
            or expected_components[affine] != expected_components[base]
        ):
            raise RuntimeError(
                "installed example violated exact refinement identity "
                f"{affine} == {base}"
            )
    for name, actual_value in components.items():
        expected_value = expected_components[name]
        if name in {"alpha mean", "beta mean"}:
            continue
        if name in REFINEMENT_NOISE_COMPONENTS:
            if not 0.0 <= actual_value <= noise_scale_upper:
                raise RuntimeError(
                    "installed example noise-scale refinement is outside its "
                    f"5e-14 bound for {name}: actual={actual_value!r}, "
                    f"expected={expected_value!r}, allowed_interval=[0.0, 5e-14]"
                )
            continue
        if expected_value <= 0.0 or actual_value <= 0.0:
            raise RuntimeError(
                f"installed example lost nonzero refinement evidence for {name}"
            )
        _reference_close(
            f"refinement {name}",
            actual_value,
            expected_value,
            tolerance=absolute_tolerance,
        )


def _simulation_failure(reason: str, fingerprint: dict[str, object]) -> RuntimeError:
    encoded = json.dumps(fingerprint, sort_keys=True, allow_nan=False)
    return RuntimeError(f"{reason}; observed simulation fingerprint: {encoded}")


def _refinement_regression_fingerprint(value: object) -> dict[str, object]:
    """Flatten all 28 numerical refinement components for one-line diagnostics."""
    refinement = value if isinstance(value, dict) else {}
    summary_changes = refinement.get("summary_changes")
    summaries = summary_changes if isinstance(summary_changes, dict) else {}
    predictive_tail = refinement.get("predictive_tail")
    predictive = predictive_tail if isinstance(predictive_tail, dict) else {}
    components: dict[str, object] = {
        "joint_total_variation": refinement.get("joint_total_variation"),
        "log_normalizer_change": refinement.get("log_normalizer_change"),
    }
    for quantity in sorted(QUANTITIES):
        raw_changes = summaries.get(quantity)
        changes = raw_changes if isinstance(raw_changes, dict) else {}
        for component in ("mean", "median", "interval_lower", "interval_upper"):
            components[f"summary.{quantity}.{component}"] = changes.get(component)
    for side in ("negative", "positive"):
        components[f"predictive.{side}"] = predictive.get(side)
    return {
        "contract": {
            "tolerance": refinement.get("tolerance"),
            "common_grid_points": refinement.get("common_grid_points"),
            "converged": refinement.get("converged"),
        },
        "component_count": len(components),
        "components": components,
    }


def _numerical_regression_fingerprint(
    primary: dict[str, Any],
    reflection: object,
    *,
    runtime_versions: dict[str, Any] | None,
) -> dict[str, object]:
    """Return every portable posterior-regression observation in compact form."""
    simulation = primary.get("simulation")
    simulation_values = simulation if isinstance(simulation, dict) else {}
    reflected = reflection if isinstance(reflection, dict) else {}
    inferred_runtime = {
        "python": simulation_values.get("python_version"),
        "platform_system": simulation_values.get("platform_system"),
        "platform_machine": simulation_values.get("platform_machine"),
        "numpy": simulation_values.get("numpy_version"),
        "scipy": simulation_values.get("scipy_version"),
    }
    runtime = inferred_runtime if runtime_versions is None else runtime_versions
    return {
        "runtime": runtime,
        "primary": {
            "schema_version": primary.get("schema_version"),
            "package_version": primary.get("package_version"),
            "status": primary.get("status"),
            "method": primary.get("method"),
            "parameterization": primary.get("parameterization"),
            "known_nuisance": primary.get("known_nuisance"),
            "counts": primary.get("counts"),
            "quadrature": primary.get("quadrature"),
            "parameters": primary.get("parameters"),
            "posterior_mass": primary.get("posterior_mass"),
            "identification": primary.get("identification"),
            "refinement": _refinement_regression_fingerprint(primary.get("refinement")),
            "warnings": primary.get("warnings"),
            "backend": primary.get("backend"),
            "simulation_provenance": {
                name: simulation_values.get(name)
                for name in (
                    "rng_algorithm",
                    "simulator_algorithm",
                    "platform_system",
                    "platform_machine",
                    "python_version",
                    "numpy_version",
                    "scipy_version",
                )
            },
        },
        "reflection": {
            name: reflected.get(name)
            for name in (
                "status",
                "method",
                "parameterization",
                "counts",
                "log_normalizer",
                "parameters",
                "identification",
                "warnings",
            )
        },
    }


def _numerical_regression_failure(
    reason: str,
    primary: dict[str, Any],
    reflection: object,
    *,
    runtime_versions: dict[str, Any] | None,
) -> RuntimeError:
    fingerprint = _numerical_regression_fingerprint(
        primary,
        reflection,
        runtime_versions=runtime_versions,
    )
    encoded = json.dumps(fingerprint, sort_keys=True, allow_nan=False)
    return RuntimeError(
        f"{reason}; observed numerical regression fingerprint: {encoded}"
    )


def _validate_example(
    payload: dict[str, Any],
    *,
    runtime_versions: dict[str, Any] | None = None,
) -> None:
    oracle = _oracle_document()
    independent_reference, independent_accuracy, tolerance_config = (
        _trusted_numerical_oracle(oracle)
    )
    main_tolerance = _finite_float("oracle main tolerance", tolerance_config["main"])
    tau_tolerance = _finite_float("oracle tau tolerance", tolerance_config["tau"])
    algebraic_tolerance = _finite_float(
        "oracle algebraic tolerance", tolerance_config["algebraic"]
    )
    posterior_mass_tolerance = _finite_float(
        "oracle posterior-mass tolerance", tolerance_config["posterior_mass"]
    )
    if (
        min(
            main_tolerance,
            tau_tolerance,
            algebraic_tolerance,
            posterior_mass_tolerance,
        )
        <= 0.0
    ):
        raise RuntimeError("artifact oracle has invalid numerical tolerances")
    expected_top_level = {
        "schema_version",
        "package_version",
        "status",
        "method",
        "parameterization",
        "known_nuisance",
        "seed",
        "truth",
        "inference_fixture",
        "simulation",
        "design",
        "prior",
        "counts",
        "quadrature",
        "parameters",
        "posterior_mass",
        "identification",
        "refinement",
        "backend",
        "warnings",
    }
    if set(payload) != expected_top_level:
        raise RuntimeError("installed example returned an incomplete schema")
    if (
        isinstance(payload["schema_version"], bool)
        or not isinstance(payload["schema_version"], int)
        or payload["schema_version"] != 1
    ):
        raise RuntimeError("installed example returned an unexpected schema version")
    if payload["package_version"] != PROJECT_VERSION:
        raise RuntimeError("installed example returned an unexpected package version")
    if payload.get("status") != "research_uncertified":
        raise RuntimeError("installed example returned an unexpected status")
    if payload.get("method") != "exact_finite_three_cell":
        raise RuntimeError("installed example returned an unexpected method")
    if payload.get("parameterization") != "S0":
        raise RuntimeError("installed example did not report S0")
    if (
        isinstance(payload["seed"], bool)
        or not isinstance(payload["seed"], int)
        or payload["seed"] != 20_260_824
    ):
        raise RuntimeError("installed example changed the fixed simulation seed")
    nuisance = _require_keys(
        "nuisance provenance",
        payload.get("known_nuisance"),
        {"loc", "scale", "mode", "provenance"},
    )
    if nuisance != oracle["known_nuisance"]:
        raise RuntimeError("installed example changed exact nuisance provenance")

    design = _require_keys(
        "design",
        payload.get("design"),
        {
            "n",
            "c",
            "r",
            "threshold",
            "formula_id",
            "formula_version",
            "critical_rate_relative_residual",
        },
    )
    if _strict_nonnegative_int("design sample size", design["n"]) != 5_000:
        raise RuntimeError("installed example changed the fixed sample size")
    c_value = _finite_float("design c", design["c"])
    r_value = _finite_float("design r", design["r"])
    threshold = _finite_float("design threshold", design["threshold"])
    if c_value != 1.0 or not 0.0 < r_value < 1.0 or threshold <= 0.0:
        raise RuntimeError("installed example returned an invalid fixed design")
    if (
        design["formula_id"] != "critical-rate-lambertw-loglog-threshold"
        or isinstance(design["formula_version"], bool)
        or not isinstance(design["formula_version"], int)
        or design["formula_version"] != 1
    ):
        raise RuntimeError("installed example changed the design formula")
    log_inverse_r = math.log(1.0 / r_value)
    expected_threshold = 2.0 * math.sqrt(log_inverse_r + 2.0 * math.log(log_inverse_r))
    if not math.isclose(threshold, expected_threshold, rel_tol=0.0, abs_tol=1e-14):
        raise RuntimeError("installed example returned an inconsistent threshold")
    expected_residual = abs(5_000 * r_value / log_inverse_r - 8.0 * c_value) / (
        8.0 * c_value
    )
    reported_residual = _finite_float(
        "critical-rate residual", design["critical_rate_relative_residual"]
    )
    if reported_residual > 1e-12 or not math.isclose(
        reported_residual,
        expected_residual,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("installed example returned an inconsistent design residual")
    if design != oracle["design"]:
        raise RuntimeError("installed example changed the trusted design reference")
    independent_design = _require_keys(
        "independent design reference",
        independent_reference["design"],
        {"r", "threshold"},
    )
    _reference_close(
        "design r against independent quadrature",
        r_value,
        independent_design["r"],
        tolerance=main_tolerance,
    )
    _reference_close(
        "design threshold against independent quadrature",
        threshold,
        independent_design["threshold"],
        tolerance=main_tolerance,
    )

    truth = _require_keys(
        "simulation truth",
        payload.get("truth"),
        {"alpha", "beta", "loc", "scale"},
    )
    truth_alpha = _finite_float("truth alpha", truth["alpha"])
    if not math.isclose(
        truth_alpha,
        2.0 - 1.5 * r_value,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(
            "installed example did not derive truth alpha from its design"
        )
    if (
        _finite_float("truth beta", truth["beta"]) != 0.35
        or _finite_float("truth location", truth["loc"]) != 0.0
        or _finite_float("truth scale", truth["scale"]) != 1.0
    ):
        raise RuntimeError("installed example changed the fixed simulation truth")

    fixture = _require_keys(
        "inference fixture",
        payload.get("inference_fixture"),
        {"construction", "dtype", "nbytes", "sha256"},
    )
    independently_computed_fixture = _independent_fixture_identity(threshold)
    if (
        fixture != oracle["fixture"]
        or independently_computed_fixture != oracle["fixture"]
    ):
        raise RuntimeError("installed example changed the fixed inference fixture")

    simulation = _require_keys(
        "simulation evidence",
        payload.get("simulation"),
        {
            "dtype",
            "rng_algorithm",
            "simulator_algorithm",
            "platform_system",
            "platform_machine",
            "python_version",
            "numpy_version",
            "scipy_version",
            "sample_sha256",
            "quantized_sample_sha256",
            "counts",
            "minimum",
            "maximum",
            "diagnostics",
        },
    )
    contract = _require_keys(
        "simulation contract",
        oracle["simulation_contract"],
        {
            "sample_size",
            "seed",
            "truth_rule",
            "rng_algorithm",
            "simulator_algorithm",
            "canonical_dtype",
            "raw_hash_policy",
            "quantization_algorithm",
            "quantization_steps",
            "diagnostic_absolute_tolerance",
            "approval_evidence",
            "approved_environments",
        },
    )
    if (
        contract["raw_hash_policy"] != "diagnostic_only"
        or contract["quantization_algorithm"] != "rint(x/step)->canonical-<i8:v1"
        or contract["quantization_steps"]
        != ["1e-10", "1e-11", "1e-12", "1e-13", "1e-14"]
        or contract["diagnostic_absolute_tolerance"] != 1e-12
    ):
        raise RuntimeError("artifact oracle changed the sample quantization contract")
    approval_evidence = _require_keys(
        "simulation approval evidence",
        contract["approval_evidence"],
        {"ci_run_id", "ci_run_url", "normative_selection"},
    )
    if approval_evidence != {
        "ci_run_id": 32761069162,
        "ci_run_url": (
            "https://github.com/moeketsims/stableboundary/actions/runs/32761069162"
        ),
        "normative_selection": (
            "1e-12 was the finest full-sample quantization grid identical across "
            "all observed Windows, Linux, and Darwin jobs; 1e-13 and 1e-14 "
            "diverged and remain diagnostic only."
        ),
    }:
        raise RuntimeError("artifact oracle changed the simulation approval evidence")
    quantization_steps = contract["quantization_steps"]
    quantized_hashes = _require_keys(
        "quantized simulation hashes",
        simulation["quantized_sample_sha256"],
        set(quantization_steps),
    )
    for step, digest in quantized_hashes.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                f"installed simulation returned invalid quantized hash for {step}"
            )
    diagnostics = _require_keys(
        "simulation diagnostics",
        simulation["diagnostics"],
        {"mean", "standard_deviation", "q01", "q05", "median", "q95", "q99"},
    )
    validated_diagnostics = {
        name: _finite_float(f"simulation diagnostic {name}", value)
        for name, value in diagnostics.items()
    }
    simulation_counts = _require_keys(
        "simulation cell counts",
        simulation["counts"],
        {"n_minus", "n_zero", "n_plus"},
    )
    validated_simulation_counts = {
        name: _strict_nonnegative_int(f"simulation count {name}", value)
        for name, value in simulation_counts.items()
    }
    minimum = _finite_float("simulation minimum", simulation["minimum"])
    maximum = _finite_float("simulation maximum", simulation["maximum"])
    text_fields = (
        "dtype",
        "rng_algorithm",
        "simulator_algorithm",
        "platform_system",
        "platform_machine",
        "python_version",
        "numpy_version",
        "scipy_version",
        "sample_sha256",
    )
    for name in text_fields:
        if not isinstance(simulation[name], str) or not simulation[name].strip():
            raise RuntimeError(f"installed simulation returned invalid {name}")
    if re.fullmatch(r"[0-9a-f]{64}", simulation["sample_sha256"]) is None:
        raise RuntimeError("installed simulation returned invalid raw sample hash")
    fingerprint: dict[str, object] = {
        "seed": payload["seed"],
        "truth": truth,
        "dtype": simulation["dtype"],
        "rng_algorithm": simulation["rng_algorithm"],
        "simulator_algorithm": simulation["simulator_algorithm"],
        "sample_sha256": simulation["sample_sha256"],
        "quantized_sample_sha256": quantized_hashes,
        "counts": validated_simulation_counts,
        "minimum": minimum,
        "maximum": maximum,
        "diagnostics": validated_diagnostics,
        "platform": {
            "system": simulation["platform_system"],
            "machine": simulation["platform_machine"],
        },
        "versions": {
            "python": simulation["python_version"],
            "numpy": simulation["numpy_version"],
            "scipy": simulation["scipy_version"],
        },
    }
    if (
        contract["sample_size"] != 5_000
        or contract["seed"] != payload["seed"]
        or contract["truth_rule"] != "alpha=2-design.r*1.5,beta=0.35,loc=0,scale=1"
        or simulation["dtype"] != contract["canonical_dtype"]
        or simulation["rng_algorithm"] != contract["rng_algorithm"]
        or simulation["simulator_algorithm"] != contract["simulator_algorithm"]
    ):
        raise _simulation_failure(
            "installed simulation changed its fixed algorithm contract", fingerprint
        )
    if sum(validated_simulation_counts.values()) != contract["sample_size"]:
        raise _simulation_failure(
            "installed simulation counts have the wrong sample size", fingerprint
        )
    platform_system = simulation["platform_system"]
    platform_machine = simulation["platform_machine"]
    python_version = simulation["python_version"]
    numpy_version = simulation["numpy_version"]
    scipy_version = simulation["scipy_version"]
    if runtime_versions is not None and (
        runtime_versions.get("python") != python_version
        or runtime_versions.get("numpy") != numpy_version
        or runtime_versions.get("scipy") != scipy_version
        or runtime_versions.get("platform_system") != platform_system
        or runtime_versions.get("platform_machine") != platform_machine
    ):
        raise _simulation_failure(
            "installed simulation runtime contradicts independent imports",
            fingerprint,
        )
    environment_key = (
        f"system={platform_system}|machine={platform_machine}|"
        f"numpy={numpy_version}|scipy={scipy_version}"
    )
    approved = contract["approved_environments"]
    if not isinstance(approved, dict) or environment_key not in approved:
        raise _simulation_failure(
            f"installed simulation environment is not approved: {environment_key}",
            fingerprint,
        )
    expected_simulation = approved[environment_key]
    expected_simulation_values = _require_keys(
        "approved simulation evidence",
        expected_simulation,
        {
            "platform_system",
            "platform_machine",
            "observed_python_versions",
            "dtype",
            "observed_raw_sha256",
            "normative_quantization_step",
            "quantized_sample_sha256",
            "observed_non_normative_quantized_sha256",
            "observed_diagnostics",
            "counts",
            "minimum",
            "maximum",
        },
    )
    observed_raw_hashes = expected_simulation_values["observed_raw_sha256"]
    if (
        not isinstance(observed_raw_hashes, list)
        or not observed_raw_hashes
        or any(
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in observed_raw_hashes
        )
    ):
        raise RuntimeError("artifact oracle has invalid observed raw sample hashes")
    expected_quantized_hashes = _require_keys(
        "approved quantized simulation hashes",
        expected_simulation_values["quantized_sample_sha256"],
        set(quantization_steps),
    )
    for step, digest in expected_quantized_hashes.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                f"artifact oracle has invalid approved quantized hash for {step}"
            )
    non_normative_hashes = _require_keys(
        "observed non-normative simulation hashes",
        expected_simulation_values["observed_non_normative_quantized_sha256"],
        {"1e-13", "1e-14"},
    )
    for step, digests in non_normative_hashes.items():
        if (
            not isinstance(digests, list)
            or not digests
            or len(digests) != len(set(digests))
            or any(
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in digests
            )
            or expected_quantized_hashes[step] not in digests
        ):
            raise RuntimeError(
                f"artifact oracle has invalid non-normative hashes for {step}"
            )
    normative_step = expected_simulation_values["normative_quantization_step"]
    if normative_step != "1e-12":
        raise RuntimeError("artifact oracle changed the approved normative grid")
    observed_diagnostics = _require_keys(
        "approved observed simulation diagnostics",
        expected_simulation_values["observed_diagnostics"],
        set(validated_diagnostics),
    )
    for name, value in observed_diagnostics.items():
        _finite_float(f"oracle observed simulation diagnostic {name}", value)
    observed_python_versions = expected_simulation_values["observed_python_versions"]
    if (
        not isinstance(observed_python_versions, list)
        or not observed_python_versions
        or observed_python_versions != sorted(set(observed_python_versions))
        or any(
            not isinstance(version, str) or not version
            for version in observed_python_versions
        )
    ):
        raise RuntimeError("artifact oracle has invalid observed Python versions")
    mismatches = {
        "platform_system": (
            platform_system,
            expected_simulation_values["platform_system"],
        ),
        "platform_machine": (
            platform_machine,
            expected_simulation_values["platform_machine"],
        ),
        "dtype": (simulation["dtype"], expected_simulation_values["dtype"]),
        f"quantized_sample_sha256[{normative_step}]": (
            quantized_hashes[normative_step],
            expected_quantized_hashes[normative_step],
        ),
        "counts": (
            validated_simulation_counts,
            expected_simulation_values["counts"],
        ),
        "minimum": (minimum, expected_simulation_values["minimum"]),
        "maximum": (maximum, expected_simulation_values["maximum"]),
    }
    changed = sorted(
        name for name, (actual, expected) in mismatches.items() if actual != expected
    )
    diagnostic_tolerance = _finite_float(
        "oracle simulation diagnostic tolerance",
        contract["diagnostic_absolute_tolerance"],
    )
    changed.extend(
        f"diagnostics[{name}]"
        for name, actual in validated_diagnostics.items()
        if not math.isclose(
            actual,
            _finite_float(
                f"oracle observed simulation diagnostic {name}",
                observed_diagnostics[name],
            ),
            rel_tol=0.0,
            abs_tol=diagnostic_tolerance,
        )
    )
    changed.sort()
    if changed:
        raise _simulation_failure(
            "installed simulation differs from its approved reference in "
            + ", ".join(changed),
            fingerprint,
        )

    prior = _require_keys(
        "prior",
        payload.get("prior"),
        {"family", "h_min", "h_max", "p_min", "p_max"},
    )
    if prior["family"] != "compact_uniform_rectangle":
        raise RuntimeError("installed example changed the prior family")
    bounds = {
        name: _finite_float(f"prior {name}", prior[name])
        for name in ("h_min", "h_max", "p_min", "p_max")
    }
    if bounds != {"h_min": 0.25, "h_max": 4.0, "p_min": 0.05, "p_max": 0.95}:
        raise RuntimeError("installed example changed the fixed compact prior")
    if prior != oracle["prior"]:
        raise RuntimeError("installed example changed the trusted prior reference")

    counts = payload.get("counts")
    count_names = {"n_minus", "n_zero", "n_plus", "n", "threshold"}
    if not isinstance(counts, dict) or set(counts) != count_names:
        raise RuntimeError("installed example returned invalid cell counts")
    validated_counts = {
        name: _strict_nonnegative_int(f"cell count {name}", counts[name])
        for name in ("n_minus", "n_zero", "n_plus", "n")
    }
    if (
        validated_counts["n"] == 0
        or sum(validated_counts[name] for name in ("n_minus", "n_zero", "n_plus"))
        != validated_counts["n"]
    ):
        raise RuntimeError("installed example returned invalid cell counts")
    if (
        validated_counts["n_minus"],
        validated_counts["n_zero"],
        validated_counts["n_plus"],
    ) != (1, 4_996, 3):
        raise RuntimeError("installed example changed the fixed-seed cell counts")
    if validated_counts["n"] != 5_000 or not math.isclose(
        _finite_float("cell threshold", counts["threshold"]),
        threshold,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("installed example returned counts for the wrong design")
    if counts != oracle["counts"]:
        raise RuntimeError("installed example changed the trusted count witness")

    quadrature = _require_keys(
        "quadrature",
        payload.get("quadrature"),
        {"base_nodes", "refined_nodes", "interval_mass", "log_normalizer"},
    )
    if (
        _strict_nonnegative_int("base quadrature nodes", quadrature["base_nodes"]) != 20
        or _strict_nonnegative_int(
            "refined quadrature nodes", quadrature["refined_nodes"]
        )
        != 32
    ):
        raise RuntimeError("installed example changed the fixed quadrature orders")
    interval_mass = _finite_float(
        "quadrature interval mass", quadrature["interval_mass"]
    )
    if interval_mass != 0.9:
        raise RuntimeError("installed example changed the common credible mass")
    log_normalizer = _reference_close(
        "quadrature log normalizer",
        quadrature["log_normalizer"],
        oracle["quadrature"]["log_normalizer"],
        tolerance=main_tolerance,
    )
    _reference_close(
        "quadrature log normalizer against independent quadrature",
        log_normalizer,
        independent_reference["log_normalizer"],
        tolerance=_finite_float(
            "independent log-normalizer tolerance",
            independent_accuracy["log_normalizer"],
        ),
    )
    for name in ("base_nodes", "refined_nodes", "interval_mass"):
        if quadrature[name] != oracle["quadrature"][name]:
            raise RuntimeError("installed example changed trusted quadrature settings")

    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != QUANTITIES:
        raise RuntimeError("installed example did not return all six summaries")
    parameter_bounds = {
        "h": (bounds["h_min"], bounds["h_max"]),
        "p": (bounds["p_min"], bounds["p_max"]),
        "alpha": (
            2.0 - r_value * bounds["h_max"],
            2.0 - r_value * bounds["h_min"],
        ),
        "beta": (2.0 * bounds["p_min"] - 1.0, 2.0 * bounds["p_max"] - 1.0),
        "tau_plus": (
            r_value * bounds["h_min"] * bounds["p_min"],
            r_value * bounds["h_max"] * bounds["p_max"],
        ),
        "tau_minus": (
            r_value * bounds["h_min"] * (1.0 - bounds["p_max"]),
            r_value * bounds["h_max"] * (1.0 - bounds["p_min"]),
        ),
    }
    validated_parameters = {
        quantity: _validate_parameter_summary(
            quantity,
            summary,
            lower_bound=parameter_bounds[quantity][0],
            upper_bound=parameter_bounds[quantity][1],
            interval_mass=interval_mass,
        )
        for quantity, summary in parameters.items()
    }
    regression_parameters = _require_keys(
        "oracle parameter references", oracle["parameters"], QUANTITIES
    )
    independent_parameters = _require_keys(
        "independent parameter references",
        independent_reference["parameters"],
        QUANTITIES,
    )
    independent_parameter_tolerances = _require_keys(
        "independent parameter tolerances",
        independent_accuracy["parameters"],
        QUANTITIES,
    )
    for quantity, actual_summary in validated_parameters.items():
        expected_summary = _require_keys(
            f"oracle {quantity} reference",
            regression_parameters[quantity],
            {"mean", "median", "lower", "upper"},
        )
        independent_summary = _require_keys(
            f"independent {quantity} reference",
            independent_parameters[quantity],
            {"mean", "median", "lower", "upper"},
        )
        accuracy_summary = _require_keys(
            f"independent {quantity} tolerances",
            independent_parameter_tolerances[quantity],
            {"mean", "median", "lower", "upper"},
        )
        regression_tolerance = (
            tau_tolerance if quantity in {"tau_plus", "tau_minus"} else main_tolerance
        )
        for component, actual_value in actual_summary.items():
            _reference_close(
                f"{quantity} {component}",
                actual_value,
                expected_summary[component],
                tolerance=regression_tolerance,
            )
            _reference_close(
                f"{quantity} {component} against independent quadrature",
                actual_value,
                independent_summary[component],
                tolerance=_finite_float(
                    f"independent {quantity} {component} tolerance",
                    accuracy_summary[component],
                ),
            )
    h_summary = validated_parameters["h"]
    alpha_summary = validated_parameters["alpha"]
    for alpha_name, h_name in (
        ("mean", "mean"),
        ("median", "median"),
        ("lower", "upper"),
        ("upper", "lower"),
    ):
        if not math.isclose(
            alpha_summary[alpha_name],
            2.0 - r_value * h_summary[h_name],
            rel_tol=0.0,
            abs_tol=algebraic_tolerance,
        ):
            raise RuntimeError("installed alpha and h summaries are inconsistent")
    p_summary = validated_parameters["p"]
    beta_summary = validated_parameters["beta"]
    for name in ("mean", "median", "lower", "upper"):
        if not math.isclose(
            beta_summary[name],
            2.0 * p_summary[name] - 1.0,
            rel_tol=0.0,
            abs_tol=algebraic_tolerance,
        ):
            raise RuntimeError("installed beta and p summaries are inconsistent")
    if not math.isclose(
        validated_parameters["tau_plus"]["mean"]
        + validated_parameters["tau_minus"]["mean"],
        r_value * h_summary["mean"],
        rel_tol=0.0,
        abs_tol=algebraic_tolerance,
    ):
        raise RuntimeError("installed signed-gap and h means are inconsistent")

    mass = _reference_close(
        "posterior mass",
        payload.get("posterior_mass"),
        oracle["posterior_mass"],
        tolerance=posterior_mass_tolerance,
    )
    if abs(mass - 1.0) > posterior_mass_tolerance:
        raise RuntimeError("installed example posterior mass is not normalized")
    _reference_close(
        "posterior mass against independent quadrature",
        mass,
        independent_reference["posterior_mass"],
        tolerance=_finite_float(
            "independent posterior-mass tolerance",
            independent_accuracy["posterior_mass"],
        ),
    )

    identification = _require_keys(
        "identification",
        payload.get("identification"),
        {
            "evidence_status",
            "precision_status",
            "p_kl_divergence",
            "p_interval_width_contraction",
        },
    )
    expected_evidence = (
        "two_sided_evidence"
        if validated_counts["n_minus"] and validated_counts["n_plus"]
        else "one_sided_evidence"
        if validated_counts["n_minus"] or validated_counts["n_plus"]
        else "prior_dominated"
    )
    expected_precision = (
        "unidentified" if expected_evidence == "prior_dominated" else "not_assessed"
    )
    if (
        identification["evidence_status"] != expected_evidence
        or identification["precision_status"] != expected_precision
    ):
        raise RuntimeError(
            "installed example returned inconsistent identification labels"
        )
    oracle_identification = oracle["identification"]
    if not isinstance(oracle_identification, dict) or (
        identification["evidence_status"]
        != oracle_identification.get("evidence_status")
        or identification["precision_status"]
        != oracle_identification.get("precision_status")
    ):
        raise RuntimeError("installed example changed trusted identification labels")
    p_kl = _finite_float("p KL divergence", identification["p_kl_divergence"])
    contraction = _finite_float(
        "p interval contraction", identification["p_interval_width_contraction"]
    )
    if p_kl < 0.0 or not 0.0 <= contraction <= 1.0:
        raise RuntimeError("installed example returned invalid identification metrics")
    algebraic_contraction = 1.0 - (p_summary["upper"] - p_summary["lower"]) / (
        interval_mass * (bounds["p_max"] - bounds["p_min"])
    )
    if not math.isclose(
        contraction,
        algebraic_contraction,
        rel_tol=0.0,
        abs_tol=algebraic_tolerance,
    ):
        raise RuntimeError(
            "installed p interval contraction is inconsistent with its interval"
        )
    regression_identification = _require_keys(
        "oracle identification reference",
        oracle_identification,
        {
            "evidence_status",
            "precision_status",
            "p_kl_divergence",
            "p_interval_width_contraction",
        },
    )
    independent_identification = _require_keys(
        "independent identification reference",
        independent_reference["identification"],
        {"p_kl_divergence", "p_interval_width_contraction"},
    )
    independent_identification_tolerances = _require_keys(
        "independent identification tolerances",
        independent_accuracy["identification"],
        {"p_kl_divergence", "p_interval_width_contraction"},
    )
    for name, actual_value in (
        ("p_kl_divergence", p_kl),
        ("p_interval_width_contraction", contraction),
    ):
        _reference_close(
            f"identification {name}",
            actual_value,
            regression_identification[name],
            tolerance=main_tolerance,
        )
        _reference_close(
            f"identification {name} against independent quadrature",
            actual_value,
            independent_identification[name],
            tolerance=_finite_float(
                f"independent identification {name} tolerance",
                independent_identification_tolerances[name],
            ),
        )

    warnings = payload.get("warnings")
    if warnings != oracle["warnings"]:
        raise RuntimeError("installed example changed exact warning codes or text")

    backend = _require_keys(
        "backend provenance",
        payload.get("backend"),
        {
            "method",
            "tolerance",
            "origin",
            "parameterization",
            "library",
            "library_version",
            "effective_settings",
        },
    )
    if (
        backend["parameterization"] != "S0"
        or backend["origin"] != "canonical_scipy_s0"
        or backend["method"] != "scipy-piecewise-s0-direct-log-tails"
        or backend["library"] != "scipy"
    ):
        raise RuntimeError("installed backend provenance is not canonical SciPy S0")
    if backend["library_version"] != scipy_version:
        raise RuntimeError("installed backend library version contradicts simulation")
    if runtime_versions is not None and backend[
        "library_version"
    ] != runtime_versions.get("scipy"):
        raise RuntimeError(
            "installed backend library version contradicts independent import"
        )
    for name in ("method", "origin", "library", "library_version"):
        if not isinstance(backend[name], str) or not backend[name].strip():
            raise RuntimeError(
                "installed example returned incomplete backend provenance"
            )
    backend_tolerance = _finite_float("backend tolerance", backend["tolerance"])
    if backend_tolerance <= 0.0:
        raise RuntimeError("installed example returned invalid backend tolerance")
    effective_settings = _require_keys(
        "backend settings",
        backend["effective_settings"],
        {
            "parameterization",
            "pdf_default_method",
            "cdf_default_method",
            "quad_eps",
            "piecewise_x_tol_near_zeta",
            "piecewise_alpha_tol_near_one",
            "pdf_fft_min_points_threshold",
            "pdf_fft_grid_spacing",
            "pdf_fft_n_points_two_power",
            "pdf_fft_interpolation_level",
            "pdf_fft_interpolation_degree",
        },
    )
    expected_effective_settings: dict[str, object] = {
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
    }
    if (
        effective_settings != expected_effective_settings
        or backend_tolerance != expected_effective_settings["quad_eps"]
    ):
        raise RuntimeError("installed backend settings contradict its provenance")

    _validate_refinement(
        payload.get("refinement"),
        expected=oracle["refinement"],
        tolerance_config=tolerance_config,
    )


def _assert_science_payload_parity(
    public_payload: dict[str, Any], independent_payload: dict[str, Any]
) -> None:
    if public_payload != independent_payload:
        raise RuntimeError(
            "public example differs from the independently executed installed estimator"
        )


def _summary_components(name: str, value: object) -> dict[str, float]:
    summary = _require_keys(
        f"{name} summary", value, {"mean", "median", "credible_interval"}
    )
    interval = _require_keys(
        f"{name} credible interval",
        summary["credible_interval"],
        {"lower", "upper", "mass"},
    )
    if _finite_float(f"{name} credible mass", interval["mass"]) != 0.9:
        raise RuntimeError(f"installed reflection changed {name} credible mass")
    return {
        "mean": _finite_float(f"{name} mean", summary["mean"]),
        "median": _finite_float(f"{name} median", summary["median"]),
        "lower": _finite_float(f"{name} lower", interval["lower"]),
        "upper": _finite_float(f"{name} upper", interval["upper"]),
    }


def _validate_reflection_probe(primary: dict[str, Any], reflection: object) -> None:
    """Require the installed estimator to respect exact S0 reflection symmetry."""
    values = _require_keys(
        "reflected installed estimator",
        reflection,
        {
            "status",
            "method",
            "parameterization",
            "counts",
            "log_normalizer",
            "parameters",
            "identification",
            "warnings",
        },
    )
    if (
        values["status"] != primary["status"]
        or values["method"] != primary["method"]
        or values["parameterization"] != "S0"
    ):
        raise RuntimeError("installed estimator changed method under reflection")
    reflected_counts = _require_keys(
        "reflected cell counts",
        values["counts"],
        {"n_minus", "n_zero", "n_plus", "n", "threshold"},
    )
    for name in ("n_minus", "n_zero", "n_plus", "n"):
        _strict_nonnegative_int(f"reflected count {name}", reflected_counts[name])
    primary_counts = primary["counts"]
    expected_counts = {
        "n_minus": primary_counts["n_plus"],
        "n_zero": primary_counts["n_zero"],
        "n_plus": primary_counts["n_minus"],
        "n": primary_counts["n"],
        "threshold": primary_counts["threshold"],
    }
    if reflected_counts != expected_counts:
        raise RuntimeError("installed estimator returned wrong reflected cell counts")

    oracle = _oracle_document()
    _, _, tolerances = _trusted_numerical_oracle(oracle)
    main_tolerance = _finite_float("oracle main tolerance", tolerances["main"])
    tau_tolerance = _finite_float("oracle tau tolerance", tolerances["tau"])
    primary_quadrature = primary["quadrature"]
    _reference_close(
        "reflected log normalizer",
        values["log_normalizer"],
        primary_quadrature["log_normalizer"],
        tolerance=main_tolerance,
    )

    primary_parameters = _require_keys(
        "primary parameter summaries", primary["parameters"], QUANTITIES
    )
    reflected_parameters = _require_keys(
        "reflected parameter summaries", values["parameters"], QUANTITIES
    )
    primary_summaries = {
        name: _summary_components(f"primary {name}", summary)
        for name, summary in primary_parameters.items()
    }
    reflected_summaries = {
        name: _summary_components(f"reflected {name}", summary)
        for name, summary in reflected_parameters.items()
    }
    expected_sources = {
        "h": ("h", 1.0, False),
        "alpha": ("alpha", 1.0, False),
        "p": ("p", 1.0, True),
        "beta": ("beta", -1.0, True),
        "tau_plus": ("tau_minus", 1.0, False),
        "tau_minus": ("tau_plus", 1.0, False),
    }
    for quantity, (source_quantity, multiplier, complement) in expected_sources.items():
        source = primary_summaries[source_quantity]
        expected_summary = dict(source)
        if complement:
            if quantity == "p":
                expected_summary = {
                    "mean": 1.0 - source["mean"],
                    "median": 1.0 - source["median"],
                    "lower": 1.0 - source["upper"],
                    "upper": 1.0 - source["lower"],
                }
            else:
                expected_summary = {
                    "mean": multiplier * source["mean"],
                    "median": multiplier * source["median"],
                    "lower": multiplier * source["upper"],
                    "upper": multiplier * source["lower"],
                }
        comparison_tolerance = (
            tau_tolerance if quantity in {"tau_plus", "tau_minus"} else main_tolerance
        )
        for component, actual_value in reflected_summaries[quantity].items():
            _reference_close(
                f"reflected {quantity} {component}",
                actual_value,
                expected_summary[component],
                tolerance=comparison_tolerance,
            )

    primary_identification = _require_keys(
        "primary identification",
        primary["identification"],
        {
            "evidence_status",
            "precision_status",
            "p_kl_divergence",
            "p_interval_width_contraction",
        },
    )
    reflected_identification = _require_keys(
        "reflected identification",
        values["identification"],
        set(primary_identification),
    )
    for name in ("evidence_status", "precision_status"):
        if reflected_identification[name] != primary_identification[name]:
            raise RuntimeError(
                "installed estimator changed identification on reflection"
            )
    for name in ("p_kl_divergence", "p_interval_width_contraction"):
        _reference_close(
            f"reflected {name}",
            reflected_identification[name],
            primary_identification[name],
            tolerance=main_tolerance,
        )
    if values["warnings"] != oracle["warnings"]:
        raise RuntimeError("installed estimator changed warnings on reflection")


def _validate_science_probe(
    primary: dict[str, Any],
    reflection: object,
    *,
    runtime_versions: dict[str, Any] | None,
) -> None:
    """Validate both fits and retain a complete observation on any failure."""
    try:
        _validate_example(primary, runtime_versions=runtime_versions)
        _validate_reflection_probe(primary, reflection)
    except RuntimeError as error:
        raise _numerical_regression_failure(
            str(error),
            primary,
            reflection,
            runtime_versions=runtime_versions,
        ) from error


def _exercise_archive(
    source_artifact: Path,
    *,
    inspection: ArchiveInspection,
) -> dict[str, Any]:
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
        _prepare_environment(python, cwd=work)

        if source_artifact.name == EXPECTED_SDIST:
            with _artifact_snapshot(
                source_artifact, expected_digest=inspection.sdist_sha256
            ) as sdist_snapshot:
                wheel_source, wheel_inspection = _build_inspected_sdist_wheel(
                    python, sdist_snapshot, inspection, cwd=work
                )
        elif source_artifact.name == EXPECTED_WHEEL:
            wheel_source = source_artifact
            wheel_inspection = inspection
        else:
            raise RuntimeError(f"unsupported source artifact: {source_artifact.name}")

        before = _tree_inventory(environment)
        with _artifact_snapshot(
            wheel_source, expected_digest=wheel_inspection.wheel_sha256
        ) as wheel_snapshot:
            _install_verified_wheel(python, wheel_snapshot, cwd=work)
            installed, probe = _check_prove_and_probe_installed_runtime(
                python,
                cwd=work,
                source_artifact=source_artifact,
                wheel_snapshot=wheel_snapshot,
                environment=environment,
                inspection=wheel_inspection,
                before=before,
            )
            versions = _validate_installed_runtime(
                probe, installed=installed, artifact=wheel_snapshot.path
            )
            copied_example = work / EXAMPLE.name
            shutil.copy2(EXAMPLE, copied_example)
            decoded = json.loads(
                _run(
                    [str(python), "-I", str(copied_example)],
                    cwd=work,
                    stage=f"public example for {source_artifact.name}",
                    timeout_seconds=EXAMPLE_TIMEOUT_SECONDS,
                    capture=True,
                )
            )
            if not isinstance(decoded, dict):
                raise RuntimeError("installed example did not return a JSON object")
            payload: dict[str, Any] = decoded
            science_probe = _installed_science_probe(
                python,
                cwd=work,
                artifact=source_artifact,
                import_roots=(installed.site_packages,),
            )
            science_values = _require_keys(
                "independent installed science probe",
                science_probe,
                {"primary", "reflection"},
            )
            independent_payload = science_values["primary"]
            if not isinstance(independent_payload, dict):
                raise RuntimeError(
                    "independent installed science probe omitted its payload"
                )
            _assert_science_payload_parity(payload, independent_payload)
            _validate_science_probe(
                payload,
                science_values["reflection"],
                runtime_versions=versions,
            )
            print(
                json.dumps(
                    {
                        "artifact": source_artifact.name,
                        "installed_wheel": wheel_snapshot.source_name,
                        "origin": str(installed.import_origin),
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
    inspection = _inspect_archives(wheel, sdist)
    wheel_payload = _exercise_archive(wheel, inspection=inspection)
    sdist_payload = _exercise_archive(sdist, inspection=inspection)
    if wheel_payload != sdist_payload:
        raise RuntimeError(
            "wheel and sdist executions produced different scientific evidence"
        )


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
