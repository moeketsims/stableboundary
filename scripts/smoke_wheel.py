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


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Trusted byte identity for one source or installed package file."""

    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArchiveInspection:
    """Source identities retained after exact archive inspection."""

    package_files: dict[str, FileIdentity]
    wheel_sha256: str
    sdist_sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    """Private artifact copy authenticated before every execution stage."""

    source_name: str
    path: Path
    size: int
    sha256: str


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
    destination.write_bytes(content)
    destination.chmod(stat.S_IREAD)
    return ArtifactSnapshot(
        source_name=source.name,
        path=destination,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
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


def _install_archive(python: Path, artifact: Path, *, cwd: Path) -> None:
    """Install runtime dependencies first, then only the selected local artifact."""
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "numpy>=2.2",
            "scipy>=1.18,<1.19",
        ],
        cwd=cwd,
        stage=f"runtime dependency installation for {artifact.name}",
        timeout_seconds=INSTALL_TIMEOUT_SECONDS,
    )
    _run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import importlib.util; "
                "assert importlib.util.find_spec('stableboundary') is None"
            ),
        ],
        cwd=cwd,
        stage=f"pre-install substitution check for {artifact.name}",
        timeout_seconds=IMPORT_TIMEOUT_SECONDS,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            str(artifact),
        ],
        cwd=cwd,
        stage=f"installation of {artifact.name}",
        timeout_seconds=INSTALL_TIMEOUT_SECONDS,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "check",
            "--disable-pip-version-check",
        ],
        cwd=cwd,
        stage=f"dependency consistency check for {artifact.name}",
        timeout_seconds=IMPORT_TIMEOUT_SECONDS,
    )


def _validate_installed_distribution(
    probe: object,
    *,
    artifact: Path,
    environment: Path,
    expected_digest: str,
    package_files: dict[str, FileIdentity],
) -> Path:
    values = _require_keys(
        "installed distribution probe",
        probe,
        {
            "import_origin",
            "metadata_version",
            "package_version",
            "versions",
            "package_files",
            "direct_url",
        },
    )
    if (
        values["metadata_version"] != PROJECT_VERSION
        or values["package_version"] != PROJECT_VERSION
    ):
        raise RuntimeError(f"installed metadata version does not match {artifact.name}")
    origin_value = values["import_origin"]
    if not isinstance(origin_value, str) or not origin_value:
        raise RuntimeError("installed distribution returned an invalid import origin")
    origin = Path(origin_value).resolve()
    if not origin.is_relative_to(environment) or origin.is_relative_to(REPOSITORY):
        raise RuntimeError(f"stableboundary imported from the wrong location: {origin}")
    if origin.name != "__init__.py" or origin.parent.name != PROJECT_NAME:
        raise RuntimeError(
            f"stableboundary imported from an unexpected package path: {origin}"
        )

    versions = _require_keys(
        "installed runtime versions",
        values["versions"],
        {"python", "numpy", "scipy", PROJECT_NAME},
    )
    if versions[PROJECT_NAME] != PROJECT_VERSION:
        raise RuntimeError("installed package runtime version is stale")
    for name in ("python", "numpy", "scipy"):
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

    installed_files = _require_keys(
        "installed package files",
        values["package_files"],
        set(package_files),
    )
    for name, trusted in package_files.items():
        identity = _require_keys(
            f"installed package file {name}",
            installed_files[name],
            {"size", "sha256"},
        )
        if identity != {"size": trusted.size, "sha256": trusted.sha256}:
            raise RuntimeError(
                f"installed package file differs from trusted repository source: {name}"
            )

    direct_url = _require_keys(
        "direct_url.json",
        values["direct_url"],
        {"archive_info", "url"},
    )
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

    archive_info = direct_url["archive_info"]
    if not isinstance(archive_info, dict):
        raise RuntimeError("installed direct_url.json has invalid archive_info")
    hashes = archive_info.get("hashes")
    if not isinstance(hashes, dict) or hashes.get("sha256") != expected_digest:
        raise RuntimeError(
            "installed direct_url.json does not retain the artifact hash"
        )
    legacy_hash = archive_info.get("hash")
    if legacy_hash is not None and legacy_hash != f"sha256={expected_digest}":
        raise RuntimeError("installed direct_url.json reports a contradictory hash")
    return origin


def _installed_probe(python: Path, *, cwd: Path, artifact: Path) -> dict[str, Any]:
    source = """
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

import numpy
import scipy
import stableboundary

distribution = metadata.distribution("stableboundary")
direct_url = distribution.read_text("direct_url.json")
if direct_url is None:
    raise RuntimeError("direct_url.json is missing")
package_root = Path(stableboundary.__file__).resolve().parent
package_files = {}
for path in sorted(package_root.rglob("*")):
    if path.is_symlink():
        raise RuntimeError(f"installed package contains a symbolic link: {path}")
    if not path.is_file() or (path.suffix != ".py" and path.name != "py.typed"):
        continue
    content = path.read_bytes()
    package_files[path.relative_to(package_root).as_posix()] = {
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
print(json.dumps({
    "import_origin": stableboundary.__file__,
    "metadata_version": distribution.version,
    "package_version": stableboundary.__version__,
    "versions": {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "stableboundary": stableboundary.__version__,
    },
    "package_files": package_files,
    "direct_url": json.loads(direct_url),
}, sort_keys=True))
"""
    decoded = json.loads(
        _run(
            [str(python), "-I", "-c", source],
            cwd=cwd,
            stage=f"installed provenance probe for {artifact.name}",
            timeout_seconds=IMPORT_TIMEOUT_SECONDS,
            capture=True,
        )
    )
    if not isinstance(decoded, dict):
        raise RuntimeError("installed provenance probe did not return a JSON object")
    return decoded


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
    python: Path, *, cwd: Path, artifact: Path
) -> dict[str, Any]:
    """Recompute the public experiment without importing its example module."""
    source = r"""
import hashlib
import json

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
        "numpy_version": np.__version__,
        "scipy_version": audit["backend"]["library_version"],
        "sample_sha256": hashlib.sha256(simulation_bytes).hexdigest(),
        "counts": {
            "n_minus": n_minus,
            "n_zero": int(n - n_minus - n_plus),
            "n_plus": n_plus,
        },
        "minimum": float(np.min(simulated)),
        "maximum": float(np.max(simulated)),
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
    decoded = json.loads(
        _run(
            [str(python), "-I", "-c", source],
            cwd=cwd,
            stage=f"independent installed science probe for {artifact.name}",
            timeout_seconds=EXAMPLE_TIMEOUT_SECONDS,
            capture=True,
        )
    )
    if not isinstance(decoded, dict):
        raise RuntimeError("independent installed science probe returned invalid JSON")
    return decoded


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
            f"installed example {name} differs from its trusted numerical reference"
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
            "refinement_near_zero_cutoff",
            "refinement_near_zero_floor",
            "refinement_near_zero_relative",
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
            "refinement_near_zero_cutoff",
            "refinement_near_zero_floor",
            "refinement_near_zero_relative",
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
    near_zero_cutoff = _finite_float(
        "oracle refinement near-zero cutoff",
        configured_tolerances["refinement_near_zero_cutoff"],
    )
    near_zero_floor = _finite_float(
        "oracle refinement near-zero floor",
        configured_tolerances["refinement_near_zero_floor"],
    )
    near_zero_relative = _finite_float(
        "oracle refinement near-zero relative tolerance",
        configured_tolerances["refinement_near_zero_relative"],
    )
    if (
        min(
            absolute_tolerance,
            near_zero_cutoff,
            near_zero_floor,
            near_zero_relative,
        )
        <= 0.0
    ):
        raise RuntimeError("artifact oracle has invalid refinement tolerances")
    for name, actual_value in components.items():
        expected_value = expected_components[name]
        if expected_value <= 0.0 or actual_value <= 0.0:
            raise RuntimeError(
                f"installed example lost nonzero refinement evidence for {name}"
            )
        component_tolerance = (
            max(near_zero_floor, expected_value * near_zero_relative)
            if expected_value < near_zero_cutoff
            else absolute_tolerance
        )
        _reference_close(
            f"refinement {name}",
            actual_value,
            expected_value,
            tolerance=component_tolerance,
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
            "numpy_version",
            "scipy_version",
            "sample_sha256",
            "counts",
            "minimum",
            "maximum",
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
            "approved_environments",
        },
    )
    if (
        contract["sample_size"] != 5_000
        or contract["seed"] != payload["seed"]
        or contract["truth_rule"] != "alpha=2-design.r*1.5,beta=0.35,loc=0,scale=1"
        or simulation["dtype"] != contract["canonical_dtype"]
        or simulation["rng_algorithm"] != contract["rng_algorithm"]
        or simulation["simulator_algorithm"] != contract["simulator_algorithm"]
    ):
        raise RuntimeError("installed simulation changed its fixed algorithm contract")
    numpy_version = simulation["numpy_version"]
    scipy_version = simulation["scipy_version"]
    if not isinstance(numpy_version, str) or not isinstance(scipy_version, str):
        raise RuntimeError("installed simulation returned invalid runtime versions")
    if runtime_versions is not None and (
        runtime_versions.get("numpy") != numpy_version
        or runtime_versions.get("scipy") != scipy_version
    ):
        raise RuntimeError(
            "installed simulation runtime contradicts independent imports"
        )
    environment_key = f"numpy={numpy_version}|scipy={scipy_version}"
    approved = contract["approved_environments"]
    if not isinstance(approved, dict) or environment_key not in approved:
        raise RuntimeError(
            f"installed simulation environment is not approved: {environment_key}"
        )
    expected_simulation = approved[environment_key]
    expected_simulation_values = _require_keys(
        "approved simulation evidence",
        expected_simulation,
        {"dtype", "sample_sha256", "counts", "minimum", "maximum"},
    )
    simulation_counts = _require_keys(
        "simulation cell counts",
        simulation["counts"],
        {"n_minus", "n_zero", "n_plus"},
    )
    validated_simulation_counts = {
        name: _strict_nonnegative_int(f"simulation count {name}", value)
        for name, value in simulation_counts.items()
    }
    if sum(validated_simulation_counts.values()) != contract["sample_size"]:
        raise RuntimeError("installed simulation counts have the wrong sample size")
    for name in ("dtype", "sample_sha256"):
        if simulation[name] != expected_simulation_values[name]:
            raise RuntimeError(f"installed simulation changed approved {name}")
    if validated_simulation_counts != expected_simulation_values["counts"]:
        raise RuntimeError("installed simulation changed approved counts")
    for name in ("minimum", "maximum"):
        _reference_close(
            f"simulation {name}",
            simulation[name],
            expected_simulation_values[name],
            tolerance=0.0,
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


def _exercise_archive(
    snapshot: ArtifactSnapshot,
    *,
    package_files: dict[str, FileIdentity],
) -> dict[str, Any]:
    artifact = snapshot.path
    _assert_snapshot(snapshot)
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
        _assert_snapshot(snapshot)
        _install_archive(python, artifact, cwd=work)
        _assert_snapshot(snapshot)
        probe = _installed_probe(python, cwd=work, artifact=artifact)
        _assert_snapshot(snapshot)
        origin = _validate_installed_distribution(
            probe,
            artifact=artifact,
            environment=environment,
            expected_digest=snapshot.sha256,
            package_files=package_files,
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
        _assert_snapshot(snapshot)
        science_probe = _installed_science_probe(
            python,
            cwd=work,
            artifact=artifact,
        )
        _assert_snapshot(snapshot)
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
        versions = probe.get("versions")
        if not isinstance(versions, dict):
            raise RuntimeError("installed provenance probe omitted runtime versions")
        _validate_example(payload, runtime_versions=versions)
        _validate_reflection_probe(payload, science_values["reflection"])
        _assert_snapshot(snapshot)
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
    with _artifact_snapshots(wheel, sdist) as (wheel_snapshot, sdist_snapshot):
        inspection = _inspect_archives(wheel_snapshot.path, sdist_snapshot.path)
        _assert_snapshot(wheel_snapshot)
        _assert_snapshot(sdist_snapshot)
        if (
            inspection.wheel_sha256 != wheel_snapshot.sha256
            or inspection.sdist_sha256 != sdist_snapshot.sha256
        ):
            raise RuntimeError("artifact snapshots changed during archive inspection")
        wheel_payload = _exercise_archive(
            wheel_snapshot,
            package_files=inspection.package_files,
        )
        sdist_payload = _exercise_archive(
            sdist_snapshot,
            package_files=inspection.package_files,
        )
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
