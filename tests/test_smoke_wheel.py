"""Focused regression tests for the distribution-artifact smoke runner."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import smoke_wheel


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
