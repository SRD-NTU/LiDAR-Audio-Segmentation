"""Download and verify the published SAID and Audio2Sph checkpoints."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Runbang Wang

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPException
import os
from pathlib import Path
import sys
import tempfile
from typing import BinaryIO, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None

try:  # Windows
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    _msvcrt = None


DEFAULT_RELEASE_BASE_URL: str | None = (
    "https://huggingface.co/IN03X/SAID/resolve/main/"
)
MODEL_LICENSE_NAME = "SAID Model Weights Non-Commercial Research License 1.0"


@dataclass(frozen=True)
class CheckpointAsset:
    """Immutable public identity of one complete paper model."""

    model: str
    filename: str
    size_bytes: int
    sha256: str


CHECKPOINT_ASSETS = {
    "said_passt": CheckpointAsset(
        model="said_passt",
        filename="said_passt.ckpt",
        size_bytes=547_886_573,
        sha256="85b72bd4a0749dd4ca080ddea30eba68a903af465d916408a03e21912609ca1b",
    ),
    "said_audiomae": CheckpointAsset(
        model="said_audiomae",
        filename="said_audiomae.ckpt",
        size_bytes=549_046_019,
        sha256="018f9d05bfa681456334811abe6e62c59331a0a8f4313eb763e5c32b0148c626",
    ),
    "audio2sph": CheckpointAsset(
        model="audio2sph",
        filename="audio2sph.ckpt",
        size_bytes=110_233_837,
        sha256="bd4c2c44e24d4c207785a46cfc95efffc3f9a7a9a357d4563553a886be0b917b",
    ),
}


def checkpoint_cache_directory() -> Path:
    """Return the user-overridable project checkpoint directory."""

    configured = os.environ.get("SAID_CHECKPOINT_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "said" / "utils" / "download_checkpoints.py").is_file()
        ):
            return candidate / "checkpoints"
    return working_directory / "checkpoints"


def checkpoint_cache_path(model: str) -> Path:
    """Return the canonical project path for a paper model."""

    try:
        asset = CHECKPOINT_ASSETS[model]
    except KeyError as error:
        raise ValueError(
            "model must be 'said_passt', 'said_audiomae', or 'audio2sph'"
        ) from error
    return checkpoint_cache_directory() / asset.filename


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _exclusive_file_lock(stream: BinaryIO) -> Iterator[None]:
    """Serialize checkpoint publication on POSIX and Windows."""

    if _fcntl is not None:
        _fcntl.flock(stream.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(stream.fileno(), _fcntl.LOCK_UN)
        return
    if _msvcrt is not None:  # pragma: no cover - exercised on Windows
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        _msvcrt.locking(stream.fileno(), _msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            _msvcrt.locking(stream.fileno(), _msvcrt.LK_UNLCK, 1)
        return
    raise RuntimeError("this platform does not provide a supported file lock")


def verify_checkpoint_asset(path: str | Path, model: str) -> Path:
    """Verify byte length and SHA256 before a downloaded file is trusted."""

    try:
        asset = CHECKPOINT_ASSETS[model]
    except KeyError as error:
        raise ValueError(
            "model must be 'said_passt', 'said_audiomae', or 'audio2sph'"
        ) from error
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is not a regular file: {checkpoint}")
    actual_size = checkpoint.stat().st_size
    if actual_size != asset.size_bytes:
        raise RuntimeError(
            f"checkpoint size mismatch for {checkpoint}: "
            f"got {actual_size}, expected {asset.size_bytes}"
        )
    actual_sha256 = _file_sha256(checkpoint)
    if actual_sha256 != asset.sha256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch for {checkpoint}: "
            f"got {actual_sha256}, expected {asset.sha256}"
        )
    return checkpoint


def _asset_url(asset: CheckpointAsset, base_url: str | None) -> str:
    root = (
        base_url
        or os.environ.get("SAID_CHECKPOINT_BASE_URL")
        or DEFAULT_RELEASE_BASE_URL
    )
    if root is None:
        raise RuntimeError(
            "automatic paper-checkpoint download is not enabled in this "
            "release; supply --checkpoint with an authorized local asset or "
            "set SAID_CHECKPOINT_BASE_URL to an authorized release mirror"
        )
    if not root.endswith("/"):
        root += "/"
    return urljoin(root, asset.filename)


def _show_download_progress(current_size: int, expected_size: int) -> None:
    """Render checkpoint download progress in an interactive terminal."""

    if not sys.stderr.isatty():
        return
    fraction = min(max(current_size / max(expected_size, 1), 0.0), 1.0)
    width = 24
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    current_mib = current_size / (1024 * 1024)
    expected_mib = expected_size / (1024 * 1024)
    ending = "\n" if current_size >= expected_size else ""
    print(
        f"\rCheckpoint [{bar}] {fraction:6.1%} "
        f"({current_mib:.1f}/{expected_mib:.1f} MiB)",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


def _download_with_resume(url: str, partial: Path, expected_size: int) -> None:
    maximum_requests = 16
    last_error: BaseException | None = None
    for request_index in range(maximum_requests):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset == expected_size:
            return
        if offset > expected_size:
            partial.unlink()
            offset = 0

        headers = {"User-Agent": "SAID/1.0 checkpoint downloader"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(url, headers=headers)
        try:
            response = urlopen(request, timeout=60)
        except (HTTPError, URLError, TimeoutError) as error:
            # A public CDN may close a connection before returning response
            # bytes. Preserve the partial file and retry the same range.
            last_error = error
            continue

        status = getattr(response, "status", None)
        append = offset > 0 and status == 206
        mode = "ab" if append else "wb"
        starting_size = offset if append else 0
        transfer_error: BaseException | None = None
        try:
            with response, partial.open(mode) as stream:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    _show_download_progress(stream.tell(), expected_size)
                stream.flush()
                os.fsync(stream.fileno())
        except (HTTPException, OSError, TimeoutError) as error:
            transfer_error = error

        current_size = partial.stat().st_size if partial.is_file() else 0
        if current_size == expected_size:
            return
        if current_size > expected_size:
            raise RuntimeError(
                f"checkpoint download exceeded expected size at {partial}: "
                f"got {current_size}, expected {expected_size}"
            )
        if current_size <= starting_size:
            # Treat a connection that yielded no new bytes as transient. The
            # finite request budget prevents an unavailable endpoint from
            # spinning forever while allowing Hugging Face CDN retries to
            # resume from the exact byte already stored.
            last_error = transfer_error or RuntimeError(
                f"checkpoint download made no progress from {url}"
            )
            continue
        last_error = transfer_error

    current_size = partial.stat().st_size if partial.is_file() else 0
    detail = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(
        f"checkpoint download remained incomplete after {maximum_requests} "
        f"requests: got {current_size}, expected {expected_size}{detail}"
    ) from last_error


def ensure_paper_checkpoint(
    model: str,
    *,
    destination: str | Path | None = None,
    base_url: str | None = None,
) -> Path:
    """Return a verified checkpoint, downloading it atomically when absent.

    A process lock serializes concurrent callers. Interrupted downloads remain
    as ``.part`` files and resume when the server supports HTTP byte ranges.
    The completed file is published only after its size and SHA256 match the
    paper asset registry.
    """

    try:
        asset = CHECKPOINT_ASSETS[model]
    except KeyError as error:
        raise ValueError(
            "model must be 'said_passt', 'said_audiomae', or 'audio2sph'"
        ) from error
    target = (
        checkpoint_cache_path(model)
        if destination is None
        else Path(destination).expanduser().resolve()
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    with lock_path.open("a+b") as lock, _exclusive_file_lock(lock):
        if target.exists():
            return verify_checkpoint_asset(target, model)
        partial = target.with_name(f".{target.name}.part")
        url = _asset_url(asset, base_url)
        print(
            "Downloading the separately licensed SAID paper checkpoint for "
            f"non-commercial research. Terms: {MODEL_LICENSE_NAME}, included "
            f"with the checkpoint bundle.\nCheckpoint destination: {target}",
            file=sys.stderr,
            flush=True,
        )
        _download_with_resume(
            url,
            partial,
            asset.size_bytes,
        )
        try:
            verify_checkpoint_asset(partial, model)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        os.replace(partial, target)
        if os.name != "nt":
            directory_descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        return verify_checkpoint_asset(target, model)


def materialize_local_release_asset(
    source: str | Path,
    model: str,
    release_directory: str | Path,
) -> Path:
    """Create a verified hard-linked release asset without duplicating bytes."""

    verified = verify_checkpoint_asset(source, model)
    destination_directory = Path(release_directory).expanduser().resolve()
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / CHECKPOINT_ASSETS[model].filename
    if destination.exists():
        return verify_checkpoint_asset(destination, model)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination_directory,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        os.link(verified, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return verify_checkpoint_asset(destination, model)


__all__ = [
    "CHECKPOINT_ASSETS",
    "CheckpointAsset",
    "checkpoint_cache_directory",
    "checkpoint_cache_path",
    "ensure_paper_checkpoint",
    "materialize_local_release_asset",
    "verify_checkpoint_asset",
]
