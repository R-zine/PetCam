"""Small, dependency-free MJPEG reader.

PetCam streams multipart JPEG data.  Extracting JPEG SOI/EOI markers instead
of decoding and re-encoding the frames preserves the exact camera bytes and is
also tolerant of multipart boundaries being split across network reads.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)
JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"


class MJPEGError(RuntimeError):
    """Raised when an MJPEG stream cannot produce a usable frame."""


@dataclass(frozen=True, slots=True)
class JPEGFrame:
    """An original JPEG frame and the local UTC time it was received."""

    data: bytes
    captured_at: datetime


def iter_jpeg_frames(
    chunks: Iterable[bytes], *, max_frame_bytes: int = 16 * 1024 * 1024
) -> Iterator[bytes]:
    """Extract complete JPEG images from arbitrary byte chunks.

    Multipart headers and other bytes outside JPEG markers are ignored. An
    oversized/incomplete frame is discarded so a later valid frame can still
    be recovered.
    """

    if max_frame_bytes < 4:
        raise ValueError("max_frame_bytes must be at least 4")

    buffer = bytearray()
    for chunk in chunks:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("MJPEG chunks must be bytes-like")
        if not chunk:
            continue
        buffer.extend(chunk)

        while buffer:
            start = buffer.find(JPEG_START)
            if start < 0:
                # Retain a trailing 0xff because it may be the first half of SOI.
                buffer[:] = buffer[-1:] if buffer[-1] == 0xFF else b""
                break
            if start:
                del buffer[:start]

            end = buffer.find(JPEG_END, len(JPEG_START))
            if end >= 0:
                frame_end = end + len(JPEG_END)
                yield bytes(buffer[:frame_end])
                del buffer[:frame_end]
                continue

            if len(buffer) > max_frame_bytes:
                next_start = buffer.find(JPEG_START, len(JPEG_START))
                if next_start >= 0:
                    del buffer[:next_start]
                else:
                    buffer.clear()
                LOGGER.warning("Discarded an oversized or incomplete MJPEG frame")
                continue
            break


class MJPEGStream:
    """Read JPEG frames from an HTTP MJPEG endpoint.

    ``iter_frames(reconnect=True)`` is intended for capture jobs and retries a
    dropped connection. ``iter_frames()`` and ``read_frame()`` perform a single
    connection attempt, which is convenient for one-shot inference.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 10.0,
        chunk_size: int = 64 * 1024,
        reconnect_delay: float = 1.0,
        max_frame_bytes: int = 16 * 1024 * 1024,
        opener: Callable[..., BinaryIO] | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("stream URL must start with http:// or https://")
        if timeout <= 0 or chunk_size <= 0 or reconnect_delay < 0:
            raise ValueError("timeout and chunk_size must be positive")
        self.url = url
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.reconnect_delay = reconnect_delay
        self.max_frame_bytes = max_frame_bytes
        self._opener = opener or urlopen

    def _chunks(self, response: BinaryIO) -> Iterator[bytes]:
        while True:
            chunk = response.read(self.chunk_size)
            if not chunk:
                return
            yield chunk

    def iter_frames(
        self,
        *,
        reconnect: bool = False,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[JPEGFrame]:
        """Yield frames, optionally reconnecting after temporary failures."""

        while not (stop_requested and stop_requested()):
            try:
                request = Request(self.url, headers={"User-Agent": "petcam-ml/1"})
                with self._opener(request, timeout=self.timeout) as response:
                    for data in iter_jpeg_frames(
                        self._chunks(response), max_frame_bytes=self.max_frame_bytes
                    ):
                        yield JPEGFrame(data=data, captured_at=datetime.now(timezone.utc))
                        if stop_requested and stop_requested():
                            return
                if not reconnect:
                    return
                LOGGER.warning("MJPEG stream ended; reconnecting to %s", self.url)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                if not reconnect:
                    raise MJPEGError(f"Unable to read MJPEG stream {self.url}: {exc}") from exc
                LOGGER.warning("MJPEG stream dropped (%s); retrying", exc)

            if stop_requested and stop_requested():
                return
            if self.reconnect_delay:
                time.sleep(self.reconnect_delay)

    def __iter__(self) -> Iterator[JPEGFrame]:
        return self.iter_frames()

    def read_frame(self) -> JPEGFrame:
        """Read and return one complete JPEG frame."""

        try:
            return next(self.iter_frames())
        except StopIteration as exc:
            raise MJPEGError(f"Stream {self.url} ended before a JPEG frame arrived") from exc


def read_mjpeg_frame(url: str, *, timeout: float = 10.0) -> bytes:
    """Convenience helper returning one original JPEG frame as bytes."""

    return MJPEGStream(url, timeout=timeout).read_frame().data
