from __future__ import annotations

from io import BytesIO

import pytest

from rabbitcam_ml.camera.mjpeg import MJPEGError, MJPEGStream, iter_jpeg_frames


JPEG_ONE = b"\xff\xd8first-jpeg\xff\xd9"
JPEG_TWO = b"\xff\xd8second-jpeg\xff\xd9"


def test_extracts_frames_across_arbitrary_chunk_boundaries() -> None:
    chunks = [
        b"multipart headers\r\n\xff",
        b"\xd8first-",
        b"jpeg\xff\xd9boundary\xff\xd8second-jpeg\xff",
        b"\xd9trailer",
    ]
    assert list(iter_jpeg_frames(chunks)) == [JPEG_ONE, JPEG_TWO]


def test_oversized_incomplete_frame_recovers_at_next_soi() -> None:
    chunks = [b"\xff\xd8" + b"x" * 20, JPEG_TWO]
    assert list(iter_jpeg_frames(chunks, max_frame_bytes=16)) == [JPEG_TWO]


class _Response(BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_stream_read_frame_returns_original_jpeg_bytes() -> None:
    def opener(_request: object, *, timeout: float) -> _Response:
        assert timeout == 2.0
        return _Response(b"header" + JPEG_ONE + b"boundary")

    frame = MJPEGStream("http://camera:81/stream", timeout=2.0, opener=opener).read_frame()
    assert frame.data == JPEG_ONE
    assert frame.captured_at.tzinfo is not None


def test_empty_stream_raises_clear_error() -> None:
    stream = MJPEGStream(
        "http://camera:81/stream", opener=lambda *_args, **_kwargs: _Response(b"nothing")
    )
    with pytest.raises(MJPEGError, match="before a JPEG"):
        stream.read_frame()
