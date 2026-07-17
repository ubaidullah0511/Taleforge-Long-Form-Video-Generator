"""Reads just enough bytes of a remote image to get its real width/height,
without downloading the whole file — stdlib only (no Pillow dependency).

Used to enrich providers (NASA, Internet Archive) whose search APIs don't
report dimensions, so the existing pre-download 16:9 filter
(_known_and_not_horizontal in documentary_pipeline.py) can actually catch
non-horizontal candidates from them too, instead of only finding out after a
full download.
"""
import struct

import httpx

_PROBE_BYTES = 65536


def _parse_png(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24 or data[12:16] != b"IHDR":
        return 0, 0
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _parse_jpeg(data: bytes) -> tuple[int, int]:
    if data[:2] != b"\xff\xd8":
        return 0, 0
    idx, n = 2, len(data)
    while idx + 1 < n:
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        idx += 2
        if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue  # markers with no length/payload (TEM, SOI, EOI, RSTn)
        if idx + 2 > n:
            break
        seg_len = struct.unpack(">H", data[idx:idx + 2])[0]
        is_sof = 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC)
        if is_sof:
            if idx + 7 > n:
                break
            height, width = struct.unpack(">HH", data[idx + 3:idx + 7])
            return width, height
        idx += seg_len
    return 0, 0


async def probe_remote_image_size(client: httpx.AsyncClient, url: str) -> tuple[int, int]:
    """Streams only the first _PROBE_BYTES of url (stopping the connection
    early regardless of whether the server honors the Range header) and
    parses a PNG/JPEG header out of it. Returns (0, 0) on any network,
    format, or truncation failure — callers already treat that identically
    to "provider didn't report dimensions"."""
    try:
        data = bytearray()
        async with client.stream(
            "GET", url, headers={"Range": f"bytes=0-{_PROBE_BYTES - 1}"}, timeout=10, follow_redirects=True,
        ) as resp:
            if resp.status_code not in (200, 206):
                return 0, 0
            async for chunk in resp.aiter_bytes():
                data.extend(chunk)
                if len(data) >= _PROBE_BYTES:
                    break
    except httpx.HTTPError:
        return 0, 0

    data = bytes(data)
    if data.startswith(b"\x89PNG"):
        return _parse_png(data)
    if data.startswith(b"\xff\xd8"):
        return _parse_jpeg(data)
    return 0, 0
