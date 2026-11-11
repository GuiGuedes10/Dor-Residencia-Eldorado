"""
Artifact Persistence Engine for REIA Web Navigation.

Manages filesystem storage for visual (PNG screenshots) and textual (JSON metadata)
evidence captured during web automation. Features a zero-dependency pure-Python
valid 1280x800 PNG generator using standard library struct and zlib.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid
import zlib


def generate_png_bytes(
    width: int = 1280,
    height: int = 800,
    bg_rgb: Tuple[int, int, int] = (24, 76, 120),
) -> bytes:
    """Generate a real, valid binary PNG image in pure Python.

    Constructs standard PNG chunks (IHDR, IDAT, IEND) with valid CRC checksums.
    Zero external dependencies (requires only standard library struct and zlib).

    Args:
        width: Image width in pixels (default: 1280).
        height: Image height in pixels (default: 800).
        bg_rgb: Background color as (Red, Green, Blue) tuple in [0, 255].

    Returns:
        Raw bytes conforming strictly to ISO/IEC 15948:2004 (PNG standard).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive integers, got {width}x{height}")

    r, g, b = (max(0, min(255, c)) for c in bg_rgb)

    # Standard 8-byte PNG signature: 0x89 'P' 'N' 'G' \r \n 0x1A \n
    png = bytearray(b"\x89PNG\r\n\x1a\n")

    # IHDR chunk: width (4B), height (4B), bit_depth (1B=8), color_type (1B=2 Truecolor RGB),
    # compression (1B=0), filter (1B=0), interlace (1B=0)
    ihdr_payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_payload)
    png += struct.pack(">I", len(ihdr_payload)) + b"IHDR" + ihdr_payload + struct.pack(">I", ihdr_crc)

    # IDAT chunk: raw uncompressed scanlines. Each scanline begins with filter byte 0 (None),
    # followed by 3 bytes per pixel (R, G, B).
    raw_scanline = bytes([0] + [r, g, b] * width)
    raw_image_data = raw_scanline * height
    compressed_data = zlib.compress(raw_image_data, level=6)

    idat_crc = zlib.crc32(b"IDAT" + compressed_data)
    png += struct.pack(">I", len(compressed_data)) + b"IDAT" + compressed_data + struct.pack(">I", idat_crc)

    # IEND chunk: 0 payload bytes marking end of PNG datastream
    iend_crc = zlib.crc32(b"IEND")
    png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    return bytes(png)


class ArtifactEngine:
    """Manages visual and textual artifact persistence for web navigation sessions."""

    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        """Initialize artifact engine and ensure destination directories exist.

        Args:
            artifacts_dir: Root directory for artifacts (default: 'artifacts').
        """
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.screenshots_dir = self.artifacts_dir / "screenshots"
        self.metadata_dir = self.artifacts_dir / "metadata"
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create target artifacts subdirectories if they do not exist."""
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_slug(self, name: str) -> str:
        """Sanitize an intent or file slug for safe filesystem usage."""
        if not name:
            return "general"
        return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")

    def save_screenshot(
        self,
        intent_name: str,
        image_bytes: Optional[bytes] = None,
        custom_filename: Optional[str] = None,
    ) -> str:
        """Save a screenshot PNG to artifacts/screenshots/.

        If image_bytes is None, generates a valid 1280x800 PNG using generate_png_bytes.

        Args:
            intent_name: Intent name used in file naming.
            image_bytes: Binary PNG payload. Defaults to synthesized PNG.
            custom_filename: Optional specific filename.

        Returns:
            Absolute filesystem path to saved screenshot file.
        """
        self.ensure_directories()

        if image_bytes is None:
            image_bytes = generate_png_bytes(width=1280, height=800)

        slug = self._sanitize_slug(intent_name)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:6]

        filename = custom_filename or f"screenshot_{slug}_{now_str}_{unique_token}.png"
        target_path = self.screenshots_dir / filename

        target_path.write_bytes(image_bytes)
        return str(target_path)

    def save_metadata(
        self,
        intent_name: str,
        target_url: str,
        page_title: str,
        headings: List[str],
        screenshot_path: str,
        duration: float = 0.0,
        links: Optional[List[Dict[str, str]]] = None,
        error_message: Optional[str] = None,
        custom_filename: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save navigation session metadata to artifacts/metadata/.

        Args:
            intent_name: Canonical intent identifier.
            target_url: Target URL accessed.
            page_title: Title extracted from page DOM.
            headings: Headings (H1, H2, etc.) extracted from DOM.
            screenshot_path: Path to associated screenshot file.
            duration: Total page load and processing duration in seconds.
            links: Extracted anchor links.
            error_message: Optional error message if navigation encountered an issue.
            custom_filename: Optional explicit filename.
            extra: Optional additional metadata fields.

        Returns:
            Absolute filesystem path to saved JSON metadata file.
        """
        self.ensure_directories()

        slug = self._sanitize_slug(intent_name)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:6]

        filename = custom_filename or f"metadata_{slug}_{now_str}_{unique_token}.json"
        target_path = self.metadata_dir / filename

        screenshot_file = Path(screenshot_path).name if screenshot_path else ""

        payload: Dict[str, Any] = {
            "intent_name": intent_name,
            "target_url": target_url,
            "page_title": page_title,
            "headings": list(headings),
            "links_count": len(links) if links is not None else 0,
            "links": list(links) if links is not None else [],
            "screenshot_file": screenshot_file,
            "screenshot_path": str(screenshot_path),
            "timestamp": time.time(),
            "iso_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "load_time_seconds": round(duration, 4),
            "duration_seconds": round(duration, 4),
            "error_message": error_message,
        }

        if extra:
            payload["extra"] = extra

        target_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(target_path)
