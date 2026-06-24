"""Image utility functions for multimodal labeling pipeline.

Provides base64 encoding, multimodal message construction,
image validation, and resize-for-API helpers.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Optional

from src.autodata.utils.logging_utils import get_logger

logger = get_logger("image_utils")

# Maximum image dimensions for API calls (resize if larger)
MAX_API_WIDTH = 1024
MAX_API_HEIGHT = 1024


def encode_image_to_base64(image_path: str | Path) -> str:
    """Read an image file and return a base64 data URL string.

    Returns format: data:image/jpeg;base64,<encoded_data>
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    # Determine MIME type from extension
    ext = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime = mime_map.get(ext, "image/jpeg")

    return f"data:{mime};base64,{encoded}"


def resize_image_for_api(
    image_path: str | Path,
    max_width: int = MAX_API_WIDTH,
    max_height: int = MAX_API_HEIGHT,
) -> str:
    """Resize an image in-memory if it exceeds max dimensions, then base64 encode.

    This never modifies the original file. It resizes the in-memory copy
    for API transmission efficiency.
    """
    from PIL import Image

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path)
    img = img.convert("RGB")  # Ensure RGB mode

    orig_w, orig_h = img.size
    needs_resize = orig_w > max_width or orig_h > max_height

    if needs_resize:
        # Scale down proportionally
        ratio = min(max_width / orig_w, max_height / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"Resized {path.name}: {orig_w}x{orig_h} -> {new_w}x{new_h}")

    # Encode to base64 via in-memory JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def build_multimodal_message(
    text: str,
    image_paths: list[str | Path],
    resize: bool = True,
) -> dict[str, Any]:
    """Construct an OpenAI-format multimodal user message.

    Args:
        text: The text prompt to send alongside the image(s).
        image_paths: Paths to image files to include.
        resize: Whether to resize images before encoding (default True).

    Returns:
        A dict suitable for use as a message in OpenAI chat API:
        {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]

    for path in image_paths:
        if resize:
            b64_url = resize_image_for_api(path)
        else:
            b64_url = encode_image_to_base64(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": b64_url},
        })

    return {"role": "user", "content": content}


def validate_image_file(image_path: str | Path) -> dict[str, Any]:
    """Check if an image file exists, is non-zero, and is a valid image.

    Returns dict with:
        valid: bool
        size_bytes: int
        width: int (0 if cannot determine)
        height: int (0 if cannot determine)
        reason: str (empty if valid, error description if not)
    """
    from PIL import Image

    path = Path(image_path)
    result = {
        "valid": False,
        "size_bytes": 0,
        "width": 0,
        "height": 0,
        "reason": "",
    }

    if not path.exists():
        result["reason"] = "file_not_found"
        return result

    size = path.stat().st_size
    result["size_bytes"] = size

    if size == 0:
        result["reason"] = "zero_byte_file"
        return result

    try:
        img = Image.open(path)
        img.verify()  # Check integrity without full decode
        result["width"], result["height"] = img.size
        result["valid"] = True
        result["reason"] = ""
    except Exception as e:
        result["reason"] = f"corrupt_image: {str(e)[:50]}"

    return result


def get_image_dimensions(image_path: str | Path) -> tuple[int, int]:
    """Get image dimensions quickly without full decode.

    Returns (width, height) or (0, 0) if cannot determine.
    """
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)