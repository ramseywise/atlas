# cap-vision — Templates for vision-builder subagent

## File: {OUTPUT_DIR}/vision.py

```python
from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

_SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB — Anthropic/Gemini limit


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class VisionInput(BaseModel):
    """Schema for a vision query — integrates with AssistantResponse pipeline."""

    image_path: str
    query: str
    caption: str = ""

    @field_validator("image_path")
    @classmethod
    def path_must_exist(cls, v: str) -> str:
        p = Path(v)
        if not p.exists():
            raise ValueError(f"Image file not found: {v}")
        if not p.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v

    @field_validator("image_path")
    @classmethod
    def mime_must_be_supported(cls, v: str) -> str:
        mime, _ = mimetypes.guess_type(v)
        if mime not in _SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported image type {mime!r} for {v}. "
                f"Supported: {sorted(_SUPPORTED_MIME_TYPES)}"
            )
        return v


# ---------------------------------------------------------------------------
# Base64 encoding
# ---------------------------------------------------------------------------


def encode_image_base64(path: str) -> str:
    """Read an image file from disk and return its base64-encoded contents.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        Base64-encoded string (no line breaks, no data-URI prefix).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the image exceeds the maximum allowed size.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    size = p.stat().st_size
    if size > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image {path} is {size / 1024 / 1024:.1f} MB — "
            f"exceeds {_MAX_IMAGE_BYTES / 1024 / 1024:.0f} MB limit"
        )

    with p.open("rb") as fh:
        raw = fh.read()

    return base64.standard_b64encode(raw).decode("utf-8")


# ---------------------------------------------------------------------------
# Message formatter
# ---------------------------------------------------------------------------


def image_to_message(
    path: str,
    caption: str = "",
    provider: str = "anthropic",
) -> dict[str, Any]:
    """Format a local image file as a provider-compatible multimodal message dict.

    Supports Anthropic (Claude) and Google (Gemini) message formats.

    Args:
        path: Path to the image file.
        caption: Optional human-readable description of the image.
        provider: "anthropic" (default) or "gemini".

    Returns:
        A dict ready to be embedded in the messages list for the provider's API.

    Example (Anthropic):
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
                {"type": "text", "text": "What does this image show?"}
            ]
        }

    Example (Gemini via google-genai):
        {
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": "..."}},
                {"text": "What does this image show?"}
            ]
        }
    """
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type not in _SUPPORTED_MIME_TYPES:
        raise ValueError(f"Unsupported image MIME type: {mime_type!r}")

    image_b64 = encode_image_base64(path)
    text_part = caption or "Please analyse this image."

    if provider == "anthropic":
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": text_part,
                },
            ],
        }

    elif provider == "gemini":
        return {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_b64,
                    }
                },
                {"text": text_part},
            ]
        }

    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'anthropic' or 'gemini'.")


# ---------------------------------------------------------------------------
# Convenience: open with Pillow and resize before encoding
# ---------------------------------------------------------------------------


def resize_and_encode(
    path: str,
    max_dimension: int = 1568,
    quality: int = 85,
) -> tuple[str, str]:
    """Resize an image to fit within max_dimension on the longest side and base64-encode it.

    Returns (base64_string, mime_type). Useful for large screenshots or scanned documents
    that would exceed the provider's size limit.

    Requires: Pillow (pip install Pillow)
    """
    try:
        from PIL import Image  # type: ignore[import]  # noqa: PLC0415
        import io  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Pillow is required for image resizing. Run: pip install Pillow") from exc

    with Image.open(path) as img:
        # Convert RGBA / P (palette) → RGB for JPEG compatibility
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            output_format = "JPEG"
            mime_type = "image/jpeg"
        else:
            output_format = img.format or "JPEG"
            mime_type, _ = mimetypes.guess_type(path)
            mime_type = mime_type or "image/jpeg"

        # Resize preserving aspect ratio
        w, h = img.size
        if max(w, h) > max_dimension:
            ratio = max_dimension / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            logger.debug("Resized image from %dx%d to %dx%d", w, h, *new_size)

        buf = io.BytesIO()
        save_kwargs: dict[str, Any] = {"format": output_format}
        if output_format == "JPEG":
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        img.save(buf, **save_kwargs)

    encoded = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return encoded, mime_type
```
