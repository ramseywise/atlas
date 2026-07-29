"""Shared configuration constants."""

from __future__ import annotations

import os

ATLAS_LLM_MODEL = os.environ.get("ATLAS_LLM_MODEL", "claude-haiku-4-5-20251001")
