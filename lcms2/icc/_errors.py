"""Errors raised while reading or writing ICC profiles."""

from __future__ import annotations

__all__ = ["ICCParseError", "ICCWriteError"]


class ICCParseError(ValueError):
    """Raised when an ICC profile byte stream is malformed or unsupported."""


class ICCWriteError(ValueError):
    """Raised when a ColorSpace cannot be represented as a valid ICC profile."""
