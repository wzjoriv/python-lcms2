"""Exceptions raised by lcms2."""

from ._lcms2 import LCMSError

#: Raised when LittleCMS rejects an operation; the message is LittleCMS's own text.
#: A subclass of :class:`ValueError`.
CMSError = LCMSError


class ICCParseError(CMSError):
    """Raised when an ICC profile byte stream is malformed or unsupported."""


class ICCWriteError(CMSError):
    """Raised when a ColorSpace cannot be represented as a valid ICC profile."""


__all__ = ["CMSError", "ICCParseError", "ICCWriteError"]
