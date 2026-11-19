"""Exceptions raised by lcms2."""

from ._lcms2 import LCMSError

#: Raised when LittleCMS rejects an operation; the message is LittleCMS's own text.
#: A subclass of :class:`ValueError`.
CMSError = LCMSError

__all__ = ["CMSError"]
