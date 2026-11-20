"""
`ICCSpace`: a color space read from an ICC profile.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import Any

import numpy as np

from .._errors import ICCParseError
from ..spaces import ColorSpace
from ._io import tagtypes as tt
from ._io.profile import ICCProfile
from ._reader import build_transform, read_circular_channels, read_device_domain
from ._versions import normalize_icc_version

__all__ = ["ICCSpace"]


class ICCSpace(ColorSpace):
    """A color space read from an ICC profile (v2, v4 or v5).

    Parameters
    ----------
    source : str, bytes or ICCProfile
        Path to a profile, its bytes, or a parsed `ICCProfile`.

    Attributes
    ----------
    profile : ICCProfile
        The parsed profile.
    description : str or None
        The profile's description text.
    kind : str
        ``'matrix_trc'``, ``'gray_trc'`` or ``'lut'``.
    """

    _abstract = True

    def __init__(self, source: str | bytes | ICCProfile):
        self.icc = source if isinstance(source, ICCProfile) else ICCProfile.load(source)
        self.description = self._read_description()
        built = build_transform(self.icc)
        self.kind = built.kind
        to_xyz, from_xyz = built.to_xyz, built.from_xyz

        domain = read_device_domain(self.icc, built.n_channels)
        if domain is not None:
            lo = np.array([p[0] for p in domain])
            span = np.array([p[1] - p[0] for p in domain])
            raw_to, raw_from = to_xyz, from_xyz

            def to_xyz(values: Any, _f: Callable = raw_to) -> np.ndarray:
                return _f((np.asarray(values, dtype=np.float64) - lo) / span)

            def from_xyz(xyz: Any, _f: Callable = raw_from) -> np.ndarray:
                return _f(xyz) * span + lo

        self.circular_device_channels = read_circular_channels(self.icc)

        header = self.icc.header
        try:
            super().__init__(
                n_channels=built.n_channels,
                to_xyz=to_xyz,
                from_xyz=from_xyz,
                name=self.description or header.data_colour_space.decode("ascii", errors="replace").strip(),
                device_domain=domain,
            )
        except ValueError as e:
            raise ICCParseError(f"invalid device range or circular channels: {e}") from e
        self.icc_colour_space_signature = header.data_colour_space

    def _read_description(self) -> str | None:
        body = self.icc.get_tag(b"desc")
        if body is None:
            return None
        try:
            if body[:4] == b"mluc":
                texts = tt.decode_mluc(body)
                return next(iter(texts.values()), None)
            return tt.decode_desc_v2(body)
        except (ValueError, IndexError, struct.error):
            return None

    def to_icc_profile(self, path: str | None = None, **kwargs: Any) -> dict:
        """Write this space as an ICC profile.

        At the profile's own major version (and without ``force_clut``)
        the original tags are written unchanged with ``method`` set to
        ``'verbatim_tags'``; otherwise this behaves like
        `ColorSpace.to_icc_profile`.

        Parameters
        ----------
        path : str, optional
            File to write the profile to.
        **kwargs
            Keyword arguments of `ColorSpace.to_icc_profile`.

        Returns
        -------
        dict
            ``{'bytes': bytes, 'method': str, 'notes': list of str}``.
        """
        version = normalize_icc_version(kwargs.get("version", 4))
        header = self.icc.header
        if version[0] != header.version_major or kwargs.get("force_clut"):
            return super().to_icc_profile(path, **{**kwargs, "version": version})
        raw = ICCProfile.build_bytes(
            version=version,
            device_class=header.device_class,
            data_colour_space=header.data_colour_space,
            pcs=header.pcs,
            rendering_intent=kwargs.get("rendering_intent", header.rendering_intent),
            pcs_illuminant=header.pcs_illuminant,
            tags=dict(self.icc.tags),
        )
        if path is not None:
            with open(path, "wb") as f:
                f.write(raw)
        return {"bytes": raw, "method": "verbatim_tags", "notes": ["original tags copied"]}
