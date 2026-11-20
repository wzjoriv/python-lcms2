"""Shared state for writing one color space: domain, samples and structure."""

from __future__ import annotations

import numpy as np

from ..._errors import ICCWriteError
from .._tracer.chain import decompose_chain
from .._tracer.probe import probe_curves_then_matrix, probe_matrix_then_curves
from .._tracer.tracing import TraceAbort, trace_function
from .structure import chain_as_curves_matrix, chain_as_matrix_curves
from .verify import agrees, call

N_VERIFY = 4096


def normalized_functions(colorspace, domain):
    """`to_xyz`/`from_xyz` of `colorspace` with device values mapped from `domain` to [0, 1]."""
    lo = np.array([d[0] for d in domain])
    span = np.array([d[1] - d[0] for d in domain])
    if np.allclose(lo, 0.0) and np.allclose(span, 1.0):
        return colorspace.to_xyz, colorspace.from_xyz

    def fwd(d01):
        return colorspace.to_xyz(d01 * span + lo)

    def bwd(xyz):
        return (colorspace.from_xyz(xyz) - lo) / span

    return fwd, bwd


class Context:
    """Everything the tier writers share about one space being written."""

    def __init__(self, colorspace, version_major: int, grid_size):
        self.n = colorspace.n_channels
        self.major = version_major
        self.grid_size = grid_size
        self.domain = list(colorspace.device_domain)
        self.fwd, self.bwd = normalized_functions(colorspace, self.domain)
        circular = colorspace.circular_device_channels
        self.periods = {i: 1.0 / (self.domain[i][1] - self.domain[i][0]) for i in circular}
        rng = np.random.default_rng(0)
        self.device_samples = rng.random((N_VERIFY, self.n))
        self.xyz_samples = call(self.fwd, self.device_samples)
        if self.xyz_samples.shape != (N_VERIFY, 3):
            raise ICCWriteError(
                f"to_xyz returned shape {self.xyz_samples.shape[1:]} per pixel, expected (3,)"
            )
        finite = np.all(np.isfinite(self.xyz_samples), axis=-1)
        if finite.sum() < 16:
            raise ICCWriteError("to_xyz returns almost no finite values on the device domain")
        self.xyz_finite = self.xyz_samples[finite]
        self._fwd_root = self._bwd_root = None
        self._traced = False

    def _trace(self):
        if self._traced:
            return
        self._traced = True
        self._fwd_root = trace_root(self.fwd, self.n, 3)
        self._bwd_root = trace_root(self.bwd, 3, self.n)

    @property
    def fwd_root(self):
        self._trace()
        return self._fwd_root

    @property
    def bwd_root(self):
        self._trace()
        return self._bwd_root

    def forward_curves_matrix(self):
        """``(curves, matrix, offset)`` with ``fwd(x) == c(x) @ matrix + offset``, or None."""
        found = None
        if self.fwd_root is not None:
            found = chain_as_curves_matrix(decompose_chain(self.fwd_root, self.n, 3), self.n)
        if found is not None and self._forward_ok(*found):
            return found
        found = probe_curves_then_matrix(self.fwd, self.n, 3)
        if found is not None and self._forward_ok(*found):
            return found
        return None

    def _forward_ok(self, curves, matrix, offset) -> bool:
        x = self.device_samples
        pred = np.stack([call(c, x[:, i]) for i, c in enumerate(curves)], axis=-1) @ matrix
        return agrees(self.xyz_samples, pred + offset)

    def backward_matrix_curves(self, hint=None):
        """``(matrix, offset, curves, ranges)`` with ``bwd(x) == c(x @ matrix + offset)``."""
        if self.bwd_root is not None:
            chain = decompose_chain(self.bwd_root, 3, self.n)
            found = chain_as_matrix_curves(chain, self.n)
            if found is not None:
                matrix, offset, curves = found
                u = self.xyz_finite @ matrix + offset
                ranges = [(float(u[:, j].min()), float(u[:, j].max())) for j in range(self.n)]
                if self._backward_ok(matrix, offset, curves):
                    return matrix, offset, curves, ranges
        probed = probe_matrix_then_curves(self.bwd, 3, self.n, self.xyz_finite, matrix_hint=hint)
        if probed is not None:
            matrix, curves, ranges = probed
            offset = np.zeros(self.n)
            if self._backward_ok(matrix, offset, curves):
                return matrix, offset, curves, ranges
        return None

    def _backward_ok(self, matrix, offset, curves) -> bool:
        x = self.xyz_finite
        u = x @ matrix + offset
        pred = np.stack([call(c, u[:, j]) for j, c in enumerate(curves)], axis=-1)
        return self.backward_agrees(call(self.bwd, x), pred)

    def inverts_forward(self) -> bool:
        """Whether `bwd` undoes `fwd` on the device samples."""
        finite = np.all(np.isfinite(self.xyz_samples), axis=-1)
        device = self.device_samples[finite]
        return self.backward_agrees(device, call(self.bwd, self.xyz_samples[finite]))

    def backward_agrees(self, expected: np.ndarray, actual: np.ndarray) -> bool:
        """Whether device values agree directly or map to the same XYZ."""
        if agrees(expected, actual, self.periods or None):
            return True
        return agrees(call(self.fwd, expected), call(self.fwd, actual))


def trace_root(fn, n_in: int, n_out: int):
    """Traced graph of `fn` with `n_out` output channels, or None."""
    try:
        node, out_ch = trace_function(fn, n_in)
    except TraceAbort:
        return None
    return node if out_ch == n_out else None
