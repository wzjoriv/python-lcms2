"""Shared state for writing one color space: domain, samples and structure."""

from __future__ import annotations

import numpy as np

from ..._errors import ICCWriteError
from .._tracer.chain import decompose_chain
from .._tracer.probe import probe_curves_then_matrix, probe_matrix_then_curves
from .._tracer.tracing import TraceAbort, TracedArray, trace_function
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


class _Pixelwise:
    """`fn` that also works for code written for a single pixel.

    Functions that call ``float(x[0])``, ``.item()``, iterate over the channels or use another
    non-NumPy library fail (or return the wrong shape) on a batch of pixels; those are evaluated
    one pixel at a time instead, as are functions whose result for a pixel depends on the other
    pixels of the batch (``x / x.max()``). Traced placeholders pass through untouched.
    """

    def __init__(self, fn, n_in: int, n_out: int, name: str):
        self.fn, self.n_in, self.n_out, self.name = fn, n_in, n_out, name
        self.rows = False

    def __call__(self, x):
        if isinstance(x, TracedArray):
            return self.fn(x)
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape[:-1] + (self.n_out,)
        if not self.rows:
            try:
                y = _real(self.fn(x))
                if y.shape == shape:
                    return y
            except Exception:  # noqa: BLE001 - retried one pixel at a time below
                pass
            self.rows = True
        out, failed, first_error = self._by_rows(x.reshape(-1, self.n_in))
        if failed.size and failed.all():
            raise ICCWriteError(
                f"{self.name} failed: {type(first_error).__name__}: {first_error}"
            ) from first_error
        return out.reshape(shape)

    def _by_rows(self, flat):
        """``(values, failed mask, first error)``; pixels the function rejects are NaN."""
        rows, failed, first_error = [], [], None
        for r in flat:
            try:
                rows.append(_real(self.fn(r)).reshape(self.n_out))
                failed.append(False)
            except Exception as e:  # noqa: BLE001 - a pixel the function rejects is not a value
                first_error = first_error or e
                rows.append(np.full(self.n_out, np.nan))
                failed.append(True)
        out = np.stack(rows) if rows else np.empty((0, self.n_out))
        return out, np.array(failed, dtype=bool), first_error

    def depends_on_batch(self, sample) -> bool:
        """Whether results for `sample` differ from those pixel by pixel; switches to pixels if so."""
        if self.rows:
            return False
        batch = call(self, sample)
        if self.rows:  # the batch call itself needed pixels one at a time
            return False
        rows, failed, _ = self._by_rows(sample)
        ok = ~failed
        if not ok.any() or np.allclose(batch[ok], rows[ok], rtol=1e-9, atol=1e-12, equal_nan=True):
            return False
        self.rows = True
        return True


def _real(y) -> np.ndarray:
    """`y` as float64, complex values (a root of a negative number) becoming NaN."""
    y = np.asarray(y)
    if np.iscomplexobj(y):
        y = np.where(y.imag == 0.0, y.real, np.nan)
    return np.asarray(y, dtype=np.float64)


_BATCH_REASON = (
    "{name} gives a pixel different values depending on the other pixels it is called with "
    "(a reduction such as x.max() or x.mean() over all pixels)"
)


class Context:
    """Everything the tier writers share about one space being written."""

    def __init__(self, colorspace, version_major: int, grid_size):
        self.n = colorspace.n_channels
        self.major = version_major
        self.grid_size = grid_size
        self.domain = list(colorspace.device_domain)
        fwd, bwd = normalized_functions(colorspace, self.domain)
        self.fwd = _Pixelwise(fwd, self.n, 3, "to_xyz")
        self.bwd = _Pixelwise(bwd, 3, self.n, "from_xyz")
        #: why a direction could not be traced, keyed by ``'to_xyz'`` / ``'from_xyz'``
        self.trace_failures: dict[str, str] = {}
        circular = colorspace.circular_device_channels
        self.periods = {i: 1.0 / (self.domain[i][1] - self.domain[i][0]) for i in circular}
        rng = np.random.default_rng(0)
        self.device_samples = rng.random((N_VERIFY, self.n))
        self._sample()
        self._unstable = self._unstable_directions()
        for name, fn, sample in (
            ("to_xyz", self.fwd, self.device_samples[:24]),
            ("from_xyz", self.bwd, self.xyz_finite[:24]),
        ):
            if name not in self._unstable and fn.depends_on_batch(sample):
                self._unstable[name] = _BATCH_REASON.format(name=name)
        if self.fwd.rows:  # the batch results above were not the per-pixel ones
            self._sample()
        self.deterministic = not self._unstable
        self._fwd_root = self._bwd_root = None
        self._traced = False

    def _sample(self) -> None:
        self.xyz_samples = call(self.fwd, self.device_samples)
        if self.xyz_samples.shape != (N_VERIFY, 3):
            raise ICCWriteError(
                f"to_xyz returned shape {self.xyz_samples.shape[1:]} per pixel, expected (3,)"
            )
        finite = np.all(np.isfinite(self.xyz_samples), axis=-1)
        if finite.sum() < 16:
            raise ICCWriteError("to_xyz returns almost no finite values on the device domain")
        self.xyz_finite = self.xyz_samples[finite]

    def _unstable_directions(self) -> dict:
        """Directions that give different results when called twice (random numbers, state)."""
        probes = (
            (self.fwd, self.device_samples[:64], "to_xyz"),
            (self.bwd, self.xyz_finite[:64], "from_xyz"),
        )
        unstable = {}
        for fn, x, name in probes:
            if not np.array_equal(call(fn, x), call(fn, x), equal_nan=True):
                unstable[name] = (
                    f"{name} gives different results when called twice with the same values "
                    "(random numbers or other hidden state)"
                )
        return unstable

    def _trace(self):
        if self._traced:
            return
        self._traced = True
        self._fwd_root = trace_root(self.fwd, self.n, 3, "to_xyz", self.trace_failures)
        self._bwd_root = trace_root(self.bwd, 3, self.n, "from_xyz", self.trace_failures)
        for name, reason in self._unstable.items():
            self.trace_failures.setdefault(name, reason)
        if self._unstable:
            self._fwd_root = self._bwd_root = None

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


def trace_root(fn, n_in: int, n_out: int, name: str = "function", failures: dict | None = None):
    """Traced graph of `fn` with `n_out` output channels, or None.

    The reason for a failure is stored in `failures` under `name`.
    """
    try:
        node, out_ch = trace_function(fn, n_in)
    except TraceAbort as e:
        if failures is not None:
            failures[name] = f"{name} is not traceable: {e}"
        return None
    if out_ch != n_out:
        if failures is not None:
            failures[name] = f"{name} returns {out_ch} channels, expected {n_out}"
        return None
    return node
