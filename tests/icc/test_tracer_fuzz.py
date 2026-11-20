"""Seeded random colour spaces: random matrices and per-channel curves, written the many ways users write
them. Each must become an exact (not sampled) profile at v4 and v5 that LittleCMS and the pure-Python
reader evaluate back to the NumPy functions."""

from __future__ import annotations

import numpy as np
import pytest

import lcms2
from lcms2.icc import ICCSpace
from lcms2.spaces import ColorSpace, SRGBSpace

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

M = SRGBSpace.matrix


def _curve(rng):
    """A (decode, encode) pair of monotonic curves on 0..1, in one of several user styles."""
    kind = rng.choice(["power", "srgb_where", "power_np", "linear"])
    g = float(rng.uniform(1.6, 2.8))
    if kind == "power":
        return kind, (lambda x: x**g), (lambda y: y ** (1 / g))
    if kind == "power_np":
        return (
            kind,
            (lambda x: np.power(np.clip(x, 0, None), g)),
            (lambda y: np.power(np.clip(y, 0, None), 1 / g)),
        )
    if kind == "srgb_where":
        a, k = 0.055, 12.92
        return (
            kind,
            lambda x: np.where(x <= 0.04045, x / k, ((x + a) / (1 + a)) ** 2.4),
            lambda y: np.where(y <= 0.0031308, y * k, (1 + a) * y ** (1 / 2.4) - a),
        )
    return kind, (lambda x: x * 1.0), (lambda y: y * 1.0)


def _random_space(seed):
    rng = np.random.default_rng(seed)
    m = M @ (np.eye(3) + rng.uniform(-0.15, 0.15, (3, 3)) * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    m_inv = np.linalg.inv(m)
    shared = rng.random() < 0.5
    curves = [_curve(rng)] * 3 if shared else [_curve(rng) for _ in range(3)]
    style = rng.choice(["matmul", "einsum", "dot"])

    def mat(x, a):
        if style == "matmul":
            return x @ a.T
        if style == "einsum":
            return np.einsum("ij,...j->...i", a, x)
        return np.dot(x, a.T)

    def to_xyz(rgb):
        rgb = np.asarray(rgb, dtype=np.float64)
        lin = np.stack([curves[i][1](rgb[..., i]) for i in range(3)], axis=-1)
        return mat(lin, m)

    def from_xyz(xyz):
        lin = mat(np.asarray(xyz, dtype=np.float64), m_inv)
        return np.stack([curves[i][2](lin[..., i]) for i in range(3)], axis=-1)

    kinds = "/".join(c[0] for c in curves)
    return ColorSpace(n_channels=3, to_xyz=to_xyz, from_xyz=from_xyz, name=f"fuzz{seed}"), f"{style}:{kinds}"


SEEDS = range(40)
GRID = np.stack(np.meshgrid(*[np.linspace(0, 1, 7)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("version", [4, 5])
def test_random_space_is_exact_and_evaluates_back(seed, version):
    space, desc = _random_space(seed)
    result = space.to_icc_profile(version=version, strict=True)
    assert result["method"] != "clut_fallback", (desc, result["notes"])
    want = space.to_xyz(GRID)
    back = ICCSpace(result["bytes"])
    assert np.max(np.abs(back.to_xyz(GRID) - want)) < 2e-3, desc
    if version == 4:
        got = lcms2.convert(GRID, lcms2.Profile(result["bytes"]), "XYZ", intent="relative")
        assert np.max(np.abs(got - want)) < 2e-3, desc
