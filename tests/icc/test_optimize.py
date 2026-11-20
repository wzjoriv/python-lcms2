"""Graph optimization of traced expressions."""

import numpy as np
import pytest

from lcms2.colorimetry import D50
from lcms2.icc._tracer.chain import decompose_chain
from lcms2.icc._tracer.tracing import (
    TracedArray,
    _PatchArrayConstructors,
    trace_function,
)
from lcms2.spaces import ColorSpace


def _trace_unoptimized(fn, n_channels):
    x = TracedArray.input(n_channels)
    with _PatchArrayConstructors():
        y = fn(x)
    return y.node, y.n_channels


def test_constant_subgraph_folds_to_single_const():
    def fn(x):
        c = (2.0 + 3.0) * 4.0 - 1.0
        return x * c

    node, _ = trace_function(fn, 1)
    const_args = [a for a in node.args if a.op == "const"]
    assert len(const_args) == 1
    assert float(np.asarray(const_args[0].meta["value"])) == pytest.approx(19.0)


def test_fully_constant_expression_folds_to_bare_const():
    def fn(x):
        return x * 0.0 + (1.0 + 1.0)

    node, _ = trace_function(fn, 1)
    assert node.op == "const"
    assert float(np.asarray(node.meta["value"])) == pytest.approx(2.0)


def test_identical_subexpressions_share_one_node_object():
    def fn(x):
        a = x[..., 0] * 2.0 + x[..., 1] * 3.0
        b = x[..., 0] * 2.0 + x[..., 1] * 3.0
        return a + b

    node, _ = trace_function(fn, 2)
    left, right = node.args
    assert left is right


def test_duplicate_channel_select_is_deduplicated():
    def fn(x):
        g = x[..., 0]
        h = x[..., 0]
        return g + h

    node, _ = trace_function(fn, 1)
    a, b = node.args
    assert a is b
    assert a.op == "select" and a.meta["index"] == 0


def test_add_zero_and_mul_one_and_double_negation_simplify_away():
    def fn(x):
        a = x[..., 0] * 1.0 + 0.0
        b = -(-x[..., 1])  # noqa: B002
        return a + b * 0.0

    node, _ = trace_function(fn, 2)
    assert node.op == "select" and node.meta["index"] == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimizer_preserves_numeric_output_on_random_inputs(seed):
    rng = np.random.default_rng(seed)

    def messy(x):
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        r2 = r * 1.0 + 0.0
        g2 = -(-g)  # noqa: B002
        b2 = (b + b) / 2.0
        same_a = 0.2 * r + 0.5 * g + 0.3 * b
        same_b = 0.3 * b + 0.2 * r + 0.5 * g
        return np.stack([r2 + same_a * 0.0, g2, b2 + (same_b - same_a)], axis=-1)

    x = rng.uniform(-2, 2, size=(50, 3))
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    r2 = r * 1.0 + 0.0
    g2 = -(-g)  # noqa: B002
    b2 = (b + b) / 2.0
    same_a = 0.2 * r + 0.5 * g + 0.3 * b
    same_b = 0.3 * b + 0.2 * r + 0.5 * g
    expected = np.stack([r2 + same_a * 0.0, g2, b2 + (same_b - same_a)], axis=-1)

    cs = ColorSpace(n_channels=3, to_xyz=messy, from_xyz=lambda xyz: xyz)
    got = cs.to_xyz(x)
    assert np.allclose(got, expected)

    node_opt, n_out = trace_function(messy, 3)
    chain = decompose_chain(node_opt, 3, n_out)
    assert chain is not None
    result = x
    for stage in chain:
        if stage[0] == "matrix":
            _, matrix, offset = stage
            result = result @ matrix + offset
        else:
            _, curve_fns = stage
            result = np.stack([f(result[..., i]) for i, f in enumerate(curve_fns)], axis=-1)
    assert np.allclose(result, expected, atol=1e-9)


def _messy_gray_to_xyz(gray):
    gray = np.asarray(gray, dtype=np.float64)
    g = gray[..., 0]
    h = gray[..., 0]
    y = np.power(np.clip((g + h) / 2.0, 0.0, 1.0), 2.2)
    lin = np.stack([y], axis=-1)
    return lin @ D50.reshape(1, 3)


def _messy_gray_from_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    yv = np.clip(xyz[..., 1] / D50[1], 0.0, None)
    g = np.power(yv, 1.0 / 2.2) * 1.0 + 0.0
    return np.stack([g], axis=-1)


def test_messy_space_decomposes_only_with_optimizer():
    node_raw, n_out = _trace_unoptimized(_messy_gray_to_xyz, 1)
    assert decompose_chain(node_raw, 1, n_out) is None

    node_opt, n_out2 = trace_function(_messy_gray_to_xyz, 1)
    chain = decompose_chain(node_opt, 1, n_out2)
    assert chain is not None


def test_messy_space_write_method_improves_with_optimizer():
    cs = ColorSpace(n_channels=1, to_xyz=_messy_gray_to_xyz, from_xyz=_messy_gray_from_xyz)
    result = cs.to_icc_profile()
    assert result["method"] == "exact_gray_trc"

    x = np.linspace(0.01, 1.0, 25).reshape(-1, 1)
    import os
    import tempfile

    from lcms2.spaces import ICCSpace

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "messy_gray.icc")
        cs.to_icc_profile(path)
        loaded = ICCSpace(path)
        xyz_direct = cs.to_xyz(x)
        xyz_loaded = loaded.to_xyz(x)
        assert np.allclose(xyz_direct, xyz_loaded, atol=1e-3)
