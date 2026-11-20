"""numpy operations supported by the tracer."""

import numpy as np
import pytest

from lcms2.icc._tracer.chain import _compile_scalar_expr, _find_unique_select
from lcms2.icc._tracer.tracing import TraceAbort, trace_function


def _trace_and_eval_scalar(fn, x_val: float) -> float:
    node, out_ch = trace_function(fn, 1)
    assert out_ch == 1
    assert node.op == "stack" and len(node.args) == 1
    inner = node.args[0]
    select_node = _find_unique_select(inner)
    assert select_node is not None
    compiled = _compile_scalar_expr(inner, select_node)
    return float(compiled(np.array([x_val]))[0])


@pytest.mark.parametrize(
    "fn,x,expected",
    [
        (lambda x: np.stack([np.sign(x[..., 0])], axis=-1), -3.0, -1.0),
        (lambda x: np.stack([np.square(x[..., 0])], axis=-1), 4.0, 16.0),
        (lambda x: np.stack([x[..., 0] % 3.0], axis=-1), 5.0, 2.0),
        (lambda x: np.stack([x[..., 0] // 2.0], axis=-1), 5.0, 2.0),
        (lambda x: np.stack([np.floor(x[..., 0])], axis=-1), 2.7, 2.0),
        (lambda x: np.stack([np.ceil(x[..., 0])], axis=-1), 2.1, 3.0),
        (lambda x: np.stack([np.sin(x[..., 0])], axis=-1), 0.0, 0.0),
        (lambda x: np.stack([np.cos(x[..., 0])], axis=-1), 0.0, 1.0),
        (lambda x: np.stack([np.log2(x[..., 0])], axis=-1), 8.0, 3.0),
        (lambda x: np.stack([np.log10(x[..., 0])], axis=-1), 100.0, 2.0),
        (lambda x: np.stack([np.expm1(x[..., 0])], axis=-1), 0.0, 0.0),
        (lambda x: np.stack([np.hypot(x[..., 0], 4.0)], axis=-1), 3.0, 5.0),
        (lambda x: np.stack([np.arctan2(x[..., 0], 1.0)], axis=-1), 0.0, 0.0),
        (lambda x: np.stack([+x[..., 0]], axis=-1), 5.0, 5.0),
        (
            lambda x: np.stack([np.equal(x[..., 0], 1.0).astype(float) * 9.0 + 1.0], axis=-1),
            1.0,
            10.0,
        ),
    ],
)
def test_new_ufuncs_trace_and_evaluate_correctly(fn, x, expected):
    result = _trace_and_eval_scalar(fn, x)
    assert abs(result - expected) < 1e-9


def test_iterating_over_a_traced_array_is_refused():
    """On a batch ``for c in x`` walks over pixels, so it cannot mean "over channels"."""

    def fn(x):
        chans = [c for c in x]  # noqa: C416 - iterates a traced array
        return np.stack([c * 2.0 for c in chans], axis=-1)

    with pytest.raises(TraceAbort, match="iterating"):
        trace_function(fn, 3)


def test_channel_first_view_can_be_unpacked():
    def fn(x):
        r, g, b = np.moveaxis(x, -1, 0)
        return np.stack([b * 2.0, g, r], axis=-1)

    node, out_ch = trace_function(fn, 3)
    assert out_ch == 3


def test_bool_on_traced_array_raises_clear_error_not_silent_or_obscure():
    def fn(x):
        if x[..., 0] > 0.5:
            return x
        return x * 2

    with pytest.raises(TraceAbort) as excinfo:
        trace_function(fn, 3)
    msg = str(excinfo.value)
    assert "np.where" in msg
    assert "branch" in msg or "truth value" in msg


def test_while_loop_on_traced_array_raises_clear_error():
    def fn(x):
        y = x
        while y[..., 0] > 0:
            y = y - 1
        return y

    with pytest.raises(TraceAbort):
        trace_function(fn, 3)


def test_math_module_function_raises_clear_error_not_an_opaque_typeerror():
    import math

    def fn(x):
        return np.stack([math.sqrt(x[..., 0])], axis=-1)

    with pytest.raises(TraceAbort) as excinfo:
        trace_function(fn, 1)
    msg = str(excinfo.value)
    assert "math" in msg and "np.sqrt" in msg


def test_builtin_max_on_traced_values_hits_the_bool_trap_safely():

    def fn(x):
        m = max(x[..., 0], x[..., 1])
        return np.stack([m], axis=-1)

    with pytest.raises(TraceAbort) as excinfo:
        trace_function(fn, 2)
    assert "truth value" in str(excinfo.value) or "branch" in str(excinfo.value)


def test_cross_channel_max_traces_as_a_reduce_node_not_a_matrix_op():

    def fn(x):
        m = x.max(axis=-1)
        return np.stack([m, m, m], axis=-1)

    node, out_ch = trace_function(fn, 3)
    assert out_ch == 3
    assert node.op == "stack"
    assert node.args[0].op == "reduce_max"
    from lcms2.icc._tracer.calc import compile_calc_program
    from lcms2.icc._tracer.chain import decompose_chain

    assert decompose_chain(node, 3, 3) is None
    ops = compile_calc_program(node, 3, 3)
    from lcms2.icc._io.mpe import eval_calc_element

    x = np.array([[0.2, 0.9, 0.5]])
    got = eval_calc_element({"ops": ops}, x, 3)
    assert np.allclose(got, np.max(x, axis=-1, keepdims=True))


def test_np_max_free_function_also_traces():
    def fn(x):
        m = np.max(x, axis=-1)
        return np.stack([m], axis=-1)

    node, out_ch = trace_function(fn, 3)
    assert out_ch == 1
    assert node.args[0].op == "reduce_max"


def test_len_on_traced_array_raises_clear_error():
    def fn(x):
        n = len(x)
        return x * n

    with pytest.raises(TraceAbort):
        trace_function(fn, 3)


def test_mean_over_channels_is_linear_and_traces_fine():

    def fn(x):
        m = x.mean(axis=-1)
        return np.stack([m, m, m], axis=-1)

    node, out_ch = trace_function(fn, 3)
    assert out_ch == 3


def test_ones_like_and_zeros_like_supported():

    def fn(x):
        u = x[..., 0]
        v = x[..., 1]
        return np.stack([u, np.ones_like(u), np.zeros_like(v), v], axis=-1)

    node, out_ch = trace_function(fn, 2)
    assert out_ch == 4


def test_implicit_linear_stack_recognized_as_matrix_not_clut():
    from lcms2.icc._tracer.chain import decompose_chain

    def fn(x):
        u = x[..., 0]
        v = x[..., 1]
        return np.stack([u, np.ones_like(u), v], axis=-1)

    node, out_ch = trace_function(fn, 2)
    assert out_ch == 3
    chain = decompose_chain(node, 2, 3)
    assert chain is not None
    assert len(chain) == 1 and chain[0][0] == "matrix"
    matrix, offset = chain[0][1], chain[0][2]
    np.testing.assert_allclose(matrix, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(offset, [0.0, 1.0, 0.0])


def test_np_select_piecewise_curve_traces_and_evaluates_correctly():

    def fn(x):
        v = x[..., 0]
        y = np.select([v < 0.3, v < 0.7], [v * 2.0, v + 1.0], default=v * 3.0)
        return np.stack([y], axis=-1)

    assert _trace_and_eval_scalar(fn, 0.1) == pytest.approx(0.2)
    assert _trace_and_eval_scalar(fn, 0.5) == pytest.approx(1.5)
    assert _trace_and_eval_scalar(fn, 0.9) == pytest.approx(2.7)


def test_np_dot_and_dot_method_supported():
    def fn(x):
        m = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        return np.dot(x, m)

    node, out_ch = trace_function(fn, 3)
    assert out_ch == 2

    def fn2(x):
        m = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        return x.dot(m)

    node2, out_ch2 = trace_function(fn2, 3)
    assert out_ch2 == 2


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) @ x,
        lambda x: np.dot(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), x),
    ],
)
def test_left_matmul_and_left_dot_supported(fn):
    node, out_ch = trace_function(fn, 3)
    assert out_ch == 2


def test_dot_of_two_traced_arrays_raises_clear_error_bilinear():

    def fn(x):
        return np.dot(x, x)

    with pytest.raises(TraceAbort) as excinfo:
        trace_function(fn, 3)
    assert "bilinear" in str(excinfo.value)


def test_np_interp_lookup_table_curve_traces_and_evaluates_correctly():

    def fn(x):
        v = x[..., 0]
        y = np.interp(v, [0.0, 0.5, 1.0], [0.0, 0.3, 1.0])
        return np.stack([y], axis=-1)

    assert _trace_and_eval_scalar(fn, 0.0) == pytest.approx(0.0)
    assert _trace_and_eval_scalar(fn, 0.5) == pytest.approx(0.3)
    assert _trace_and_eval_scalar(fn, 0.25) == pytest.approx(0.15)
    assert _trace_and_eval_scalar(fn, 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fn,x,expected",
    [
        (lambda x: np.stack([np.sinh(x[..., 0])], axis=-1), 0.0, 0.0),
        (lambda x: np.stack([np.cosh(x[..., 0])], axis=-1), 0.0, 1.0),
        (lambda x: np.stack([np.tanh(x[..., 0])], axis=-1), 0.0, 0.0),
        (lambda x: np.stack([np.deg2rad(x[..., 0])], axis=-1), 180.0, np.pi),
        (lambda x: np.stack([np.rad2deg(x[..., 0])], axis=-1), np.pi, 180.0),
        (lambda x: np.stack([np.fmax(x[..., 0], 2.0)], axis=-1), 1.0, 2.0),
        (lambda x: np.stack([np.fmin(x[..., 0], 2.0)], axis=-1), 5.0, 2.0),
        (lambda x: np.stack([np.copysign(x[..., 0], -1.0)], axis=-1), 3.0, -3.0),
    ],
)
def test_more_new_ufuncs_trace_and_evaluate_correctly(fn, x, expected):
    result = _trace_and_eval_scalar(fn, x)
    assert abs(result - expected) < 1e-9


def test_broadcast_and_shape_noops_supported():
    def fn(x):
        y = np.broadcast_to(x[..., 0], (1,))
        y = np.expand_dims(y, axis=-1)
        y = np.squeeze(y)
        return np.stack([y, x[..., 1]], axis=-1)

    node, out_ch = trace_function(fn, 3)
    assert out_ch == 2


def test_np_concatenate_along_last_axis_supported():
    def fn(x):
        a = x[..., 0:1] if False else np.stack([x[..., 0]], axis=-1)
        b = np.stack([x[..., 1]], axis=-1)
        return np.concatenate([a, b], axis=-1)

    node, out_ch = trace_function(fn, 3)
    assert out_ch == 2


def test_np_round_traces():
    node, width = trace_function(lambda x: np.stack([np.round(x[..., 0] * 255) / 255], -1), 1)
    assert width == 1
