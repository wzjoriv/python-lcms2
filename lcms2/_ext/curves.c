/* Tone curve bindings: construction, evaluation and analysis of cmsToneCurve handles. */
#include "common.h"

#include <string.h>

/* Arrays with fewer elements than this are evaluated without releasing the GIL. */
#define CURVE_NOGIL_THRESHOLD 4096

static int
expected_param_count(int type)
{
    switch (type < 0 ? -type : type) {
    case 1: return 1;
    case 2: return 3;
    case 3: return 4;
    case 4: return 5;
    case 5: return 7;
    case 6: return 4;
    case 7: return 5;
    case 8: return 5;
    case 108: return 1;
    case 109: return 1;
    default: return -1;
    }
}

static PyObject *
py_curve_gamma(PyObject *self, PyObject *args)
{
    double g;
    (void)self;
    if (!PyArg_ParseTuple(args, "d", &g))
        return NULL;
    lcms2_clear_error();
    return lcms2_wrap_curve(cmsBuildGamma(NULL, g));
}

static PyObject *
py_curve_parametric(PyObject *self, PyObject *args)
{
    int type;
    PyObject *seq;
    (void)self;
    if (!PyArg_ParseTuple(args, "iO", &type, &seq))
        return NULL;
    int want = expected_param_count(type);
    if (want < 0) {
        PyErr_Format(PyExc_ValueError,
                     "unsupported parametric curve type %d; use 1-8, 108 or 109 (negative for the inverse)",
                     type);
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROMANY(seq, NPY_DOUBLE, 0, 1,
                                                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (arr == NULL)
        return NULL;
    if (PyArray_SIZE(arr) != want) {
        PyErr_Format(PyExc_ValueError, "parametric curve type %d takes %d parameter(s), got %zd", type,
                     want, (Py_ssize_t)PyArray_SIZE(arr));
        Py_DECREF(arr);
        return NULL;
    }
    double params[10] = {0};
    memcpy(params, PyArray_DATA(arr), (size_t)want * sizeof(double));
    Py_DECREF(arr);
    lcms2_clear_error();
    return lcms2_wrap_curve(cmsBuildParametricToneCurve(NULL, type, params));
}

static PyObject *
py_curve_tabulated(PyObject *self, PyObject *arg)
{
    (void)self;
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROMANY(arg, NPY_NOTYPE, 1, 1,
                                                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (arr == NULL)
        return NULL;
    npy_intp n = PyArray_SIZE(arr);
    if (n < 2 || n > 65535) {
        PyErr_Format(PyExc_ValueError, "a tabulated curve needs between 2 and 65535 values, got %zd",
                     (Py_ssize_t)n);
        Py_DECREF(arr);
        return NULL;
    }
    cmsToneCurve *c;
    lcms2_clear_error();
    if (PyArray_TYPE(arr) == NPY_UINT16) {
        c = cmsBuildTabulatedToneCurve16(NULL, (cmsUInt32Number)n,
                                         (const cmsUInt16Number *)PyArray_DATA(arr));
    }
    else if (PyArray_ISFLOAT(arr) || PyArray_ISINTEGER(arr)) {
        PyArrayObject *conv = (PyArrayObject *)PyArray_FROMANY(
            (PyObject *)arr, NPY_FLOAT32, 1, 1,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
        if (conv == NULL) {
            Py_DECREF(arr);
            return NULL;
        }
        c = cmsBuildTabulatedToneCurveFloat(NULL, (cmsUInt32Number)n,
                                            (const cmsFloat32Number *)PyArray_DATA(conv));
        Py_DECREF(conv);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "tabulated curve values must be a uint16 or floating-point array");
        Py_DECREF(arr);
        return NULL;
    }
    Py_DECREF(arr);
    return lcms2_wrap_curve(c);
}

/* Evaluate `curve` over a float64 or uint16 array; `is16` selects the entry point. */
static PyObject *
curve_eval_impl(PyObject *args, int is16)
{
    PyObject *hobj, *xobj;
    if (!PyArg_ParseTuple(args, "OO", &hobj, &xobj))
        return NULL;
    cmsToneCurve *c = lcms2_get_curve(hobj);
    if (c == NULL)
        return NULL;
    int tp = is16 ? NPY_UINT16 : NPY_DOUBLE;
    PyArrayObject *in = (PyArrayObject *)PyArray_FROMANY(xobj, tp, 0, 0,
                                                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (in == NULL)
        return NULL;
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(in), PyArray_DIMS(in), tp);
    if (out == NULL) {
        Py_DECREF(in);
        return NULL;
    }
    npy_intp n = PyArray_SIZE(in), i;
    PyThreadState *ts = NULL;
    if (n >= CURVE_NOGIL_THRESHOLD)
        ts = PyEval_SaveThread();
    if (is16) {
        const cmsUInt16Number *s = PyArray_DATA(in);
        cmsUInt16Number *d = PyArray_DATA(out);
        for (i = 0; i < n; i++)
            d[i] = cmsEvalToneCurve16(c, s[i]);
    }
    else {
        const double *s = PyArray_DATA(in);
        double *d = PyArray_DATA(out);
        for (i = 0; i < n; i++)
            d[i] = (double)cmsEvalToneCurveFloat(c, (cmsFloat32Number)s[i]);
    }
    if (ts != NULL)
        PyEval_RestoreThread(ts);
    Py_DECREF(in);
    return (PyObject *)out;
}

static PyObject *
py_curve_eval(PyObject *self, PyObject *args)
{
    (void)self;
    return curve_eval_impl(args, 0);
}

static PyObject *
py_curve_eval16(PyObject *self, PyObject *args)
{
    (void)self;
    return curve_eval_impl(args, 1);
}

static PyObject *
py_curve_reverse(PyObject *self, PyObject *args)
{
    PyObject *hobj;
    int n = 4096;
    (void)self;
    if (!PyArg_ParseTuple(args, "O|i", &hobj, &n))
        return NULL;
    cmsToneCurve *c = lcms2_get_curve(hobj);
    if (c == NULL)
        return NULL;
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "inverse resolution n must be at least 2");
        return NULL;
    }
    lcms2_clear_error();
    return lcms2_wrap_curve(cmsReverseToneCurveEx((cmsUInt32Number)n, c));
}

static PyObject *
py_curve_join(PyObject *self, PyObject *args)
{
    PyObject *aobj, *bobj;
    int n = 4096;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO|i", &aobj, &bobj, &n))
        return NULL;
    cmsToneCurve *a = lcms2_get_curve(aobj);
    cmsToneCurve *b = a ? lcms2_get_curve(bobj) : NULL;
    if (a == NULL || b == NULL)
        return NULL;
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "join resolution n must be at least 2");
        return NULL;
    }
    lcms2_clear_error();
    return lcms2_wrap_curve(cmsJoinToneCurve(NULL, a, b, (cmsUInt32Number)n));
}

static PyObject *
py_curve_smooth(PyObject *self, PyObject *args)
{
    PyObject *hobj;
    double lambda;
    (void)self;
    if (!PyArg_ParseTuple(args, "Od", &hobj, &lambda))
        return NULL;
    cmsToneCurve *c = lcms2_get_curve(hobj);
    if (c == NULL)
        return NULL;
    lcms2_clear_error();
    cmsToneCurve *copy = cmsDupToneCurve(c);
    if (copy == NULL)
        return lcms2_raise("could not copy the tone curve");
    if (!cmsSmoothToneCurve(copy, lambda)) {
        cmsFreeToneCurve(copy);
        return lcms2_raise("the curve could not be smoothed (it needs at least 6 table entries)");
    }
    return lcms2_wrap_curve(copy);
}

static PyObject *
py_curve_copy(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsToneCurve *c = lcms2_get_curve(arg);
    if (c == NULL)
        return NULL;
    lcms2_clear_error();
    return lcms2_wrap_curve(cmsDupToneCurve(c));
}

static PyObject *
py_curve_estimate_gamma(PyObject *self, PyObject *args)
{
    PyObject *hobj;
    double precision = 0.01;
    (void)self;
    if (!PyArg_ParseTuple(args, "O|d", &hobj, &precision))
        return NULL;
    cmsToneCurve *c = lcms2_get_curve(hobj);
    if (c == NULL)
        return NULL;
    double g = cmsEstimateGamma(c, precision);
    if (g < 0)
        return lcms2_raise("the gamma could not be estimated within the requested precision; "
                           "try a larger precision value");
    return PyFloat_FromDouble(g);
}

static PyObject *
py_curve_table(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsToneCurve *c = lcms2_get_curve(arg);
    if (c == NULL)
        return NULL;
    npy_intp n = (npy_intp)cmsGetToneCurveEstimatedTableEntries(c);
    const cmsUInt16Number *t = cmsGetToneCurveEstimatedTable(c);
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, &n, NPY_UINT16);
    if (out == NULL)
        return NULL;
    memcpy(PyArray_DATA(out), t, (size_t)n * sizeof(cmsUInt16Number));
    return (PyObject *)out;
}

static PyObject *
py_curve_info(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsToneCurve *c = lcms2_get_curve(arg);
    if (c == NULL)
        return NULL;
    return Py_BuildValue("{s:O,s:O,s:O,s:O,s:i,s:i}",
                         "linear", cmsIsToneCurveLinear(c) ? Py_True : Py_False,
                         "monotonic", cmsIsToneCurveMonotonic(c) ? Py_True : Py_False,
                         "descending", cmsIsToneCurveDescending(c) ? Py_True : Py_False,
                         "multisegment", cmsIsToneCurveMultisegment(c) ? Py_True : Py_False,
                         "parametric_type", (int)cmsGetToneCurveParametricType(c),
                         "entries", (int)cmsGetToneCurveEstimatedTableEntries(c));
}

PyMethodDef lcms2_curves_methods[] = {
    {"curve_gamma", py_curve_gamma, METH_VARARGS,
     "curve_gamma(gamma) -> capsule\n\nBuild a pure power-law tone curve Y = X ** gamma."},
    {"curve_parametric", py_curve_parametric, METH_VARARGS,
     "curve_parametric(type, params) -> capsule\n\nBuild a parametric tone curve (types 1-8, 108, 109; "
     "negative for the inverse). The parameter count is validated per type."},
    {"curve_tabulated", py_curve_tabulated, METH_O,
     "curve_tabulated(values) -> capsule\n\nBuild a tone curve from a 1-D uint16 or float table "
     "(floats are on the 0..1 range)."},
    {"curve_eval", py_curve_eval, METH_VARARGS,
     "curve_eval(curve, x) -> ndarray\n\nEvaluate the curve over a float array (0..1); returns float64."},
    {"curve_eval16", py_curve_eval16, METH_VARARGS,
     "curve_eval16(curve, x) -> ndarray\n\nEvaluate the curve over a uint16 array; returns uint16."},
    {"curve_reverse", py_curve_reverse, METH_VARARGS,
     "curve_reverse(curve, n=4096) -> capsule\n\nReverse the curve, sampling n points when no analytic "
     "inverse exists."},
    {"curve_join", py_curve_join, METH_VARARGS,
     "curve_join(first, second, n=4096) -> capsule\n\nJoin curves: x -> second^-1(first(x))."},
    {"curve_smooth", py_curve_smooth, METH_VARARGS,
     "curve_smooth(curve, lam) -> capsule\n\nReturn a smoothed copy of the curve (lam is the "
     "smoothing parameter)."},
    {"curve_copy", py_curve_copy, METH_O, "curve_copy(curve) -> capsule\n\nDuplicate the curve."},
    {"curve_estimate_gamma", py_curve_estimate_gamma, METH_VARARGS,
     "curve_estimate_gamma(curve, precision=0.01) -> float\n\nEstimate the gamma of the curve."},
    {"curve_table", py_curve_table, METH_O,
     "curve_table(curve) -> ndarray\n\nThe 16-bit table of the curve as a uint16 array."},
    {"curve_info", py_curve_info, METH_O,
     "curve_info(curve) -> dict\n\nFlags and sizes: linear, monotonic, descending, multisegment, "
     "parametric_type, entries."},
    {NULL, NULL, 0, NULL},
};
