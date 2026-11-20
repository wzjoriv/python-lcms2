/* Colorimetry bindings: delta E, white points, CIECAM02 and the reference XYZ/Lab conversions.
 *
 * All functions take float64 arrays whose last axis has length 3 (already broadcast to a common
 * shape by the Python layer) and run their loops without the GIL.
 */
#include "common.h"

#include <math.h>
#include <string.h>

#define COLOR_FLAGS (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)

/* Convert `obj` to a C-contiguous float64 array whose last axis is 3. New reference or NULL. */
static PyArrayObject *
triple_array(PyObject *obj, const char *what)
{
    PyArrayObject *a = (PyArrayObject *)PyArray_FROMANY(obj, NPY_DOUBLE, 1, 0, COLOR_FLAGS);
    if (a == NULL)
        return NULL;
    if (PyArray_DIM(a, PyArray_NDIM(a) - 1) != 3) {
        PyErr_Format(PyExc_ValueError, "%s must have a last axis of length 3, got shape of last axis %zd",
                     what, (Py_ssize_t)PyArray_DIM(a, PyArray_NDIM(a) - 1));
        Py_DECREF(a);
        return NULL;
    }
    return a;
}

/* Allocate a float64 array: leading dims of `like`, plus a trailing axis of `last` (0 = none). */
static PyArrayObject *
alloc_like(PyArrayObject *like, int last)
{
    int nd = PyArray_NDIM(like) - 1 + (last ? 1 : 0);
    npy_intp dims[NPY_MAXDIMS];
    for (int i = 0; i < PyArray_NDIM(like) - 1; i++)
        dims[i] = PyArray_DIM(like, i);
    if (last)
        dims[nd - 1] = last;
    return (PyArrayObject *)PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
}

/* delta_e ------------------------------------------------------------------------------------ */

enum { DE_76, DE_94, DE_2000, DE_CMC, DE_BFD };

static PyObject *
py_delta_e(PyObject *self, PyObject *args)
{
    PyObject *aobj, *bobj;
    int method;
    double p0 = 1, p1 = 1, p2 = 1;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOi|ddd", &aobj, &bobj, &method, &p0, &p1, &p2))
        return NULL;
    if (method < DE_76 || method > DE_BFD) {
        PyErr_SetString(PyExc_ValueError, "unknown delta E method code");
        return NULL;
    }
    PyArrayObject *a = triple_array(aobj, "lab1");
    if (a == NULL)
        return NULL;
    PyArrayObject *b = triple_array(bobj, "lab2");
    if (b == NULL) {
        Py_DECREF(a);
        return NULL;
    }
    if (!PyArray_SAMESHAPE(a, b)) {
        PyErr_SetString(PyExc_ValueError, "lab1 and lab2 must have the same shape (broadcast them first)");
        Py_DECREF(a);
        Py_DECREF(b);
        return NULL;
    }
    PyArrayObject *out = alloc_like(a, 0);
    if (out == NULL) {
        Py_DECREF(a);
        Py_DECREF(b);
        return NULL;
    }
    npy_intp n = PyArray_SIZE(a) / 3, i;
    const double *pa = PyArray_DATA(a), *pb = PyArray_DATA(b);
    double *po = PyArray_DATA(out);
    Py_BEGIN_ALLOW_THREADS
    for (i = 0; i < n; i++) {
        cmsCIELab l1 = {pa[3 * i], pa[3 * i + 1], pa[3 * i + 2]};
        cmsCIELab l2 = {pb[3 * i], pb[3 * i + 1], pb[3 * i + 2]};
        switch (method) {
        case DE_76:   po[i] = cmsDeltaE(&l1, &l2); break;
        case DE_94:   po[i] = cmsCIE94DeltaE(&l1, &l2); break;
        case DE_2000: po[i] = cmsCIE2000DeltaE(&l1, &l2, p0, p1, p2); break;
        case DE_CMC:  po[i] = cmsCMCdeltaE(&l1, &l2, p0, p1); break;
        default:      po[i] = cmsBFDdeltaE(&l1, &l2); break;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(a);
    Py_DECREF(b);
    return (PyObject *)out;
}

/* Reference XYZ <-> Lab (for validating the NumPy versions) ---------------------------------- */

static PyObject *
convert_triples(PyObject *args, int to_lab)
{
    PyObject *wobj, *xobj;
    if (!PyArg_ParseTuple(args, "OO", &wobj, &xobj))
        return NULL;
    PyArrayObject *w = triple_array(wobj, "white");
    if (w == NULL)
        return NULL;
    PyArrayObject *x = triple_array(xobj, to_lab ? "xyz" : "lab");
    if (x == NULL) {
        Py_DECREF(w);
        return NULL;
    }
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_DOUBLE);
    if (out == NULL) {
        Py_DECREF(w);
        Py_DECREF(x);
        return NULL;
    }
    const double *pw = PyArray_DATA(w);
    cmsCIEXYZ white = {pw[0], pw[1], pw[2]};
    npy_intp n = PyArray_SIZE(x) / 3, i;
    const double *px = PyArray_DATA(x);
    double *po = PyArray_DATA(out);
    Py_BEGIN_ALLOW_THREADS
    for (i = 0; i < n; i++) {
        if (to_lab) {
            cmsCIEXYZ v = {px[3 * i], px[3 * i + 1], px[3 * i + 2]};
            cmsCIELab l;
            cmsXYZ2Lab(&white, &l, &v);
            po[3 * i] = l.L; po[3 * i + 1] = l.a; po[3 * i + 2] = l.b;
        }
        else {
            cmsCIELab l = {px[3 * i], px[3 * i + 1], px[3 * i + 2]};
            cmsCIEXYZ v;
            cmsLab2XYZ(&white, &v, &l);
            po[3 * i] = v.X; po[3 * i + 1] = v.Y; po[3 * i + 2] = v.Z;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(w);
    Py_DECREF(x);
    return (PyObject *)out;
}

static PyObject *
py_xyz_to_lab(PyObject *self, PyObject *args)
{
    (void)self;
    return convert_triples(args, 1);
}

static PyObject *
py_lab_to_xyz(PyObject *self, PyObject *args)
{
    (void)self;
    return convert_triples(args, 0);
}

/* White point / temperature ------------------------------------------------------------------ */

static PyObject *
py_white_from_temp(PyObject *self, PyObject *arg)
{
    (void)self;
    PyArrayObject *t = (PyArrayObject *)PyArray_FROMANY(arg, NPY_DOUBLE, 0, 0, COLOR_FLAGS);
    if (t == NULL)
        return NULL;
    int nd = PyArray_NDIM(t);
    npy_intp dims[NPY_MAXDIMS];
    for (int i = 0; i < nd; i++)
        dims[i] = PyArray_DIM(t, i);
    dims[nd] = 3;
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(nd + 1, dims, NPY_DOUBLE);
    if (out == NULL) {
        Py_DECREF(t);
        return NULL;
    }
    npy_intp n = PyArray_SIZE(t), i;
    const double *pt = PyArray_DATA(t);
    double *po = PyArray_DATA(out);
    int failed = 0;
    Py_BEGIN_ALLOW_THREADS
    for (i = 0; i < n; i++) {
        cmsCIExyY w;
        if (cmsWhitePointFromTemp(&w, pt[i])) {
            po[3 * i] = w.x; po[3 * i + 1] = w.y; po[3 * i + 2] = w.Y;
        }
        else {
            po[3 * i] = po[3 * i + 1] = po[3 * i + 2] = NAN;
            failed = 1;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(t);
    if (failed) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError,
                        "white point temperature must be between 4000 K and 25000 K (CIE daylight locus)");
        return NULL;
    }
    return (PyObject *)out;
}

static PyObject *
py_temp_from_white(PyObject *self, PyObject *arg)
{
    (void)self;
    PyArrayObject *w = triple_array(arg, "xyY");
    if (w == NULL)
        return NULL;
    PyArrayObject *out = alloc_like(w, 0);
    if (out == NULL) {
        Py_DECREF(w);
        return NULL;
    }
    npy_intp n = PyArray_SIZE(w) / 3, i;
    const double *pw = PyArray_DATA(w);
    double *po = PyArray_DATA(out);
    int failed = 0;
    Py_BEGIN_ALLOW_THREADS
    for (i = 0; i < n; i++) {
        cmsCIExyY c = {pw[3 * i], pw[3 * i + 1], pw[3 * i + 2]};
        double k;
        if (cmsTempFromWhitePoint(&k, &c)) {
            po[i] = k;
        }
        else {
            po[i] = NAN;
            failed = 1;
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(w);
    if (failed) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError,
                        "could not compute a correlated color temperature for this chromaticity "
                        "(it is too far from the Planckian locus)");
        return NULL;
    }
    return (PyObject *)out;
}

/* CIECAM02 ----------------------------------------------------------------------------------- */

/* cam(xyz, white, Yb, La, surround, D, reverse) -> array. XYZ values are on the 0..100 scale. */
static PyObject *
py_cam02(PyObject *self, PyObject *args)
{
    PyObject *xobj, *wobj;
    double yb, la, d;
    int surround, reverse;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOddidp", &xobj, &wobj, &yb, &la, &surround, &d, &reverse))
        return NULL;
    if (surround < AVG_SURROUND || surround > CUTSHEET_SURROUND) {
        PyErr_SetString(PyExc_ValueError, "surround must be 1 (average), 2 (dim), 3 (dark) or 4 (cutsheet)");
        return NULL;
    }
    PyArrayObject *w = triple_array(wobj, "white_point");
    if (w == NULL)
        return NULL;
    PyArrayObject *x = triple_array(xobj, reverse ? "JCh" : "XYZ");
    if (x == NULL) {
        Py_DECREF(w);
        return NULL;
    }
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_DOUBLE);
    if (out == NULL) {
        Py_DECREF(w);
        Py_DECREF(x);
        return NULL;
    }
    const double *pw = PyArray_DATA(w);
    cmsViewingConditions vc;
    vc.whitePoint.X = pw[0];
    vc.whitePoint.Y = pw[1];
    vc.whitePoint.Z = pw[2];
    vc.Yb = yb;
    vc.La = la;
    vc.surround = (cmsUInt32Number)surround;
    vc.D_value = d;
    Py_DECREF(w);

    lcms2_clear_error();
    cmsHANDLE model = cmsCIECAM02Init(NULL, &vc);
    if (model == NULL) {
        Py_DECREF(x);
        Py_DECREF(out);
        return lcms2_raise("could not create the CIECAM02 model");
    }
    npy_intp n = PyArray_SIZE(x) / 3, i;
    const double *px = PyArray_DATA(x);
    double *po = PyArray_DATA(out);
    Py_BEGIN_ALLOW_THREADS
    for (i = 0; i < n; i++) {
        if (reverse) {
            cmsJCh in = {px[3 * i], px[3 * i + 1], px[3 * i + 2]};
            cmsCIEXYZ o;
            cmsCIECAM02Reverse(model, &in, &o);
            po[3 * i] = o.X; po[3 * i + 1] = o.Y; po[3 * i + 2] = o.Z;
        }
        else {
            cmsCIEXYZ in = {px[3 * i], px[3 * i + 1], px[3 * i + 2]};
            cmsJCh o;
            cmsCIECAM02Forward(model, &in, &o);
            po[3 * i] = o.J; po[3 * i + 1] = o.C; po[3 * i + 2] = o.h;
        }
    }
    Py_END_ALLOW_THREADS
    cmsCIECAM02Done(model);
    Py_DECREF(x);
    return (PyObject *)out;
}

PyMethodDef lcms2_colorimetry_methods[] = {
    {"delta_e", py_delta_e, METH_VARARGS,
     "delta_e(lab1, lab2, method, p0=1.0, p1=1.0, p2=1.0) -> ndarray\n\nColor difference over two "
     "equally-shaped (..., 3) Lab arrays. method: 0=76, 1=94, 2=2000 (p0..p2 = Kl, Kc, Kh), "
     "3=CMC (p0, p1 = l, c), 4=BFD."},
    {"_xyz_to_lab", py_xyz_to_lab, METH_VARARGS,
     "_xyz_to_lab(white, xyz) -> ndarray\n\nReference cmsXYZ2Lab over a (..., 3) array (used by the tests)."},
    {"_lab_to_xyz", py_lab_to_xyz, METH_VARARGS,
     "_lab_to_xyz(white, lab) -> ndarray\n\nReference cmsLab2XYZ over a (..., 3) array (used by the tests)."},
    {"white_from_temp", py_white_from_temp, METH_O,
     "white_from_temp(kelvin) -> ndarray\n\nxyY of the daylight white point (4000-25000 K); output has a "
     "trailing axis of 3."},
    {"temp_from_white", py_temp_from_white, METH_O,
     "temp_from_white(xyy) -> ndarray\n\nCorrelated color temperature in kelvin of (..., 3) xyY values."},
    {"cam02", py_cam02, METH_VARARGS,
     "cam02(values, white, yb, la, surround, d, reverse) -> ndarray\n\nCIECAM02 forward (XYZ to JCh) or "
     "reverse (JCh to XYZ) over (..., 3) arrays; XYZ on the 0..100 scale, d=-1 computes the degree "
     "of adaptation."},
    {NULL, NULL, 0, NULL},
};
