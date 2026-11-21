/* Transforms: creation (plain, proofing, multi-profile, extended), apply, device links. */
#include "common.h"

#include <stdarg.h>

/* ValueError with a printf-style message (PyErr_Format has no %g). */
static void
value_errorf(const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    PyOS_vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    PyErr_SetString(PyExc_ValueError, buf);
}


#define MAX_PROFILES 255

/* Profiles from a sequence; fills hs[] (borrowed). Returns the count or -1. */
static Py_ssize_t
get_profiles(PyObject *obj, cmsHPROFILE *hs)
{
    PyObject *seq = PySequence_Fast(obj, "profiles must be a sequence of Profile objects");
    if (seq == NULL)
        return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n < 1 || n > MAX_PROFILES) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "expected 1 to %d profiles, got %zd", MAX_PROFILES, n);
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        /* items stay alive through the caller's argument */
        if ((hs[i] = lcms2_get_profile(PySequence_Fast_GET_ITEM(seq, i))) == NULL) {
            Py_DECREF(seq);
            return -1;
        }
    }
    Py_DECREF(seq);
    return n;
}

static PyObject *
py_create(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *src, *dst;
    unsigned long in_fmt, out_fmt, flags;
    unsigned int intent;
    if (!PyArg_ParseTuple(args, "OOkkIk:transform_create", &src, &dst, &in_fmt, &out_fmt, &intent, &flags))
        return NULL;
    cmsHPROFILE hs = lcms2_get_profile(src), hd;
    if (hs == NULL || (hd = lcms2_get_profile(dst)) == NULL)
        return NULL;
    lcms2_clear_error();
    cmsHTRANSFORM t = cmsCreateTransform(hs, (cmsUInt32Number)in_fmt, hd, (cmsUInt32Number)out_fmt, intent,
                                         (cmsUInt32Number)flags | cmsFLAGS_NOCACHE);
    if (t == NULL)
        return lcms2_raise("cannot create the transform");
    return lcms2_wrap_transform(t);
}

static PyObject *
py_create_proofing(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *src, *dst, *proof;
    unsigned long in_fmt, out_fmt, flags;
    unsigned int intent, pintent;
    if (!PyArg_ParseTuple(args, "OOOkkIIk:transform_create_proofing", &src, &dst, &proof, &in_fmt, &out_fmt,
                          &intent, &pintent, &flags))
        return NULL;
    cmsHPROFILE hs = lcms2_get_profile(src), hd = NULL, hp = NULL;
    if (hs == NULL || (hd = lcms2_get_profile(dst)) == NULL || (hp = lcms2_get_profile(proof)) == NULL)
        return NULL;
    if (pintent > 3) {
        PyErr_Format(PyExc_ValueError, "proofing intent must be 0-3, got %u", pintent);
        return NULL;
    }
    lcms2_clear_error();
    cmsHTRANSFORM t = cmsCreateProofingTransform(hs, (cmsUInt32Number)in_fmt, hd, (cmsUInt32Number)out_fmt, hp,
                                                 intent, pintent, (cmsUInt32Number)flags | cmsFLAGS_NOCACHE);
    if (t == NULL)
        return lcms2_raise("cannot create the proofing transform");
    return lcms2_wrap_transform(t);
}

static PyObject *
py_create_multi(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *profiles;
    unsigned long in_fmt, out_fmt, flags;
    unsigned int intent;
    cmsHPROFILE hs[MAX_PROFILES];
    if (!PyArg_ParseTuple(args, "OkkIk:transform_create_multi", &profiles, &in_fmt, &out_fmt, &intent, &flags))
        return NULL;
    Py_ssize_t n = get_profiles(profiles, hs);
    if (n < 0)
        return NULL;
    lcms2_clear_error();
    cmsHTRANSFORM t = cmsCreateMultiprofileTransform(hs, (cmsUInt32Number)n, (cmsUInt32Number)in_fmt,
                                                     (cmsUInt32Number)out_fmt, intent,
                                                     (cmsUInt32Number)flags | cmsFLAGS_NOCACHE);
    if (t == NULL)
        return lcms2_raise("cannot create the multi-profile transform");
    return lcms2_wrap_transform(t);
}

/* A per-profile sequence of length n into `out` via `conv`. */
static int
per_profile(PyObject *obj, Py_ssize_t n, const char *what, int (*conv)(PyObject *, void *, Py_ssize_t),
            void *out)
{
    PyObject *seq = PySequence_Fast(obj, "");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "%s must be a sequence with one value per profile", what);
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(seq) != n) {
        PyErr_Format(PyExc_ValueError, "%s needs one value per profile (%zd), got %zd", what, n,
                     PySequence_Fast_GET_SIZE(seq));
        Py_DECREF(seq);
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++)
        if (conv(PySequence_Fast_GET_ITEM(seq, i), out, i) < 0) {
            Py_DECREF(seq);
            return -1;
        }
    Py_DECREF(seq);
    return 0;
}

static int
conv_bool(PyObject *o, void *out, Py_ssize_t i)
{
    int v = PyObject_IsTrue(o);
    if (v < 0)
        return -1;
    ((cmsBool *)out)[i] = v;
    return 0;
}

static int
conv_intent(PyObject *o, void *out, Py_ssize_t i)
{
    unsigned long v = PyLong_AsUnsignedLong(o);
    if (v == (unsigned long)-1 && PyErr_Occurred())
        return -1;
    ((cmsUInt32Number *)out)[i] = (cmsUInt32Number)v;
    return 0;
}

static int
conv_double(PyObject *o, void *out, Py_ssize_t i)
{
    double v = PyFloat_AsDouble(o);
    if (v == -1.0 && PyErr_Occurred())
        return -1;
    ((cmsFloat64Number *)out)[i] = v;
    return 0;
}

static PyObject *
py_create_extended(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *profiles, *bpc_obj, *intents_obj, *adapt_obj, *gamut;
    unsigned int gamut_pos;
    unsigned long in_fmt, out_fmt, flags;
    cmsHPROFILE hs[MAX_PROFILES];
    cmsBool bpc[MAX_PROFILES];
    cmsUInt32Number intents[MAX_PROFILES];
    cmsFloat64Number adapt[MAX_PROFILES];
    if (!PyArg_ParseTuple(args, "OOOOOIkkk:transform_create_extended", &profiles, &bpc_obj, &intents_obj,
                          &adapt_obj, &gamut, &gamut_pos, &in_fmt, &out_fmt, &flags))
        return NULL;
    Py_ssize_t n = get_profiles(profiles, hs);
    if (n < 0 || per_profile(bpc_obj, n, "bpc", conv_bool, bpc) < 0 ||
        per_profile(intents_obj, n, "intents", conv_intent, intents) < 0 ||
        per_profile(adapt_obj, n, "adaptation", conv_double, adapt) < 0)
        return NULL;
    cmsHPROFILE hg = NULL;
    if (gamut != Py_None) {
        if ((hg = lcms2_get_profile(gamut)) == NULL)
            return NULL;
        if (gamut_pos == 0 || gamut_pos > (unsigned int)n) {
            PyErr_Format(PyExc_ValueError, "gamut position must be between 1 and %zd", n);
            return NULL;
        }
    }
    lcms2_clear_error();
    cmsHTRANSFORM t = cmsCreateExtendedTransform(NULL, (cmsUInt32Number)n, hs, bpc, intents, adapt, hg, gamut_pos,
                                                 (cmsUInt32Number)in_fmt, (cmsUInt32Number)out_fmt,
                                                 (cmsUInt32Number)flags | cmsFLAGS_NOCACHE);
    if (t == NULL)
        return lcms2_raise("cannot create the extended transform");
    return lcms2_wrap_transform(t);
}

/* Apply ----------------------------------------------------------------------------------------- */

static int
expected_type(cmsUInt32Number fmt)
{
    cmsUInt32Number b = T_BYTES(fmt);
    if (T_FLOAT(fmt))
        return b == 2 ? NPY_HALF : b == 4 ? NPY_FLOAT32 : NPY_FLOAT64;
    return b == 1 ? NPY_UINT8 : b == 2 ? NPY_UINT16 : NPY_FLOAT64;
}

static Py_ssize_t
sample_size(cmsUInt32Number fmt)
{
    cmsUInt32Number b = T_BYTES(fmt);
    return b == 0 ? 8 : (Py_ssize_t)b;
}

static int
check_array(PyObject *obj, cmsUInt32Number fmt, int writable, const char *what)
{
    if (!PyArray_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a numpy array", what);
        return -1;
    }
    PyArrayObject *a = (PyArrayObject *)obj;
    if (!PyArray_IS_C_CONTIGUOUS(a) || !PyArray_ISALIGNED(a)) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous and aligned (use np.ascontiguousarray)", what);
        return -1;
    }
    if (writable && !PyArray_ISWRITEABLE(a)) {
        PyErr_Format(PyExc_ValueError, "%s must be writable", what);
        return -1;
    }
    if (PyArray_TYPE(a) != expected_type(fmt) || PyArray_ITEMSIZE(a) != sample_size(fmt)) {
        PyArray_Descr *d = PyArray_DescrFromType(expected_type(fmt));
        PyErr_Format(PyExc_TypeError, "%s has dtype %S but the pixel format needs %S", what,
                     (PyObject *)PyArray_DESCR(a), (PyObject *)d);
        Py_XDECREF(d);
        return -1;
    }
    return 0;
}

#define CHUNK_PIXELS ((Py_ssize_t)1 << 22)

static PyObject *
py_apply(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *th, *src, *dst;
    if (!PyArg_ParseTuple(args, "OOO:transform_apply", &th, &src, &dst))
        return NULL;
    cmsHTRANSFORM t = lcms2_get_transform(th);
    if (t == NULL)
        return NULL;
    cmsUInt32Number fin = cmsGetTransformInputFormat(t), fout = cmsGetTransformOutputFormat(t);
    if (check_array(src, fin, 0, "input") < 0 || check_array(dst, fout, 1, "output") < 0)
        return NULL;
    Py_ssize_t sin = sample_size(fin), sout = sample_size(fout);
    Py_ssize_t cin = T_CHANNELS(fin) + T_EXTRA(fin), cout = T_CHANNELS(fout) + T_EXTRA(fout);
    if (cin == 0 || cout == 0) {
        PyErr_SetString(PyExc_ValueError, "the transform has a format without channels");
        return NULL;
    }
    Py_ssize_t nin = PyArray_NBYTES((PyArrayObject *)src), nout = PyArray_NBYTES((PyArrayObject *)dst);
    if (nin % (sin * cin) != 0) {
        PyErr_Format(PyExc_ValueError, "input holds %zd samples, not a multiple of %zd channels", nin / sin, cin);
        return NULL;
    }
    Py_ssize_t npix = nin / (sin * cin);
    if (nout != npix * sout * cout) {
        PyErr_Format(PyExc_ValueError, "output must hold %zd pixels of %zd channels (%zd samples), has %zd",
                     npix, cout, npix * cout, nout / sout);
        return NULL;
    }
    if (npix == 0)
        Py_RETURN_NONE;
    int pin = T_PLANAR(fin), pout = T_PLANAR(fout);
    if ((pin && npix * sin > 0xFFFFFFFFll) || (pout && npix * sout > 0xFFFFFFFFll)) {
        PyErr_SetString(PyExc_ValueError, "planar planes larger than 4 GiB are not supported");
        return NULL;
    }
    const char *in = PyArray_BYTES((PyArrayObject *)src);
    char *out = PyArray_BYTES((PyArrayObject *)dst);
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t start = 0; start < npix; start += CHUNK_PIXELS) {
        Py_ssize_t n = npix - start < CHUNK_PIXELS ? npix - start : CHUNK_PIXELS;
        const char *ip = in + start * (pin ? sin : sin * cin);
        char *op = out + start * (pout ? sout : sout * cout);
        cmsUInt32Number plane_in = (cmsUInt32Number)(pin ? npix * sin : n * sin * cin);
        cmsUInt32Number plane_out = (cmsUInt32Number)(pout ? npix * sout : n * sout * cout);
        cmsDoTransformLineStride(t, ip, op, (cmsUInt32Number)n, 1, (cmsUInt32Number)(n * sin * cin),
                                 (cmsUInt32Number)(n * sout * cout), plane_in, plane_out);
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

/* Misc ------------------------------------------------------------------------------------------- */

static PyObject *
py_to_device_link(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *th;
    double version;
    unsigned long flags;
    if (!PyArg_ParseTuple(args, "Odk:transform_to_device_link", &th, &version, &flags))
        return NULL;
    cmsHTRANSFORM t = lcms2_get_transform(th);
    if (t == NULL)
        return NULL;
    if (version < 2.0 || version >= 5.0) {
        value_errorf("device link version must be between 2.0 and 4.4, got %g", version);
        return NULL;
    }
    lcms2_clear_error();
    cmsHPROFILE h = cmsTransform2DeviceLink(t, version, (cmsUInt32Number)flags);
    if (h == NULL)
        return lcms2_raise("cannot convert the transform to a device link");
    return lcms2_wrap_profile(h);
}

static PyObject *
py_get_alarm_codes(PyObject *self, PyObject *noargs)
{
    (void)self;
    (void)noargs;
    cmsUInt16Number c[cmsMAXCHANNELS];
    cmsGetAlarmCodes(c);
    PyObject *out = PyTuple_New(cmsMAXCHANNELS);
    if (out == NULL)
        return NULL;
    for (int i = 0; i < cmsMAXCHANNELS; i++) {
        PyObject *v = PyLong_FromLong(c[i]);
        if (v == NULL) {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, v);
    }
    return out;
}

static PyObject *
py_set_alarm_codes(PyObject *self, PyObject *arg)
{
    (void)self;
    PyObject *seq = PySequence_Fast(arg, "alarm codes must be a sequence of up to 16 integers 0-65535");
    if (seq == NULL)
        return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n < 1 || n > cmsMAXCHANNELS) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "expected 1 to 16 alarm codes, got %zd", n);
        return NULL;
    }
    cmsUInt16Number c[cmsMAXCHANNELS] = {0};
    for (Py_ssize_t i = 0; i < n; i++) {
        long v = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if ((v == -1 && PyErr_Occurred()) || v < 0 || v > 65535) {
            Py_DECREF(seq);
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_ValueError, "alarm codes must be integers 0-65535");
            return NULL;
        }
        c[i] = (cmsUInt16Number)v;
    }
    Py_DECREF(seq);
    cmsSetAlarmCodes(c);
    Py_RETURN_NONE;
}

static PyObject *
py_set_adaptation_state(PyObject *self, PyObject *arg)
{
    (void)self;
    double d = PyFloat_AsDouble(arg);
    if (d == -1.0 && PyErr_Occurred())
        return NULL;
    if (d > 1.0) {
        value_errorf("adaptation state must be 0..1 (negative queries), got %g", d);
        return NULL;
    }
    return PyFloat_FromDouble(cmsSetAdaptationState(d));
}

PyMethodDef lcms2_transform_methods[] = {
    {"transform_create", py_create, METH_VARARGS,
     "transform_create(src, dst, in_fmt, out_fmt, intent, flags) -> object\n\nCreate a transform."},
    {"transform_create_proofing", py_create_proofing, METH_VARARGS,
     "transform_create_proofing(src, dst, proof, in_fmt, out_fmt, intent, proof_intent, flags) -> object\n\n"
     "Create a soft-proofing transform."},
    {"transform_create_multi", py_create_multi, METH_VARARGS,
     "transform_create_multi(profiles, in_fmt, out_fmt, intent, flags) -> object\n\n"
     "Create a transform through a chain of profiles."},
    {"transform_create_extended", py_create_extended, METH_VARARGS,
     "transform_create_extended(profiles, bpc, intents, adaptation, gamut, gamut_position, in_fmt, out_fmt, "
     "flags) -> object\n\nCreate a transform with per-profile intents, BPC and adaptation states."},
    {"transform_apply", py_apply, METH_VARARGS,
     "transform_apply(transform, src, dst) -> None\n\n"
     "Transform contiguous `src` into `dst` (dtypes and sizes checked against the formats)."},
    {"transform_to_device_link", py_to_device_link, METH_VARARGS,
     "transform_to_device_link(transform, version, flags) -> object\n\nDevice link profile of a transform."},
    {"get_alarm_codes", py_get_alarm_codes, METH_NOARGS,
     "get_alarm_codes() -> tuple[int, ...]\n\nGamut-check alarm codes (16 values, global)."},
    {"set_alarm_codes", py_set_alarm_codes, METH_O,
     "set_alarm_codes(codes) -> None\n\nSet the gamut-check alarm codes (global)."},
    {"set_adaptation_state", py_set_adaptation_state, METH_O,
     "set_adaptation_state(state) -> float\n\nSet the global adaptation state; returns the previous one "
     "(negative = query only)."},
    {NULL, NULL, 0, NULL},
};
