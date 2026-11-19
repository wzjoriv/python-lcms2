/* Profiles: open from memory, save to bytes, built-in and synthetic profiles.
 *
 * Header, info text, tags and readers live in profile_info.c; both files share profile.h and
 * this file owns the method table.
 */
#include "profile.h"

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


/* Helpers ------------------------------------------------------------------------------------ */

/* xyY from a sequence of 2 (Y = 1) or 3 numbers. */
static int
parse_xyY(PyObject *obj, cmsCIExyY *out, const char *what)
{
    PyObject *seq = PySequence_Fast(obj, "");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "%s must be a sequence of 2 or 3 numbers (x, y[, Y]), got %.100s", what,
                     Py_TYPE(obj)->tp_name);
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    double v[3] = {0.0, 0.0, 1.0};
    if (n != 2 && n != 3) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s must have 2 or 3 values (x, y[, Y]), got %zd", what, n);
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        v[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
        if (v[i] == -1.0 && PyErr_Occurred()) {
            Py_DECREF(seq);
            return -1;
        }
    }
    Py_DECREF(seq);
    if (!(v[1] > 0.0)) {
        value_errorf("%s: chromaticity y must be > 0, got %g", what, v[1]);
        return -1;
    }
    out->x = v[0];
    out->y = v[1];
    out->Y = v[2];
    return 0;
}

static int
parametric_count(int type)
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

/* A tone curve from a gamma number, a (type, params) tuple or a curve handle. *owned tells whether
 * the caller must free it. */
static cmsToneCurve *
build_curve(PyObject *obj, int *owned)
{
    *owned = 0;
    if (PyFloat_Check(obj) || PyLong_Check(obj)) {
        double g = PyFloat_AsDouble(obj);
        if (g == -1.0 && PyErr_Occurred())
            return NULL;
        if (!(g > 0.0)) {
            value_errorf("gamma must be > 0, got %g", g);
            return NULL;
        }
        lcms2_clear_error();
        cmsToneCurve *c = cmsBuildGamma(NULL, g);
        if (c == NULL)
            return (cmsToneCurve *)lcms2_raise("cannot build gamma curve");
        *owned = 1;
        return c;
    }
    if (PyTuple_Check(obj)) {
        int type;
        PyObject *params;
        if (!PyArg_ParseTuple(obj, "iO;a parametric curve is (type, params)", &type, &params))
            return NULL;
        int need = parametric_count(type);
        if (need < 0) {
            PyErr_Format(PyExc_ValueError, "unknown parametric curve type %d (use 1-8, 108 or 109)", type);
            return NULL;
        }
        PyObject *seq = PySequence_Fast(params, "parametric curve params must be a sequence of numbers");
        if (seq == NULL)
            return NULL;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        if (n != need) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError, "parametric curve type %d takes %d parameters, got %zd", type, need, n);
            return NULL;
        }
        double p[10] = {0};
        for (Py_ssize_t i = 0; i < n; i++) {
            p[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
            if (p[i] == -1.0 && PyErr_Occurred()) {
                Py_DECREF(seq);
                return NULL;
            }
        }
        Py_DECREF(seq);
        lcms2_clear_error();
        cmsToneCurve *c = cmsBuildParametricToneCurve(NULL, type, p);
        if (c == NULL)
            return (cmsToneCurve *)lcms2_raise("cannot build parametric curve");
        *owned = 1;
        return c;
    }
    cmsToneCurve *c = lcms2_get_curve(obj);
    if (c == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "a curve must be a gamma number, a (type, params) tuple or a ToneCurve, got %.100s",
                     Py_TYPE(obj)->tp_name);
    }
    return c;
}

/* Up to cmsMAXCHANNELS curves from a sequence; with `broadcast` a single item fills `n` slots. */
static int
build_curves(PyObject *obj, Py_ssize_t n, cmsToneCurve **curves, int *owned, const char *what)
{
    for (Py_ssize_t i = 0; i < n; i++) {
        curves[i] = NULL;
        owned[i] = 0;
    }
    PyObject *seq = PySequence_Fast(obj, "");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "%s must be a sequence of curves", what);
        return -1;
    }
    Py_ssize_t m = PySequence_Fast_GET_SIZE(seq);
    if (m != 1 && m != n) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s: expected 1 or %zd curves, got %zd", what, n, m);
        return -1;
    }
    for (Py_ssize_t i = 0; i < m; i++) {
        curves[i] = build_curve(PySequence_Fast_GET_ITEM(seq, i), &owned[i]);
        if (curves[i] == NULL) {
            Py_DECREF(seq);
            return -1;
        }
    }
    Py_DECREF(seq);
    for (Py_ssize_t i = m; i < n; i++)
        curves[i] = curves[0]; /* broadcast, not owned */
    return 0;
}

static void
free_curves(cmsToneCurve **curves, int *owned, Py_ssize_t n)
{
    for (Py_ssize_t i = 0; i < n; i++)
        if (owned[i] && curves[i] != NULL)
            cmsFreeToneCurve(curves[i]);
}

/* Open / save ---------------------------------------------------------------------------------- */

static PyObject *
py_open(PyObject *self, PyObject *arg)
{
    (void)self;
    Py_buffer buf;
    if (PyObject_GetBuffer(arg, &buf, PyBUF_SIMPLE) < 0) {
        PyErr_Format(PyExc_TypeError, "expected bytes-like profile data, got %.100s", Py_TYPE(arg)->tp_name);
        return NULL;
    }
    if (buf.len == 0 || (size_t)buf.len > 0xFFFFFFFFu) {
        PyBuffer_Release(&buf);
        PyErr_SetString(lcms2_Error, "profile data is empty or larger than 4 GiB");
        return NULL;
    }
    lcms2_clear_error();
    cmsHPROFILE h = cmsOpenProfileFromMem(buf.buf, (cmsUInt32Number)buf.len);
    PyBuffer_Release(&buf);
    if (h == NULL)
        return lcms2_raise("not a valid ICC profile");
    return lcms2_wrap_profile(h);
}

static PyObject *
py_to_bytes(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHPROFILE h = lcms2_get_profile(arg);
    if (h == NULL)
        return NULL;
    cmsUInt32Number size = 0;
    lcms2_clear_error();
    if (!cmsSaveProfileToMem(h, NULL, &size) || size == 0)
        return lcms2_raise("cannot serialise the profile");
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)size);
    if (out == NULL)
        return NULL;
    if (!cmsSaveProfileToMem(h, PyBytes_AS_STRING(out), &size)) {
        Py_DECREF(out);
        return lcms2_raise("cannot serialise the profile");
    }
    if ((Py_ssize_t)size != PyBytes_GET_SIZE(out) && _PyBytes_Resize(&out, (Py_ssize_t)size) < 0)
        return NULL;
    return out;
}

/* Creation ----------------------------------------------------------------------------------------- */

static PyObject *
py_create_srgb(PyObject *self, PyObject *noargs)
{
    (void)self;
    (void)noargs;
    lcms2_clear_error();
    return lcms2_wrap_profile(cmsCreate_sRGBProfile());
}

static PyObject *
py_create_xyz(PyObject *self, PyObject *noargs)
{
    (void)self;
    (void)noargs;
    lcms2_clear_error();
    return lcms2_wrap_profile(cmsCreateXYZProfile());
}

static PyObject *
py_create_null(PyObject *self, PyObject *noargs)
{
    (void)self;
    (void)noargs;
    lcms2_clear_error();
    return lcms2_wrap_profile(cmsCreateNULLProfile());
}

static PyObject *
py_create_lab(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static char *kwlist[] = {"version", "white_point", NULL};
    int version = 4;
    PyObject *wp_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iO:create_lab", kwlist, &version, &wp_obj))
        return NULL;
    if (version != 2 && version != 4) {
        PyErr_Format(PyExc_ValueError, "Lab profile version must be 2 or 4, got %d", version);
        return NULL;
    }
    cmsCIExyY wp;
    const cmsCIExyY *pwp = NULL;
    if (wp_obj != Py_None) {
        if (parse_xyY(wp_obj, &wp, "white_point") < 0)
            return NULL;
        pwp = &wp;
    }
    lcms2_clear_error();
    return lcms2_wrap_profile(version == 2 ? cmsCreateLab2Profile(pwp) : cmsCreateLab4Profile(pwp));
}

static PyObject *
py_create_rgb(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *wp_obj, *r, *g, *b, *curves_obj;
    if (!PyArg_ParseTuple(args, "O(OOO)O:create_rgb", &wp_obj, &r, &g, &b, &curves_obj))
        return NULL;
    cmsCIExyY wp;
    cmsCIExyYTRIPLE prim;
    if (parse_xyY(wp_obj, &wp, "white_point") < 0 || parse_xyY(r, &prim.Red, "red primary") < 0 ||
        parse_xyY(g, &prim.Green, "green primary") < 0 || parse_xyY(b, &prim.Blue, "blue primary") < 0)
        return NULL;
    cmsToneCurve *curves[3];
    int owned[3];
    if (build_curves(curves_obj, 3, curves, owned, "curves") < 0) {
        free_curves(curves, owned, 3);
        return NULL;
    }
    lcms2_clear_error();
    cmsHPROFILE h = cmsCreateRGBProfile(&wp, &prim, curves);
    free_curves(curves, owned, 3);
    return lcms2_wrap_profile(h);
}

static PyObject *
py_create_gray(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *wp_obj, *curve_obj;
    if (!PyArg_ParseTuple(args, "OO:create_gray", &wp_obj, &curve_obj))
        return NULL;
    cmsCIExyY wp;
    if (parse_xyY(wp_obj, &wp, "white_point") < 0)
        return NULL;
    int owned = 0;
    cmsToneCurve *c = build_curve(curve_obj, &owned);
    if (c == NULL)
        return NULL;
    lcms2_clear_error();
    cmsHPROFILE h = cmsCreateGrayProfile(&wp, c);
    if (owned)
        cmsFreeToneCurve(c);
    return lcms2_wrap_profile(h);
}

static PyObject *
py_create_bchsw(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static char *kwlist[] = {"lut_points", "brightness", "contrast", "hue", "saturation",
                             "temp_src", "temp_dst", NULL};
    int points = 17;
    double bright = 0.0, contrast = 1.0, hue = 0.0, sat = 0.0;
    int tsrc = 0, tdst = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iddddii:create_bchsw", kwlist, &points, &bright, &contrast,
                                     &hue, &sat, &tsrc, &tdst))
        return NULL;
    if (points < 2 || points > 255) {
        PyErr_Format(PyExc_ValueError, "lut_points must be between 2 and 255, got %d", points);
        return NULL;
    }
    if (tsrc < 0 || tdst < 0) {
        PyErr_SetString(PyExc_ValueError, "temperatures must be >= 0 kelvin (0 = no white point change)");
        return NULL;
    }
    lcms2_clear_error();
    return lcms2_wrap_profile(cmsCreateBCHSWabstractProfile((cmsUInt32Number)points, bright, contrast, hue, sat,
                                                            (cmsUInt32Number)tsrc, (cmsUInt32Number)tdst));
}

static PyObject *
py_create_ink_limiting(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *cs_obj;
    double limit;
    cmsUInt32Number cs;
    if (!PyArg_ParseTuple(args, "Od:create_ink_limiting", &cs_obj, &limit))
        return NULL;
    if (lcms2_parse_signature(cs_obj, &cs, "color space") < 0)
        return NULL;
    if (limit < 0.0 || limit > 400.0) {
        value_errorf("ink limit must be between 0 and 400 percent, got %g", limit);
        return NULL;
    }
    lcms2_clear_error();
    return lcms2_wrap_profile(cmsCreateInkLimitingDeviceLink((cmsColorSpaceSignature)cs, limit));
}

static PyObject *
py_create_linearization(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *cs_obj, *curves_obj;
    cmsUInt32Number cs;
    if (!PyArg_ParseTuple(args, "OO:create_linearization", &cs_obj, &curves_obj))
        return NULL;
    if (lcms2_parse_signature(cs_obj, &cs, "color space") < 0)
        return NULL;
    cmsInt32Number n = cmsChannelsOfColorSpace((cmsColorSpaceSignature)cs);
    if (n <= 0 || n > cmsMAXCHANNELS) {
        PyErr_SetString(PyExc_ValueError, "unknown colour space signature for a linearization device link");
        return NULL;
    }
    cmsToneCurve *curves[cmsMAXCHANNELS];
    int owned[cmsMAXCHANNELS];
    if (build_curves(curves_obj, n, curves, owned, "curves") < 0) {
        free_curves(curves, owned, n);
        return NULL;
    }
    lcms2_clear_error();
    cmsHPROFILE h = cmsCreateLinearizationDeviceLink((cmsColorSpaceSignature)cs, curves);
    free_curves(curves, owned, n);
    return lcms2_wrap_profile(h);
}

#define KW (METH_VARARGS | METH_KEYWORDS)
#define CF(f) ((PyCFunction)(void (*)(void))(f))

PyMethodDef lcms2_profile_methods[] = {
    {"profile_open", py_open, METH_O,
     "profile_open(data) -> object\n\nOpen an ICC profile from bytes-like data."},
    {"profile_to_bytes", py_to_bytes, METH_O, "profile_to_bytes(profile) -> bytes\n\nSerialise a profile."},
    {"profile_create_srgb", py_create_srgb, METH_NOARGS, "profile_create_srgb() -> object\n\nBuilt-in sRGB."},
    {"profile_create_xyz", py_create_xyz, METH_NOARGS,
     "profile_create_xyz() -> object\n\nXYZ identity profile (D50 PCS)."},
    {"profile_create_null", py_create_null, METH_NOARGS,
     "profile_create_null() -> object\n\nOutput profile that discards everything (gray, all zeros)."},
    {"profile_create_lab", CF(py_create_lab), KW,
     "profile_create_lab(version=4, white_point=None) -> object\n\n"
     "Lab identity profile; white_point is xyY (D50 when None)."},
    {"profile_create_rgb", py_create_rgb, METH_VARARGS,
     "profile_create_rgb(white_point, primaries, curves) -> object\n\n"
     "Matrix-shaper RGB profile. primaries is (red, green, blue) xyY; curves holds 1 or 3 items,\n"
     "each a gamma number, a (type, params) parametric tuple or a ToneCurve."},
    {"profile_create_gray", py_create_gray, METH_VARARGS,
     "profile_create_gray(white_point, curve) -> object\n\nGray profile from a white point and a curve."},
    {"profile_create_bchsw", CF(py_create_bchsw), KW,
     "profile_create_bchsw(lut_points=17, brightness=0.0, contrast=1.0, hue=0.0, saturation=0.0, "
     "temp_src=0, temp_dst=0) -> object\n\nAbstract Lab profile adjusting brightness, contrast, hue, "
     "saturation and white point."},
    {"profile_create_ink_limiting", py_create_ink_limiting, METH_VARARGS,
     "profile_create_ink_limiting(color_space, limit) -> object\n\nInk-limiting device link (CMYK only)."},
    {"profile_create_linearization", py_create_linearization, METH_VARARGS,
     "profile_create_linearization(color_space, curves) -> object\n\nDevice link applying one curve per "
     "channel."},
    {"profile_get_header", lcms2_profile_get_header, METH_O,
     "profile_get_header(profile) -> dict\n\nHeader fields of a profile."},
    {"profile_set_header", CF(lcms2_profile_set_header), KW,
     "profile_set_header(profile, *, device_class=None, color_space=None, pcs=None, version=None, "
     "rendering_intent=None, flags=None, manufacturer=None, model=None, attributes=None, profile_id=None)"
     " -> None\n\nChange header fields."},
    {"profile_compute_id", lcms2_profile_compute_id, METH_O,
     "profile_compute_id(profile) -> bytes\n\nCompute the MD5 profile ID, store it in the header, return it."},
    {"profile_get_info", CF(lcms2_profile_get_info), KW,
     "profile_get_info(profile, info, language='en', country='US') -> str\n\n"
     "Text of info 0=description, 1=manufacturer, 2=model, 3=copyright."},
    {"profile_set_info", CF(lcms2_profile_set_info), KW,
     "profile_set_info(profile, info, text, language='en', country='US') -> None\n\n"
     "Write an info text tag (0=description, 1=manufacturer, 2=model, 3=copyright)."},
    {"profile_tags", lcms2_profile_tags, METH_O, "profile_tags(profile) -> list[str]\n\nTag signatures."},
    {"profile_is_tag", lcms2_profile_is_tag, METH_VARARGS,
     "profile_is_tag(profile, sig) -> bool\n\nWhether the profile has a tag."},
    {"profile_read_raw_tag", lcms2_profile_read_raw_tag, METH_VARARGS,
     "profile_read_raw_tag(profile, sig) -> bytes\n\nSerialised tag data (with the type header)."},
    {"profile_write_raw_tag", lcms2_profile_write_raw_tag, METH_VARARGS,
     "profile_write_raw_tag(profile, sig, data) -> None\n\nStore raw tag data."},
    {"profile_read_xyz", lcms2_profile_read_xyz, METH_VARARGS,
     "profile_read_xyz(profile, sig) -> tuple[float, float, float] | None\n\n"
     "Read an XYZ tag (wtpt, bkpt, rXYZ, gXYZ, bXYZ, lumi)."},
    {"profile_read_chad", lcms2_profile_read_chad, METH_O,
     "profile_read_chad(profile) -> tuple | None\n\nChromatic adaptation matrix (3x3 nested tuples)."},
    {"profile_read_trc", lcms2_profile_read_trc, METH_VARARGS,
     "profile_read_trc(profile, sig) -> object | None\n\nCopy of a TRC curve (rTRC, gTRC, bTRC, kTRC)."},
    {"profile_is_matrix_shaper", lcms2_profile_is_matrix_shaper, METH_O,
     "profile_is_matrix_shaper(profile) -> bool\n\nWhether the profile is a matrix-shaper."},
    {"profile_is_clut", lcms2_profile_is_clut, METH_VARARGS,
     "profile_is_clut(profile, intent, direction) -> bool\n\nWhether the intent is implemented by a CLUT."},
    {"profile_is_intent_supported", lcms2_profile_is_intent_supported, METH_VARARGS,
     "profile_is_intent_supported(profile, intent, direction) -> bool\n\n"
     "Direction 0 = input, 1 = output, 2 = proof."},
    {"get_supported_intents", lcms2_profile_supported_intents, METH_NOARGS,
     "get_supported_intents() -> list[tuple[int, str]]\n\nIntents the engine implements."},
    {"profile_detect_black_point", CF(lcms2_profile_detect_black_point), KW,
     "profile_detect_black_point(profile, intent, flags=0, destination=False) -> tuple | None\n\n"
     "Black point XYZ, or None when it cannot be detected."},
    {"profile_detect_tac", lcms2_profile_detect_tac, METH_O,
     "profile_detect_tac(profile) -> float\n\nTotal area coverage in percent (0 when unknown)."},
    {"profile_named_colors", lcms2_profile_named_colors, METH_O,
     "profile_named_colors(profile) -> list | None\n\n"
     "Named colours as (name, prefix, suffix, pcs, colorants), or None without a ncl2 tag."},
    {NULL, NULL, 0, NULL},
};
