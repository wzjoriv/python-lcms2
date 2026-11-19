/* Pixel formats: the TYPE_* table, format decoding and per-profile format construction. */
#include "common.h"

#include "formats_table.h"

/* NumPy dtype name of one sample of `fmt`, or NULL when the format has no array layout. */
static const char *
sample_dtype(cmsUInt32Number fmt)
{
    cmsUInt32Number bytes = T_BYTES(fmt);
    if (T_FLOAT(fmt)) {
        switch (bytes) {
        case 0: return "float64"; /* BYTES 0 means double */
        case 2: return "float16";
        case 4: return "float32";
        case 8: return "float64";
        default: return NULL;
        }
    }
    switch (bytes) {
    case 0: return "float64"; /* legacy double encoding */
    case 1: return "uint8";
    case 2: return "uint16";
    default: return NULL;
    }
}

/* (dtype, channels incl. extra, extra, planar) for a format. */
static PyObject *
info_tuple(cmsUInt32Number fmt)
{
    const char *dtype = sample_dtype(fmt);
    int channels = (int)(T_CHANNELS(fmt) + T_EXTRA(fmt));
    if (dtype == NULL) {
        PyErr_Format(PyExc_ValueError, "pixel format 0x%08lx has an unsupported sample size",
                     (unsigned long)fmt);
        return NULL;
    }
    return Py_BuildValue("(siiO)", dtype, channels, (int)T_EXTRA(fmt), T_PLANAR(fmt) ? Py_True : Py_False);
}

static PyObject *
py_format_info(PyObject *self, PyObject *arg)
{
    (void)self;
    unsigned long fmt = PyLong_AsUnsignedLong(arg);
    if (fmt == (unsigned long)-1 && PyErr_Occurred())
        return NULL;
    return info_tuple((cmsUInt32Number)fmt);
}

static PyObject *
py_format_for_profile(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static char *kwlist[] = {"profile", "bytes", "is_float", "extra", "pcs", NULL};
    PyObject *profile;
    int nbytes, is_float, extra = 0, pcs = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oip|ip:format_for_profile", kwlist, &profile, &nbytes,
                                     &is_float, &extra, &pcs))
        return NULL;
    cmsHPROFILE h = lcms2_get_profile(profile);
    if (h == NULL)
        return NULL;
    if (nbytes != 0 && nbytes != 1 && nbytes != 2 && nbytes != 4) {
        PyErr_Format(PyExc_ValueError, "bytes must be 0 (double), 1, 2 or 4, got %d", nbytes);
        return NULL;
    }
    if (extra < 0 || extra > 7) {
        PyErr_Format(PyExc_ValueError, "extra must be between 0 and 7, got %d", extra);
        return NULL;
    }
    lcms2_clear_error();
    cmsUInt32Number fmt = pcs ? cmsFormatterForPCSOfProfile(h, (cmsUInt32Number)nbytes, is_float)
                              : cmsFormatterForColorspaceOfProfile(h, (cmsUInt32Number)nbytes, is_float);
    if (fmt == 0 || T_CHANNELS(fmt) == 0)
        return lcms2_raise(pcs ? "the profile's PCS has no pixel format"
                               : "the profile's colour space has no pixel format");
    fmt |= EXTRA_SH(extra);
    return PyLong_FromUnsignedLong(fmt);
}

static int
add_formats_dict(PyObject *module)
{
    PyObject *d = PyDict_New();
    if (d == NULL)
        return -1;
    for (const lcms2_format_entry *e = lcms2_format_table; e->name != NULL; e++) {
        const char *dtype = sample_dtype(e->value);
        PyObject *v = Py_BuildValue("(ksiiO)", (unsigned long)e->value, dtype ? dtype : "",
                                    (int)(T_CHANNELS(e->value) + T_EXTRA(e->value)), (int)T_EXTRA(e->value),
                                    T_PLANAR(e->value) ? Py_True : Py_False);
        if (v == NULL || PyDict_SetItemString(d, e->name, v) < 0) {
            Py_XDECREF(v);
            Py_DECREF(d);
            return -1;
        }
        Py_DECREF(v);
    }
    int rc = PyModule_AddObjectRef(module, "FORMATS", d);
    Py_DECREF(d);
    return rc;
}

int
lcms2_formats_init(PyObject *module)
{
    return add_formats_dict(module);
}

PyMethodDef lcms2_formats_methods[] = {
    {"format_info", py_format_info, METH_O,
     "format_info(fmt) -> tuple[str, int, int, bool]\n\n"
     "Decode a pixel format: (sample dtype, channels including extra, extra channels, planar)."},
    {"format_for_profile", (PyCFunction)(void (*)(void))py_format_for_profile, METH_VARARGS | METH_KEYWORDS,
     "format_for_profile(profile, bytes, is_float, extra=0, pcs=False) -> int\n\n"
     "Pixel format for the colour space (or PCS) of a profile with `bytes` per sample (0 = double)\n"
     "and `extra` extra (alpha) channels."},
    {NULL, NULL, 0, NULL},
};
