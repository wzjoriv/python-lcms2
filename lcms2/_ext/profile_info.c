/* Profiles: signatures, header fields and info text (registered in profile.c). */
#include "profile.h"

#include <string.h>
#include <time.h>

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


/* Signatures ------------------------------------------------------------------------------------ */

int
lcms2_parse_signature(PyObject *obj, cmsUInt32Number *out, const char *what)
{
    if (PyLong_Check(obj)) {
        unsigned long v = PyLong_AsUnsignedLong(obj);
        if (v == (unsigned long)-1 && PyErr_Occurred())
            return -1;
        *out = (cmsUInt32Number)v;
        return 0;
    }
    if (!PyUnicode_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a 4-character signature string, got %.100s", what,
                     Py_TYPE(obj)->tp_name);
        return -1;
    }
    Py_ssize_t n;
    const char *s = PyUnicode_AsUTF8AndSize(obj, &n);
    if (s == NULL)
        return -1;
    if (n < 1 || n > 4) {
        PyErr_Format(PyExc_ValueError, "%s must be 1 to 4 ASCII characters (e.g. 'desc'), got %R", what, obj);
        return -1;
    }
    cmsUInt32Number sig = 0;
    for (Py_ssize_t i = 0; i < 4; i++) {
        unsigned char c = i < n ? (unsigned char)s[i] : ' ';
        if (c > 127) {
            PyErr_Format(PyExc_ValueError, "%s must be ASCII, got %R", what, obj);
            return -1;
        }
        sig = (sig << 8) | c;
    }
    *out = sig;
    return 0;
}

PyObject *
lcms2_signature_str(cmsUInt32Number sig)
{
    char s[4] = {(char)(sig >> 24), (char)(sig >> 16), (char)(sig >> 8), (char)sig};
    return PyUnicode_DecodeLatin1(s, 4, NULL);
}

/* Header ------------------------------------------------------------------------------------------ */

static int
set_item(PyObject *d, const char *key, PyObject *v)
{
    if (v == NULL)
        return -1;
    int rc = PyDict_SetItemString(d, key, v);
    Py_DECREF(v);
    return rc;
}

PyObject *
lcms2_profile_get_header(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHPROFILE h = lcms2_get_profile(arg);
    if (h == NULL)
        return NULL;
    PyObject *d = PyDict_New();
    if (d == NULL)
        return NULL;
    cmsColorSpaceSignature cs = cmsGetColorSpace(h);
    cmsUInt64Number attributes = 0;
    cmsUInt8Number id[16];
    struct tm t;
    memset(&t, 0, sizeof(t));
    cmsGetHeaderAttributes(h, &attributes);
    cmsGetHeaderProfileID(h, id);
    PyObject *date = cmsGetHeaderCreationDateTime(h, &t)
                         ? Py_BuildValue("(iiiiii)", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour,
                                         t.tm_min, t.tm_sec)
                         : Py_NewRef(Py_None);
    if (set_item(d, "device_class", lcms2_signature_str(cmsGetDeviceClass(h))) < 0 ||
        set_item(d, "color_space", lcms2_signature_str(cs)) < 0 ||
        set_item(d, "pcs", lcms2_signature_str(cmsGetPCS(h))) < 0 ||
        set_item(d, "channels", PyLong_FromLong(cmsChannelsOfColorSpace(cs))) < 0 ||
        set_item(d, "pcs_channels", PyLong_FromLong(cmsChannelsOfColorSpace(cmsGetPCS(h)))) < 0 ||
        set_item(d, "version", PyFloat_FromDouble(cmsGetProfileVersion(h))) < 0 ||
        set_item(d, "encoded_version", PyLong_FromUnsignedLong(cmsGetEncodedICCversion(h))) < 0 ||
        set_item(d, "rendering_intent", PyLong_FromUnsignedLong(cmsGetHeaderRenderingIntent(h))) < 0 ||
        set_item(d, "flags", PyLong_FromUnsignedLong(cmsGetHeaderFlags(h))) < 0 ||
        set_item(d, "manufacturer", PyLong_FromUnsignedLong(cmsGetHeaderManufacturer(h))) < 0 ||
        set_item(d, "model", PyLong_FromUnsignedLong(cmsGetHeaderModel(h))) < 0 ||
        set_item(d, "creator", PyLong_FromUnsignedLong(cmsGetHeaderCreator(h))) < 0 ||
        set_item(d, "attributes", PyLong_FromUnsignedLongLong(attributes)) < 0 ||
        set_item(d, "creation_date", date) < 0 ||
        set_item(d, "profile_id", PyBytes_FromStringAndSize((const char *)id, 16)) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

/* Converts optional keyword values; NULL / None means "leave alone". */
static int
opt_u32(PyObject *o, cmsUInt32Number *v, int *set, const char *what)
{
    *set = 0;
    if (o == NULL || o == Py_None)
        return 0;
    unsigned long x = PyLong_AsUnsignedLong(o);
    if (x == (unsigned long)-1 && PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "%s must be an unsigned 32-bit integer", what);
        return -1;
    }
    if (x > 0xFFFFFFFFul) {
        PyErr_Format(PyExc_ValueError, "%s must fit in 32 bits", what);
        return -1;
    }
    *v = (cmsUInt32Number)x;
    *set = 1;
    return 0;
}

static int
opt_sig(PyObject *o, cmsUInt32Number *v, int *set, const char *what)
{
    *set = 0;
    if (o == NULL || o == Py_None)
        return 0;
    if (lcms2_parse_signature(o, v, what) < 0)
        return -1;
    *set = 1;
    return 0;
}

PyObject *
lcms2_profile_set_header(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static char *kwlist[] = {"profile", "device_class", "color_space", "pcs", "version", "rendering_intent",
                             "flags", "manufacturer", "model", "attributes", "profile_id", NULL};
    PyObject *p, *dc = NULL, *cs = NULL, *pcs = NULL, *ver = NULL, *intent = NULL, *flags = NULL, *mfg = NULL,
                 *model = NULL, *attr = NULL, *pid = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$OOOOOOOOOO:profile_set_header", kwlist, &p, &dc, &cs,
                                     &pcs, &ver, &intent, &flags, &mfg, &model, &attr, &pid))
        return NULL;
    cmsHPROFILE h = lcms2_get_profile(p);
    if (h == NULL)
        return NULL;
    cmsUInt32Number vdc = 0, vcs = 0, vpcs = 0, vint = 0, vflags = 0, vmfg = 0, vmodel = 0;
    int sdc, scs, spcs, sint, sflags, smfg, smodel;
    if (opt_sig(dc, &vdc, &sdc, "device_class") < 0 || opt_sig(cs, &vcs, &scs, "color_space") < 0 ||
        opt_sig(pcs, &vpcs, &spcs, "pcs") < 0 || opt_u32(intent, &vint, &sint, "rendering_intent") < 0 ||
        opt_u32(flags, &vflags, &sflags, "flags") < 0 || opt_sig(mfg, &vmfg, &smfg, "manufacturer") < 0 ||
        opt_sig(model, &vmodel, &smodel, "model") < 0)
        return NULL;
    double version = 0.0;
    if (ver != NULL && ver != Py_None) {
        version = PyFloat_AsDouble(ver);
        if (version == -1.0 && PyErr_Occurred())
            return NULL;
        if (version < 1.0 || version >= 6.0) {
            value_errorf("version must be an ICC version such as 2.1, 4.3 or 5.0, got %g",
                         version);
            return NULL;
        }
    }
    unsigned long long vattr = 0;
    if (attr != NULL && attr != Py_None) {
        vattr = PyLong_AsUnsignedLongLong(attr);
        if (vattr == (unsigned long long)-1 && PyErr_Occurred())
            return NULL;
    }
    Py_buffer idbuf = {0};
    if (pid != NULL && pid != Py_None) {
        if (PyObject_GetBuffer(pid, &idbuf, PyBUF_SIMPLE) < 0)
            return NULL;
        if (idbuf.len != 16) {
            PyBuffer_Release(&idbuf);
            PyErr_Format(PyExc_ValueError, "profile_id must be 16 bytes, got %zd", idbuf.len);
            return NULL;
        }
    }
    if (sdc) cmsSetDeviceClass(h, (cmsProfileClassSignature)vdc);
    if (scs) cmsSetColorSpace(h, (cmsColorSpaceSignature)vcs);
    if (spcs) cmsSetPCS(h, (cmsColorSpaceSignature)vpcs);
    if (ver != NULL && ver != Py_None) cmsSetProfileVersion(h, version);
    if (sint) cmsSetHeaderRenderingIntent(h, vint);
    if (sflags) cmsSetHeaderFlags(h, vflags);
    if (smfg) cmsSetHeaderManufacturer(h, vmfg);
    if (smodel) cmsSetHeaderModel(h, vmodel);
    if (attr != NULL && attr != Py_None) cmsSetHeaderAttributes(h, (cmsUInt64Number)vattr);
    if (idbuf.obj != NULL) {
        cmsSetHeaderProfileID(h, (cmsUInt8Number *)idbuf.buf);
        PyBuffer_Release(&idbuf);
    }
    Py_RETURN_NONE;
}

PyObject *
lcms2_profile_compute_id(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHPROFILE h = lcms2_get_profile(arg);
    if (h == NULL)
        return NULL;
    lcms2_clear_error();
    if (!cmsMD5computeID(h))
        return lcms2_raise("cannot compute the profile ID");
    cmsUInt8Number id[16];
    cmsGetHeaderProfileID(h, id);
    return PyBytes_FromStringAndSize((const char *)id, 16);
}

/* Info text -------------------------------------------------------------------------------------- */

static const cmsTagSignature info_tags[] = {cmsSigProfileDescriptionTag, cmsSigDeviceMfgDescTag,
                                            cmsSigDeviceModelDescTag, cmsSigCopyrightTag};

static int
parse_locale(const char *s, char out[3], const char *what)
{
    if (s == NULL) {
        out[0] = out[1] = out[2] = 0;
        return 0;
    }
    if (strlen(s) != 2) {
        PyErr_Format(PyExc_ValueError, "%s must be a 2-letter code such as 'en' / 'US', got '%s'", what, s);
        return -1;
    }
    out[0] = s[0];
    out[1] = s[1];
    out[2] = 0;
    return 0;
}

PyObject *
lcms2_profile_get_info(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static char *kwlist[] = {"profile", "info", "language", "country", NULL};
    PyObject *p;
    int info;
    const char *lang = NULL, *country = NULL;
    char l[3], c[3];
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|zz:profile_get_info", kwlist, &p, &info, &lang, &country))
        return NULL;
    cmsHPROFILE h = lcms2_get_profile(p);
    if (h == NULL)
        return NULL;
    if (info < 0 || info > 3) {
        PyErr_Format(PyExc_ValueError, "info must be 0-3 (description, manufacturer, model, copyright), got %d",
                     info);
        return NULL;
    }
    if (parse_locale(lang, l, "language") < 0 || parse_locale(country, c, "country") < 0)
        return NULL;
    /* Wide-char API: LittleCMS's UTF-8 path sign-extends non-ASCII bytes. */
    cmsUInt32Number n = cmsGetProfileInfo(h, (cmsInfoType)info, l, c, NULL, 0);
    if (n < sizeof(wchar_t) * 2)
        return PyUnicode_FromString("");
    wchar_t *buf = PyMem_Malloc(n);
    if (buf == NULL)
        return PyErr_NoMemory();
    n = cmsGetProfileInfo(h, (cmsInfoType)info, l, c, buf, n);
    Py_ssize_t len = 0, max = (Py_ssize_t)(n / sizeof(wchar_t));
    while (len < max && buf[len] != 0)
        len++;
    PyObject *out = PyUnicode_FromWideChar(buf, len);
    PyMem_Free(buf);
    return out;
}

PyObject *
lcms2_profile_set_info(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static char *kwlist[] = {"profile", "info", "text", "language", "country", NULL};
    PyObject *p;
    int info;
    PyObject *text;
    const char *lang = "en", *country = "US";
    char l[3], c[3];
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiU|ss:profile_set_info", kwlist, &p, &info, &text, &lang,
                                     &country))
        return NULL;
    cmsHPROFILE h = lcms2_get_profile(p);
    if (h == NULL)
        return NULL;
    if (info < 0 || info > 3) {
        PyErr_Format(PyExc_ValueError, "info must be 0-3 (description, manufacturer, model, copyright), got %d",
                     info);
        return NULL;
    }
    if (parse_locale(lang, l, "language") < 0 || parse_locale(country, c, "country") < 0)
        return NULL;
    wchar_t *wtext = PyUnicode_AsWideCharString(text, NULL);
    if (wtext == NULL)
        return NULL;
    lcms2_clear_error();
    cmsMLU *mlu = cmsMLUalloc(NULL, 1);
    if (mlu == NULL) {
        PyMem_Free(wtext);
        return lcms2_raise("cannot allocate text");
    }
    int ok = cmsMLUsetWide(mlu, l, c, wtext) && cmsWriteTag(h, info_tags[info], mlu);
    cmsMLUfree(mlu);
    PyMem_Free(wtext);
    if (!ok)
        return lcms2_raise("cannot write the text tag");
    Py_RETURN_NONE;
}

