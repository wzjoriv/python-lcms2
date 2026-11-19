/* Profiles: header fields, info text, tags and tag readers (registered in profile.c). */
#include "profile.h"

#include <string.h>
#include <time.h>

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

static int
parse_profile_sig(PyObject *args, const char *fmt, cmsHPROFILE *h, cmsUInt32Number *sig)
{
    PyObject *p, *s;
    if (!PyArg_ParseTuple(args, fmt, &p, &s))
        return -1;
    if ((*h = lcms2_get_profile(p)) == NULL)
        return -1;
    return lcms2_parse_signature(s, sig, "tag signature");
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
            PyErr_Format(PyExc_ValueError, "version must be an ICC version such as 2.1, 4.3 or 5.0, got %g",
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
    cmsUInt32Number n = cmsGetProfileInfoUTF8(h, (cmsInfoType)info, l, c, NULL, 0);
    if (n <= 1)
        return PyUnicode_FromString("");
    char *buf = PyMem_Malloc(n);
    if (buf == NULL)
        return PyErr_NoMemory();
    n = cmsGetProfileInfoUTF8(h, (cmsInfoType)info, l, c, buf, n);
    PyObject *out = PyUnicode_DecodeUTF8(buf, n > 0 ? (Py_ssize_t)strnlen(buf, n) : 0, "replace");
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
    const char *text, *lang = "en", *country = "US";
    char l[3], c[3];
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Ois|ss:profile_set_info", kwlist, &p, &info, &text, &lang,
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
    lcms2_clear_error();
    cmsMLU *mlu = cmsMLUalloc(NULL, 1);
    if (mlu == NULL)
        return lcms2_raise("cannot allocate text");
    int ok = cmsMLUsetUTF8(mlu, l, c, text) && cmsWriteTag(h, info_tags[info], mlu);
    cmsMLUfree(mlu);
    if (!ok)
        return lcms2_raise("cannot write the text tag");
    Py_RETURN_NONE;
}

/* Tags ----------------------------------------------------------------------------------------------- */

PyObject *
lcms2_profile_tags(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHPROFILE h = lcms2_get_profile(arg);
    if (h == NULL)
        return NULL;
    cmsInt32Number n = cmsGetTagCount(h);
    PyObject *out = PyList_New(0);
    if (out == NULL)
        return NULL;
    for (cmsInt32Number i = 0; i < n; i++) {
        PyObject *s = lcms2_signature_str(cmsGetTagSignature(h, (cmsUInt32Number)i));
        if (s == NULL || PyList_Append(out, s) < 0) {
            Py_XDECREF(s);
            Py_DECREF(out);
            return NULL;
        }
        Py_DECREF(s);
    }
    return out;
}

PyObject *
lcms2_profile_is_tag(PyObject *self, PyObject *args)
{
    (void)self;
    cmsHPROFILE h;
    cmsUInt32Number sig;
    if (parse_profile_sig(args, "OO:profile_is_tag", &h, &sig) < 0)
        return NULL;
    return PyBool_FromLong(cmsIsTag(h, (cmsTagSignature)sig));
}

PyObject *
lcms2_profile_read_raw_tag(PyObject *self, PyObject *args)
{
    (void)self;
    cmsHPROFILE h;
    cmsUInt32Number sig;
    if (parse_profile_sig(args, "OO:profile_read_raw_tag", &h, &sig) < 0)
        return NULL;
    if (!cmsIsTag(h, (cmsTagSignature)sig)) {
        PyObject *s = lcms2_signature_str(sig);
        if (s != NULL)
            PyErr_Format(PyExc_KeyError, "the profile has no %R tag", s);
        Py_XDECREF(s);
        return NULL;
    }
    lcms2_clear_error();
    cmsUInt32Number n = cmsReadRawTag(h, (cmsTagSignature)sig, NULL, 0);
    if (n == 0)
        return lcms2_raise("cannot read the tag");
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)n);
    if (out == NULL)
        return NULL;
    cmsUInt32Number got = cmsReadRawTag(h, (cmsTagSignature)sig, PyBytes_AS_STRING(out), n);
    if (got == 0) {
        Py_DECREF(out);
        return lcms2_raise("cannot read the tag");
    }
    if (got != n && _PyBytes_Resize(&out, (Py_ssize_t)got) < 0)
        return NULL;
    return out;
}

PyObject *
lcms2_profile_write_raw_tag(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *p, *s;
    Py_buffer buf;
    cmsUInt32Number sig;
    if (!PyArg_ParseTuple(args, "OOy*:profile_write_raw_tag", &p, &s, &buf))
        return NULL;
    cmsHPROFILE h = lcms2_get_profile(p);
    if (h == NULL || lcms2_parse_signature(s, &sig, "tag signature") < 0) {
        PyBuffer_Release(&buf);
        return NULL;
    }
    if (buf.len < 8 || (size_t)buf.len > 0xFFFFFFFFu) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "raw tag data must start with an 8-byte type header");
        return NULL;
    }
    lcms2_clear_error();
    int ok = cmsWriteRawTag(h, (cmsTagSignature)sig, buf.buf, (cmsUInt32Number)buf.len);
    PyBuffer_Release(&buf);
    if (!ok)
        return lcms2_raise("cannot write the tag");
    Py_RETURN_NONE;
}

/* Readers ---------------------------------------------------------------------------------------- */

PyObject *
lcms2_profile_read_xyz(PyObject *self, PyObject *args)
{
    (void)self;
    cmsHPROFILE h;
    cmsUInt32Number sig;
    if (parse_profile_sig(args, "OO:profile_read_xyz", &h, &sig) < 0)
        return NULL;
    if (sig != cmsSigMediaWhitePointTag && sig != cmsSigMediaBlackPointTag && sig != cmsSigRedColorantTag &&
        sig != cmsSigGreenColorantTag && sig != cmsSigBlueColorantTag && sig != cmsSigLuminanceTag) {
        PyErr_SetString(PyExc_ValueError, "XYZ tags are 'wtpt', 'bkpt', 'rXYZ', 'gXYZ', 'bXYZ' and 'lumi'");
        return NULL;
    }
    cmsCIEXYZ *v = (cmsCIEXYZ *)cmsReadTag(h, (cmsTagSignature)sig);
    if (v == NULL) {
        lcms2_clear_error();
        Py_RETURN_NONE;
    }
    return Py_BuildValue("(ddd)", v->X, v->Y, v->Z);
}

PyObject *
lcms2_profile_read_chad(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHPROFILE h = lcms2_get_profile(arg);
    if (h == NULL)
        return NULL;
    const cmsFloat64Number *m = (const cmsFloat64Number *)cmsReadTag(h, cmsSigChromaticAdaptationTag);
    if (m == NULL) {
        lcms2_clear_error();
        Py_RETURN_NONE;
    }
    return Py_BuildValue("((ddd)(ddd)(ddd))", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
}

PyObject *
lcms2_profile_read_trc(PyObject *self, PyObject *args)
{
    (void)self;
    cmsHPROFILE h;
    cmsUInt32Number sig;
    if (parse_profile_sig(args, "OO:profile_read_trc", &h, &sig) < 0)
        return NULL;
    if (sig != cmsSigRedTRCTag && sig != cmsSigGreenTRCTag && sig != cmsSigBlueTRCTag &&
        sig != cmsSigGrayTRCTag) {
        PyErr_SetString(PyExc_ValueError, "TRC tags are 'rTRC', 'gTRC', 'bTRC' and 'kTRC'");
        return NULL;
    }
    const cmsToneCurve *c = (const cmsToneCurve *)cmsReadTag(h, (cmsTagSignature)sig);
    if (c == NULL) {
        lcms2_clear_error();
        Py_RETURN_NONE;
    }
    lcms2_clear_error();
    return lcms2_wrap_curve(cmsDupToneCurve(c));
}

PyObject *
lcms2_profile_is_matrix_shaper(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHPROFILE h = lcms2_get_profile(arg);
    if (h == NULL)
        return NULL;
    return PyBool_FromLong(cmsIsMatrixShaper(h));
}

static int
parse_intent_direction(PyObject *args, const char *fmt, cmsHPROFILE *h, int *intent, int *direction)
{
    PyObject *p;
    if (!PyArg_ParseTuple(args, fmt, &p, intent, direction))
        return -1;
    if ((*h = lcms2_get_profile(p)) == NULL)
        return -1;
    if (*intent < 0 || *direction < 0 || *direction > 2) {
        PyErr_SetString(PyExc_ValueError, "intent must be >= 0 and direction 0 (input), 1 (output) or 2 (proof)");
        return -1;
    }
    return 0;
}

PyObject *
lcms2_profile_is_clut(PyObject *self, PyObject *args)
{
    (void)self;
    cmsHPROFILE h;
    int intent, direction;
    if (parse_intent_direction(args, "Oii:profile_is_clut", &h, &intent, &direction) < 0)
        return NULL;
    return PyBool_FromLong(cmsIsCLUT(h, (cmsUInt32Number)intent, (cmsUInt32Number)direction));
}

PyObject *
lcms2_profile_is_intent_supported(PyObject *self, PyObject *args)
{
    (void)self;
    cmsHPROFILE h;
    int intent, direction;
    if (parse_intent_direction(args, "Oii:profile_is_intent_supported", &h, &intent, &direction) < 0)
        return NULL;
    return PyBool_FromLong(cmsIsIntentSupported(h, (cmsUInt32Number)intent, (cmsUInt32Number)direction));
}

PyObject *
lcms2_profile_supported_intents(PyObject *self, PyObject *noargs)
{
    (void)self;
    (void)noargs;
    cmsUInt32Number codes[64];
    char *descs[64];
    cmsUInt32Number n = cmsGetSupportedIntents(64, codes, descs);
    PyObject *out = PyList_New(0);
    if (out == NULL)
        return NULL;
    for (cmsUInt32Number i = 0; i < n && i < 64; i++) {
        PyObject *item = Py_BuildValue("(ks)", (unsigned long)codes[i], descs[i] ? descs[i] : "");
        if (item == NULL || PyList_Append(out, item) < 0) {
            Py_XDECREF(item);
            Py_DECREF(out);
            return NULL;
        }
        Py_DECREF(item);
    }
    return out;
}

PyObject *
lcms2_profile_detect_black_point(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static char *kwlist[] = {"profile", "intent", "flags", "destination", NULL};
    PyObject *p;
    int intent, destination = 0;
    unsigned long flags = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|kp:profile_detect_black_point", kwlist, &p, &intent, &flags,
                                     &destination))
        return NULL;
    cmsHPROFILE h = lcms2_get_profile(p);
    if (h == NULL)
        return NULL;
    if (intent < 0) {
        PyErr_SetString(PyExc_ValueError, "intent must be >= 0");
        return NULL;
    }
    cmsCIEXYZ bp = {0.0, 0.0, 0.0};
    lcms2_clear_error();
    int ok = destination ? cmsDetectDestinationBlackPoint(&bp, h, (cmsUInt32Number)intent, (cmsUInt32Number)flags)
                         : cmsDetectBlackPoint(&bp, h, (cmsUInt32Number)intent, (cmsUInt32Number)flags);
    lcms2_clear_error();
    if (!ok)
        Py_RETURN_NONE;
    return Py_BuildValue("(ddd)", bp.X, bp.Y, bp.Z);
}

PyObject *
lcms2_profile_detect_tac(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHPROFILE h = lcms2_get_profile(arg);
    if (h == NULL)
        return NULL;
    double tac;
    Py_BEGIN_ALLOW_THREADS
    tac = cmsDetectTAC(h);
    Py_END_ALLOW_THREADS
    lcms2_clear_error();
    return PyFloat_FromDouble(tac);
}

PyObject *
lcms2_profile_named_colors(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHPROFILE h = lcms2_get_profile(arg);
    if (h == NULL)
        return NULL;
    if (!cmsIsTag(h, cmsSigNamedColor2Tag))
        Py_RETURN_NONE;
    lcms2_clear_error();
    cmsNAMEDCOLORLIST *list = (cmsNAMEDCOLORLIST *)cmsReadTag(h, cmsSigNamedColor2Tag);
    if (list == NULL)
        return lcms2_raise("cannot read the named colour list");
    cmsInt32Number nch = cmsChannelsOfColorSpace(cmsGetColorSpace(h));
    if (nch < 0 || nch > cmsMAXCHANNELS)
        nch = 0;
    cmsUInt32Number n = cmsNamedColorCount(list);
    PyObject *out = PyList_New(0);
    if (out == NULL)
        return NULL;
    for (cmsUInt32Number i = 0; i < n; i++) {
        char name[cmsMAX_PATH] = {0}, prefix[33] = {0}, suffix[33] = {0};
        cmsUInt16Number pcs[3] = {0}, colorant[cmsMAXCHANNELS] = {0};
        if (!cmsNamedColorInfo(list, i, name, prefix, suffix, pcs, colorant))
            continue;
        PyObject *cols = PyTuple_New(nch);
        if (cols == NULL)
            goto fail;
        for (cmsInt32Number k = 0; k < nch; k++) {
            PyObject *v = PyLong_FromLong(colorant[k]);
            if (v == NULL) {
                Py_DECREF(cols);
                goto fail;
            }
            PyTuple_SET_ITEM(cols, k, v);
        }
        PyObject *item = Py_BuildValue("(sss(iii)N)", name, prefix, suffix, pcs[0], pcs[1], pcs[2], cols);
        if (item == NULL || PyList_Append(out, item) < 0) {
            Py_XDECREF(item);
            goto fail;
        }
        Py_DECREF(item);
    }
    return out;
fail:
    Py_DECREF(out);
    return NULL;
}
