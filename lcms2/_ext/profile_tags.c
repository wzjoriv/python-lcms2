/* Profiles: tags, tag readers and capability queries (registered in profile.c). */
#include "profile.h"

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
