/* LittleCMS error capture and handle (capsule) helpers. */
#include "common.h"

#include <string.h>

#if defined(_MSC_VER)
#define LCMS2_THREAD_LOCAL __declspec(thread)
#else
#define LCMS2_THREAD_LOCAL _Thread_local
#endif

PyObject *lcms2_Error = NULL;

static LCMS2_THREAD_LOCAL char last_error[512];
static LCMS2_THREAD_LOCAL int has_error = 0;

static void
log_error_handler(cmsContext ctx, cmsUInt32Number code, const char *text)
{
    (void)ctx;
    (void)code;
    if (text == NULL)
        text = "unknown LittleCMS error";
    strncpy(last_error, text, sizeof(last_error) - 1);
    last_error[sizeof(last_error) - 1] = '\0';
    has_error = 1;
}

void
lcms2_install_error_handler(void)
{
    cmsSetLogErrorHandler(log_error_handler);
}

void
lcms2_clear_error(void)
{
    has_error = 0;
    last_error[0] = '\0';
}

PyObject *
lcms2_raise(const char *fallback)
{
    if (PyErr_Occurred())
        return NULL;
    if (has_error)
        PyErr_SetString(lcms2_Error, last_error);
    else
        PyErr_SetString(lcms2_Error, fallback ? fallback : "LittleCMS operation failed");
    lcms2_clear_error();
    return NULL;
}

/* Handles ---------------------------------------------------------------------------------- */

static void *
get_handle(PyObject *obj, const char *name, const char *what)
{
    PyObject *cap = obj;
    int owned = 0;
    if (!PyCapsule_CheckExact(obj)) {
        cap = PyObject_GetAttrString(obj, "_handle");
        if (cap == NULL) {
            PyErr_Format(PyExc_TypeError, "expected a %s, got %.200s", what, Py_TYPE(obj)->tp_name);
            return NULL;
        }
        owned = 1;
    }
    void *ptr = NULL;
    if (cap == Py_None)
        PyErr_Format(PyExc_ValueError, "the %s has been closed", what);
    else if (!PyCapsule_IsValid(cap, name))
        PyErr_Format(PyExc_TypeError, "expected a %s handle", what);
    else
        ptr = PyCapsule_GetPointer(cap, name);
    /* The capsule stays alive through obj; callers must keep obj alive while using ptr. */
    if (owned)
        Py_DECREF(cap);
    return ptr;
}

cmsHPROFILE   lcms2_get_profile(PyObject *o)   { return get_handle(o, LCMS2_CAPSULE_PROFILE, "Profile"); }
cmsHTRANSFORM lcms2_get_transform(PyObject *o) { return get_handle(o, LCMS2_CAPSULE_TRANSFORM, "Transform"); }
cmsToneCurve *lcms2_get_curve(PyObject *o)     { return get_handle(o, LCMS2_CAPSULE_CURVE, "ToneCurve"); }
cmsHANDLE     lcms2_get_it8(PyObject *o)       { return get_handle(o, LCMS2_CAPSULE_IT8, "CGATS"); }

static void free_profile(PyObject *c)   { cmsCloseProfile(PyCapsule_GetPointer(c, LCMS2_CAPSULE_PROFILE)); }
static void free_transform(PyObject *c) { cmsDeleteTransform(PyCapsule_GetPointer(c, LCMS2_CAPSULE_TRANSFORM)); }
static void free_curve(PyObject *c)     { cmsFreeToneCurve(PyCapsule_GetPointer(c, LCMS2_CAPSULE_CURVE)); }
static void free_it8(PyObject *c)       { cmsIT8Free(PyCapsule_GetPointer(c, LCMS2_CAPSULE_IT8)); }

#define WRAP(fn, type, name, freefn, closefn)                                  \
    PyObject *fn(type h)                                                       \
    {                                                                          \
        if (h == NULL)                                                         \
            return lcms2_raise(NULL);                                          \
        PyObject *cap = PyCapsule_New((void *)h, name, freefn);                \
        if (cap == NULL)                                                       \
            closefn(h);                                                        \
        return cap;                                                            \
    }

WRAP(lcms2_wrap_profile, cmsHPROFILE, LCMS2_CAPSULE_PROFILE, free_profile, cmsCloseProfile)
WRAP(lcms2_wrap_transform, cmsHTRANSFORM, LCMS2_CAPSULE_TRANSFORM, free_transform, cmsDeleteTransform)
WRAP(lcms2_wrap_curve, cmsToneCurve *, LCMS2_CAPSULE_CURVE, free_curve, cmsFreeToneCurve)
WRAP(lcms2_wrap_it8, cmsHANDLE, LCMS2_CAPSULE_IT8, free_it8, cmsIT8Free)

/* Python-visible ----------------------------------------------------------------------------- */

static PyObject *
py_get_version(PyObject *self, PyObject *noargs)
{
    (void)self;
    (void)noargs;
    return PyLong_FromLong(cmsGetEncodedCMMversion());
}

PyMethodDef lcms2_errors_methods[] = {
    {"get_version", py_get_version, METH_NOARGS,
     "get_version() -> int\n\nEncoded LittleCMS version of the linked library (2200 for 2.20)."},
    {NULL, NULL, 0, NULL},
};
