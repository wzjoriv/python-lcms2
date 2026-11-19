/* Shared declarations for the lcms2._lcms2 extension.
 *
 * Every .c file in lcms2/_ext includes this header first. module.c is the only file that
 * defines LCMS2_MODULE_MAIN (it owns the NumPy C-API table and the module init).
 *
 * Handles cross the C/Python boundary as PyCapsules named LCMS2_CAPSULE_*; the Python layer
 * wraps them in classes (lcms2/_profile.py, lcms2/_transform.py, lcms2/_curves.py) and
 * stores the capsule in the attribute "_handle". Functions taking a handle accept either the
 * capsule or any object with a "_handle" attribute.
 */
#ifndef LCMS2_EXT_COMMON_H
#define LCMS2_EXT_COMMON_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL lcms2_ARRAY_API
#ifndef LCMS2_MODULE_MAIN
#define NO_IMPORT_ARRAY
#endif
#include <numpy/arrayobject.h>

#include <lcms2.h>

#define LCMS2_CAPSULE_PROFILE   "lcms2.profile"
#define LCMS2_CAPSULE_TRANSFORM "lcms2.transform"
#define LCMS2_CAPSULE_CURVE     "lcms2.curve"
#define LCMS2_CAPSULE_IT8       "lcms2.it8"

/* errors.c ------------------------------------------------------------------------------- */

/* lcms2._lcms2.LCMSError, created in module init. */
extern PyObject *lcms2_Error;

/* Install the LittleCMS log handler that records the last error message per thread. */
void lcms2_install_error_handler(void);

/* Forget any recorded LittleCMS error (call before an lcms operation you will check). */
void lcms2_clear_error(void);

/* Raise LCMSError with the recorded LittleCMS message, or `fallback` when none was recorded.
 * Always returns NULL so callers can `return lcms2_raise(...)`. */
PyObject *lcms2_raise(const char *fallback);

/* Handles: unwrap (borrowed, NULL + exception on failure) and wrap (new reference; on failure
 * the handle is released and NULL returned). */
cmsHPROFILE   lcms2_get_profile(PyObject *obj);
cmsHTRANSFORM lcms2_get_transform(PyObject *obj);
cmsToneCurve *lcms2_get_curve(PyObject *obj);
cmsHANDLE     lcms2_get_it8(PyObject *obj);

PyObject *lcms2_wrap_profile(cmsHPROFILE h);
PyObject *lcms2_wrap_transform(cmsHTRANSFORM h);
PyObject *lcms2_wrap_curve(cmsToneCurve *c);
PyObject *lcms2_wrap_it8(cmsHANDLE h);

/* Method tables, one per source file; module.c concatenates them. Each ends with {NULL}. */
extern PyMethodDef lcms2_errors_methods[];
extern PyMethodDef lcms2_formats_methods[];
extern PyMethodDef lcms2_profile_methods[];
extern PyMethodDef lcms2_transform_methods[];
extern PyMethodDef lcms2_curves_methods[];
extern PyMethodDef lcms2_colorimetry_methods[];
extern PyMethodDef lcms2_cgats_methods[];

/* Optional per-file module setup (constants, tables). Return 0 on success, -1 with an
 * exception set. */
int lcms2_formats_init(PyObject *module);

#endif
