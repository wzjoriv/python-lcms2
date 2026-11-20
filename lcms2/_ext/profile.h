/* Private declarations shared by profile.c and profile_info.c. */
#ifndef LCMS2_EXT_PROFILE_H
#define LCMS2_EXT_PROFILE_H

#include "common.h"

/* A 4-character signature ("RGB ", "mntr", "desc"; shorter strings are space-padded) or an int.
 * Returns 0 on success, -1 with an exception set. */
int lcms2_parse_signature(PyObject *obj, cmsUInt32Number *out, const char *what);

/* The 4-character string of a signature (new reference). */
PyObject *lcms2_signature_str(cmsUInt32Number sig);

/* Accessors defined in profile_info.c, registered in profile.c's method table. */
PyObject *lcms2_profile_get_header(PyObject *self, PyObject *arg);
PyObject *lcms2_profile_set_header(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *lcms2_profile_compute_id(PyObject *self, PyObject *arg);
PyObject *lcms2_profile_get_info(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *lcms2_profile_set_info(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *lcms2_profile_tags(PyObject *self, PyObject *arg);
PyObject *lcms2_profile_is_tag(PyObject *self, PyObject *args);
PyObject *lcms2_profile_read_raw_tag(PyObject *self, PyObject *args);
PyObject *lcms2_profile_write_raw_tag(PyObject *self, PyObject *args);
PyObject *lcms2_profile_read_xyz(PyObject *self, PyObject *args);
PyObject *lcms2_profile_read_chad(PyObject *self, PyObject *arg);
PyObject *lcms2_profile_read_trc(PyObject *self, PyObject *args);
PyObject *lcms2_profile_is_matrix_shaper(PyObject *self, PyObject *arg);
PyObject *lcms2_profile_is_clut(PyObject *self, PyObject *args);
PyObject *lcms2_profile_is_intent_supported(PyObject *self, PyObject *args);
PyObject *lcms2_profile_supported_intents(PyObject *self, PyObject *noargs);
PyObject *lcms2_profile_detect_black_point(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *lcms2_profile_detect_tac(PyObject *self, PyObject *arg);
PyObject *lcms2_profile_named_colors(PyObject *self, PyObject *arg);

#endif
