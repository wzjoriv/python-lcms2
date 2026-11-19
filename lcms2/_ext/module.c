#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

#if PY_VERSION_HEX < 0x030c0000
  #define Py_T_INT        T_INT
  #define Py_T_OBJECT_EX  T_OBJECT_EX
#endif

#include <numpy/arrayobject.h>
#include <lcms2.h>
#include <lcms2_internal.h>


#include "profile.h"
#include "transform.h"
#include "definitions.h"
#include "white_point.h"

static PyObject *
get_version (PyObject *self, PyObject *args) {
	return Py_BuildValue("i",  LCMS_VERSION);
}

static
PyMethodDef pycms_methods[] = {
    {"create_profile", create_profile, METH_VARARGS},
    {"create_default_profile", create_default_profile, METH_NOARGS},
    {"create_rgb_profile", create_rgb_profile, METH_VARARGS},
    {"open_profile", open_profile, METH_VARARGS},
    {"profile_from_memory", profile_from_memory, METH_VARARGS},
    {"profile_to_bytes", profile_to_bytes, METH_VARARGS},
    {"white_point_from_temperature", white_point_from_temperature, METH_VARARGS},
	{"get_version", get_version, METH_NOARGS},
	{NULL, NULL}
};


static struct PyModuleDef lcms2_def =
{
    PyModuleDef_HEAD_INIT,
    "_lcms2",     /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    pycms_methods
};

PyMODINIT_FUNC PyInit__lcms2(void)
{
    PyObject *m;
    m = PyModule_Create(&lcms2_def);
    if (m == NULL)
        return NULL;    
		
	if (PyType_Ready(&profile_type) < 0) {
        Py_DECREF(m);
		return NULL;
	}

	if (PyModule_AddObjectRef(m, "Profile", (PyObject *) &profile_type) < 0) {
		Py_DECREF(m);
		return NULL;
	}

	if (PyType_Ready(&transform_type) < 0) {
		Py_DECREF(m);
		return NULL;
	}

	if (PyModule_AddObjectRef(m, "Transform", (PyObject *) &transform_type) < 0) {
        Py_DECREF(m);
        return NULL;
    }

	PyObject* types_dict = create_data_types_dict();
	if (types_dict == NULL)
	{
        Py_DECREF(m);
        return NULL;		
	}
	if (PyModule_AddObjectRef(m, "DATA_TYPES", (PyObject *) types_dict) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    import_array(); // initialize  NumPy

    return m;
}
