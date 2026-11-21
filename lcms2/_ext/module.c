/* Module init for lcms2._lcms2: concatenates the per-file method tables. */
#define LCMS2_MODULE_MAIN
#include "common.h"

static PyMethodDef *method_tables[] = {
    lcms2_errors_methods,
    lcms2_formats_methods,
    lcms2_profile_methods,
    lcms2_transform_methods,
    lcms2_curves_methods,
    lcms2_colorimetry_methods,
    lcms2_cgats_methods,
};

static PyMethodDef *all_methods = NULL;

static struct PyModuleDef lcms2_def = {
    PyModuleDef_HEAD_INIT,
    "_lcms2",
    "Low-level LittleCMS2 bindings. Use the lcms2 package instead.",
    -1,
    NULL,
};

static int
build_method_table(void)
{
    size_t total = 0, n = sizeof(method_tables) / sizeof(method_tables[0]);
    for (size_t i = 0; i < n; i++)
        for (PyMethodDef *m = method_tables[i]; m->ml_name != NULL; m++)
            total++;
    all_methods = PyMem_Calloc(total + 1, sizeof(PyMethodDef));
    if (all_methods == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    size_t k = 0;
    for (size_t i = 0; i < n; i++)
        for (PyMethodDef *m = method_tables[i]; m->ml_name != NULL; m++)
            all_methods[k++] = *m;
    return 0;
}

PyMODINIT_FUNC
PyInit__lcms2(void)
{
    import_array();

    if (all_methods == NULL && build_method_table() < 0)
        return NULL;
    lcms2_def.m_methods = all_methods;

    PyObject *m = PyModule_Create(&lcms2_def);
    if (m == NULL)
        return NULL;
#ifdef Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    lcms2_Error = PyErr_NewException("lcms2._lcms2.LCMSError", PyExc_ValueError, NULL);
    if (lcms2_Error == NULL || PyModule_AddObjectRef(m, "LCMSError", lcms2_Error) < 0)
        goto fail;
    if (PyModule_AddIntConstant(m, "LCMS_VERSION", LCMS_VERSION) < 0)
        goto fail;
    if (lcms2_formats_init(m) < 0)
        goto fail;

    lcms2_install_error_handler();
    return m;

fail:
    Py_DECREF(m);
    return NULL;
}
