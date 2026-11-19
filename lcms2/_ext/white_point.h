#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <lcms2.h>

static PyObject *
white_point_from_temperature(PyObject *self, PyObject *args)
{
    double temp;
    if (!PyArg_ParseTuple(args, "d", &temp))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid argument - expected temperature in Kelvin");
        return NULL;
    }

    cmsCIExyY wp;
    if (cmsWhitePointFromTemp(&wp, temp))
    {
        return Py_BuildValue("[d,d,d]", wp.x, wp.y, wp.Y);
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Unable to convert temperature to white point coordinates");
        return NULL;
    }
}
