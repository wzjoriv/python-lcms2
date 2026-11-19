/* CGATS / IT8 bindings. Handles are capsules (LCMS2_CAPSULE_IT8); the Python class keeps the
 * current table index, since the library has no getter for it. */
#include "common.h"

#include <stdlib.h>
#include <string.h>

static int
table_dims(cmsHANDLE h, int *sets, int *fields)
{
    const char *s = cmsIT8GetProperty(h, "NUMBER_OF_SETS");
    const char *f = cmsIT8GetProperty(h, "NUMBER_OF_FIELDS");
    *sets = s ? atoi(s) : 0;
    *fields = f ? atoi(f) : 0;
    return 0;
}

static PyObject *
py_it8_new(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    lcms2_clear_error();
    return lcms2_wrap_it8(cmsIT8Alloc(NULL));
}

static PyObject *
py_it8_load(PyObject *self, PyObject *arg)
{
    Py_buffer view;
    (void)self;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0)
        return NULL;
    if (view.len == 0 || view.len > 0x7fffffff) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "the CGATS data is empty or larger than 2 GiB");
        return NULL;
    }
    lcms2_clear_error();
    cmsHANDLE h = cmsIT8LoadFromMem(NULL, view.buf, (cmsUInt32Number)view.len);
    PyBuffer_Release(&view);
    if (h == NULL)
        return lcms2_raise("could not parse the data as a CGATS/IT8 file");
    return lcms2_wrap_it8(h);
}

static PyObject *
py_it8_save(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHANDLE h = lcms2_get_it8(arg);
    if (h == NULL)
        return NULL;
    lcms2_clear_error();
    cmsUInt32Number need = 0;
    if (!cmsIT8SaveToMem(h, NULL, &need) || need < 1)
        return lcms2_raise("could not serialise the CGATS data");
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)need);
    if (out == NULL)
        return NULL;
    cmsUInt32Number size = need;
    if (!cmsIT8SaveToMem(h, PyBytes_AS_STRING(out), &size)) {
        Py_DECREF(out);
        return lcms2_raise("could not serialise the CGATS data");
    }
    /* Drop the terminating NUL. */
    if (_PyBytes_Resize(&out, (Py_ssize_t)need - 1) < 0)
        return NULL;
    return out;
}

static PyObject *
py_it8_table_count(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHANDLE h = lcms2_get_it8(arg);
    if (h == NULL)
        return NULL;
    return PyLong_FromUnsignedLong(cmsIT8TableCount(h));
}

static PyObject *
py_it8_set_table(PyObject *self, PyObject *args)
{
    PyObject *hobj;
    int n;
    (void)self;
    if (!PyArg_ParseTuple(args, "Oi", &hobj, &n))
        return NULL;
    cmsHANDLE h = lcms2_get_it8(hobj);
    if (h == NULL)
        return NULL;
    if (n < 0 || (cmsUInt32Number)n > cmsIT8TableCount(h)) {
        PyErr_Format(PyExc_IndexError, "table %d out of range; the file has %u table(s)", n,
                     (unsigned)cmsIT8TableCount(h));
        return NULL;
    }
    lcms2_clear_error();
    if (cmsIT8SetTable(h, (cmsUInt32Number)n) < 0)
        return lcms2_raise("could not select the table");
    Py_RETURN_NONE;
}

static PyObject *
py_it8_sheet_type(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHANDLE h = lcms2_get_it8(arg);
    if (h == NULL)
        return NULL;
    const char *s = cmsIT8GetSheetType(h);
    if (s == NULL)
        Py_RETURN_NONE;
    return PyUnicode_FromString(s);
}

static PyObject *
py_it8_set_sheet_type(PyObject *self, PyObject *args)
{
    PyObject *hobj;
    const char *s;
    (void)self;
    if (!PyArg_ParseTuple(args, "Os", &hobj, &s))
        return NULL;
    cmsHANDLE h = lcms2_get_it8(hobj);
    if (h == NULL)
        return NULL;
    lcms2_clear_error();
    if (!cmsIT8SetSheetType(h, s))
        return lcms2_raise("could not set the sheet type");
    Py_RETURN_NONE;
}

static PyObject *
py_it8_properties(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHANDLE h = lcms2_get_it8(arg);
    if (h == NULL)
        return NULL;
    char **names = NULL;
    cmsUInt32Number n = cmsIT8EnumProperties(h, &names);
    PyObject *list = PyList_New((Py_ssize_t)n);
    if (list == NULL)
        return NULL;
    for (cmsUInt32Number i = 0; i < n; i++) {
        PyObject *s = PyUnicode_FromString(names[i]);
        if (s == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, (Py_ssize_t)i, s);
    }
    return list;
}

static PyObject *
py_it8_get_property(PyObject *self, PyObject *args)
{
    PyObject *hobj;
    const char *name;
    (void)self;
    if (!PyArg_ParseTuple(args, "Os", &hobj, &name))
        return NULL;
    cmsHANDLE h = lcms2_get_it8(hobj);
    if (h == NULL)
        return NULL;
    const char *v = cmsIT8GetProperty(h, name);
    if (v == NULL)
        Py_RETURN_NONE;
    return PyUnicode_FromString(v);
}

/* it8_set_property(h, name, value, quoted): quoted strings are written between quotes. */
static PyObject *
py_it8_set_property(PyObject *self, PyObject *args)
{
    PyObject *hobj;
    const char *name, *value;
    int quoted;
    (void)self;
    if (!PyArg_ParseTuple(args, "Ossp", &hobj, &name, &value, &quoted))
        return NULL;
    cmsHANDLE h = lcms2_get_it8(hobj);
    if (h == NULL)
        return NULL;
    lcms2_clear_error();
    cmsBool ok = quoted ? cmsIT8SetPropertyStr(h, name, value) : cmsIT8SetPropertyUncooked(h, name, value);
    if (!ok)
        return lcms2_raise("could not set the property (NUMBER_OF_FIELDS and NUMBER_OF_SETS can be set once per table)");
    Py_RETURN_NONE;
}

static PyObject *
py_it8_data_format(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHANDLE h = lcms2_get_it8(arg);
    if (h == NULL)
        return NULL;
    char **names = NULL;
    int n = cmsIT8EnumDataFormat(h, &names);
    PyObject *list = PyList_New(n > 0 ? n : 0);
    if (list == NULL)
        return NULL;
    for (int i = 0; i < n; i++) {
        PyObject *s = names && names[i] ? PyUnicode_FromString(names[i]) : PyUnicode_FromString("");
        if (s == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, s);
    }
    return list;
}

/* it8_define_table(h, names, n_rows): declare the columns and row count of the current table. */
static PyObject *
py_it8_define_table(PyObject *self, PyObject *args)
{
    PyObject *hobj, *names;
    int rows;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOi", &hobj, &names, &rows))
        return NULL;
    cmsHANDLE h = lcms2_get_it8(hobj);
    if (h == NULL)
        return NULL;
    PyObject *seq = PySequence_Fast(names, "column names must be a sequence of strings");
    if (seq == NULL)
        return NULL;
    Py_ssize_t ncol = PySequence_Fast_GET_SIZE(seq);
    if (ncol < 1 || rows < 0) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "a table needs at least one column and a non-negative row count");
        return NULL;
    }
    char buf[32];
    lcms2_clear_error();
    snprintf(buf, sizeof buf, "%zd", ncol);
    if (!cmsIT8SetPropertyUncooked(h, "NUMBER_OF_FIELDS", buf))
        goto fail;
    snprintf(buf, sizeof buf, "%d", rows);
    if (!cmsIT8SetPropertyUncooked(h, "NUMBER_OF_SETS", buf))
        goto fail;
    for (Py_ssize_t i = 0; i < ncol; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        const char *s = PyUnicode_Check(item) ? PyUnicode_AsUTF8(item) : NULL;
        if (s == NULL) {
            Py_DECREF(seq);
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_TypeError, "column names must be strings");
            return NULL;
        }
        if (!cmsIT8SetDataFormat(h, (int)i, s))
            goto fail;
    }
    Py_DECREF(seq);
    Py_RETURN_NONE;
fail:
    Py_DECREF(seq);
    return lcms2_raise("could not define the table (it already has columns or rows; use a new table)");
}

static PyObject *
py_it8_dims(PyObject *self, PyObject *arg)
{
    (void)self;
    cmsHANDLE h = lcms2_get_it8(arg);
    if (h == NULL)
        return NULL;
    int sets, fields;
    table_dims(h, &sets, &fields);
    return Py_BuildValue("(ii)", sets, fields);
}

static PyObject *
py_it8_get_cell(PyObject *self, PyObject *args)
{
    PyObject *hobj;
    int row, col;
    (void)self;
    if (!PyArg_ParseTuple(args, "Oii", &hobj, &row, &col))
        return NULL;
    cmsHANDLE h = lcms2_get_it8(hobj);
    if (h == NULL)
        return NULL;
    const char *v = cmsIT8GetDataRowCol(h, row, col);
    if (v == NULL)
        Py_RETURN_NONE;
    return PyUnicode_FromString(v);
}

static PyObject *
py_it8_set_cell(PyObject *self, PyObject *args)
{
    PyObject *hobj;
    int row, col;
    const char *value;
    (void)self;
    if (!PyArg_ParseTuple(args, "Oiis", &hobj, &row, &col, &value))
        return NULL;
    cmsHANDLE h = lcms2_get_it8(hobj);
    if (h == NULL)
        return NULL;
    int sets, fields;
    table_dims(h, &sets, &fields);
    if (row < 0 || row >= sets || col < 0 || col >= fields) {
        PyErr_Format(PyExc_IndexError, "cell (%d, %d) is outside the table (%d rows x %d columns)", row, col,
                     sets, fields);
        return NULL;
    }
    lcms2_clear_error();
    if (!cmsIT8SetDataRowCol(h, row, col, value))
        return lcms2_raise("could not set the cell");
    Py_RETURN_NONE;
}

PyMethodDef lcms2_cgats_methods[] = {
    {"it8_new", py_it8_new, METH_NOARGS, "it8_new() -> capsule\n\nCreate an empty CGATS.17 container."},
    {"it8_load", py_it8_load, METH_O,
     "it8_load(data) -> capsule\n\nParse CGATS/IT8 text from a bytes-like object."},
    {"it8_save", py_it8_save, METH_O, "it8_save(it8) -> bytes\n\nSerialise every table to CGATS text."},
    {"it8_table_count", py_it8_table_count, METH_O, "it8_table_count(it8) -> int\n\nNumber of tables."},
    {"it8_set_table", py_it8_set_table, METH_VARARGS,
     "it8_set_table(it8, n) -> None\n\nSelect table n (n equal to the count appends a new table)."},
    {"it8_sheet_type", py_it8_sheet_type, METH_O,
     "it8_sheet_type(it8) -> str | None\n\nThe sheet type of the current table."},
    {"it8_set_sheet_type", py_it8_set_sheet_type, METH_VARARGS,
     "it8_set_sheet_type(it8, sheet_type) -> None\n\nSet the sheet type of the current table."},
    {"it8_properties", py_it8_properties, METH_O,
     "it8_properties(it8) -> list\n\nProperty names of the current table."},
    {"it8_get_property", py_it8_get_property, METH_VARARGS,
     "it8_get_property(it8, name) -> str | None\n\nValue of a property of the current table."},
    {"it8_set_property", py_it8_set_property, METH_VARARGS,
     "it8_set_property(it8, name, value, quoted) -> None\n\nSet a property of the current table."},
    {"it8_data_format", py_it8_data_format, METH_O,
     "it8_data_format(it8) -> list\n\nColumn names of the current table."},
    {"it8_define_table", py_it8_define_table, METH_VARARGS,
     "it8_define_table(it8, names, rows) -> None\n\nDeclare columns and row count of the current table."},
    {"it8_dims", py_it8_dims, METH_O,
     "it8_dims(it8) -> tuple\n\n(rows, columns) of the current table."},
    {"it8_get_cell", py_it8_get_cell, METH_VARARGS,
     "it8_get_cell(it8, row, col) -> str | None\n\nCell text, or None when empty."},
    {"it8_set_cell", py_it8_set_cell, METH_VARARGS,
     "it8_set_cell(it8, row, col, value) -> None\n\nSet a cell from text."},
    {NULL, NULL, 0, NULL},
};
