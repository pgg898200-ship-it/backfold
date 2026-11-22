/* Optional CPython binding. All record decisions live in eventfold.h. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "eventfold.h"

static PyObject *encode(PyObject *self, PyObject *args) {
    PyObject *previous, *frame;
    size_t n, capacity, written;
    unsigned char small[1024], *out;
    const unsigned char *before, *now;
    (void)self;
    if (!PyArg_ParseTuple(args, "SS:encode", &previous, &frame)) return NULL;
    n = (size_t)PyBytes_GET_SIZE(frame);
    capacity = ef_bound(n);
    if (!capacity || capacity > (size_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "frame exceeds the supported record size");
        return NULL;
    }
    out = capacity <= sizeof small ? small : PyMem_Malloc(capacity);
    if (!out) return PyErr_NoMemory();
    before = (const unsigned char *)PyBytes_AS_STRING(previous);
    now = (const unsigned char *)PyBytes_AS_STRING(frame);
    size_t previous_size = (size_t)PyBytes_GET_SIZE(previous);
    if (n >= 65536) {
        Py_BEGIN_ALLOW_THREADS
        written = ef_encode(before, previous_size, now, n, out, capacity);
        Py_END_ALLOW_THREADS
    } else {
        written = ef_encode(before, previous_size, now, n, out, capacity);
    }
    PyObject *result = NULL;
    if (written) {
        result = PyBytes_FromStringAndSize((const char *)out, (Py_ssize_t)written);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "EFL1 encoder rejected a validated frame");
    }
    if (out != small) PyMem_Free(out);
    return result;
}

static PyMethodDef methods[] = {
    {"encode", encode, METH_VARARGS, "Encode one EFL1 record from two bytes objects."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_native", "EFL1 record encoder.", -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module);
}
