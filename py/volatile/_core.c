#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "core/vol.h"
#include "core/imgproc.h"
#include <stdlib.h>
#include <string.h>

static int g_log_level = 1; /* default: INFO */

// ---------------------------------------------------------------------------
// PyCapsule helpers for volume*
// ---------------------------------------------------------------------------

#define VOL_CAPSULE_NAME "volatile.volume"

static void vol_capsule_destructor(PyObject *cap) {
  volume *v = (volume *)PyCapsule_GetPointer(cap, VOL_CAPSULE_NAME);
  if (v) vol_free(v);
}

static volume *capsule_to_vol(PyObject *cap) {
  if (!PyCapsule_IsValid(cap, VOL_CAPSULE_NAME)) {
    PyErr_SetString(PyExc_TypeError, "expected a vol capsule");
    return NULL;
  }
  return (volume *)PyCapsule_GetPointer(cap, VOL_CAPSULE_NAME);
}

// ---------------------------------------------------------------------------
// vol_open(path) -> capsule
// ---------------------------------------------------------------------------

static PyObject *
core_vol_open(PyObject *self, PyObject *args) {
  const char *path;
  if (!PyArg_ParseTuple(args, "s", &path)) return NULL;

  volume *v = vol_open(path);
  if (!v) {
    PyErr_Format(PyExc_OSError, "vol_open failed for path: %s", path);
    return NULL;
  }
  return PyCapsule_New(v, VOL_CAPSULE_NAME, vol_capsule_destructor);
}

// ---------------------------------------------------------------------------
// vol_free(capsule) -> None  (also invalidates capsule)
// ---------------------------------------------------------------------------

static PyObject *
core_vol_free(PyObject *self, PyObject *args) {
  PyObject *cap;
  if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;

  volume *v = capsule_to_vol(cap);
  if (!v) return NULL;

  vol_free(v);
  // Rename capsule so PyCapsule_IsValid with VOL_CAPSULE_NAME returns false,
  // preventing double-free. Also clear the destructor since we freed already.
  PyCapsule_SetDestructor(cap, NULL);
  PyCapsule_SetName(cap, "volatile.volume.freed");
  Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// vol_num_levels(capsule) -> int
// ---------------------------------------------------------------------------

static PyObject *
core_vol_num_levels(PyObject *self, PyObject *args) {
  PyObject *cap;
  if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;

  volume *v = capsule_to_vol(cap);
  if (!v) return NULL;

  return PyLong_FromLong(vol_num_levels(v));
}

// ---------------------------------------------------------------------------
// vol_shape(capsule, level) -> tuple of ints (length = ndim)
// ---------------------------------------------------------------------------

static PyObject *
core_vol_shape(PyObject *self, PyObject *args) {
  PyObject *cap;
  int level = 0;
  if (!PyArg_ParseTuple(args, "O|i", &cap, &level)) return NULL;

  volume *v = capsule_to_vol(cap);
  if (!v) return NULL;

  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m) {
    PyErr_Format(PyExc_IndexError, "level %d out of range", level);
    return NULL;
  }

  PyObject *tup = PyTuple_New((Py_ssize_t)m->ndim);
  if (!tup) return NULL;
  for (int d = 0; d < m->ndim; d++) {
    PyTuple_SET_ITEM(tup, d, PyLong_FromLongLong((long long)m->shape[d]));
  }
  return tup;
}

// ---------------------------------------------------------------------------
// vol_sample(capsule, level, z, y, x) -> float
// ---------------------------------------------------------------------------

static PyObject *
core_vol_sample(PyObject *self, PyObject *args) {
  PyObject *cap;
  int level = 0;
  double z = 0.0, y = 0.0, x = 0.0;
  if (!PyArg_ParseTuple(args, "O|iddd", &cap, &level, &z, &y, &x)) return NULL;

  volume *v = capsule_to_vol(cap);
  if (!v) return NULL;

  float val = vol_sample(v, level, (float)z, (float)y, (float)x);
  return PyFloat_FromDouble((double)val);
}

// ---------------------------------------------------------------------------
// imgproc bindings
// ---------------------------------------------------------------------------

// gaussian_blur_2d(data, height, width, sigma) -> bytes
static PyObject *
core_gaussian_blur_2d(PyObject *self, PyObject *args) {
  Py_buffer view;
  int height, width;
  double sigma;
  if (!PyArg_ParseTuple(args, "y*iid", &view, &height, &width, &sigma)) return NULL;

  size_t expected = (size_t)height * (size_t)width * sizeof(float);
  if ((size_t)view.len < expected) {
    PyBuffer_Release(&view);
    PyErr_Format(PyExc_ValueError, "buffer too small: need %zu bytes, got %zd", expected, view.len);
    return NULL;
  }

  float *out = (float *)malloc(expected);
  if (!out) { PyBuffer_Release(&view); return PyErr_NoMemory(); }

  gaussian_blur_2d((const float *)view.buf, out, height, width, (float)sigma);
  PyBuffer_Release(&view);

  PyObject *result = PyBytes_FromStringAndSize((const char *)out, (Py_ssize_t)expected);
  free(out);
  return result;
}

// structure_tensor_3d(data, depth, height, width, deriv_sigma, smooth_sigma) -> bytes
static PyObject *
core_structure_tensor_3d(PyObject *self, PyObject *args) {
  Py_buffer view;
  int depth, height, width;
  double deriv_sigma, smooth_sigma;
  if (!PyArg_ParseTuple(args, "y*iiidd", &view, &depth, &height, &width, &deriv_sigma, &smooth_sigma)) return NULL;

  size_t expected_in = (size_t)depth * (size_t)height * (size_t)width * sizeof(float);
  if ((size_t)view.len < expected_in) {
    PyBuffer_Release(&view);
    PyErr_Format(PyExc_ValueError, "buffer too small: need %zu bytes, got %zd", expected_in, view.len);
    return NULL;
  }

  size_t out_bytes = expected_in * 6;  // 6 channels
  float *out = (float *)malloc(out_bytes);
  if (!out) { PyBuffer_Release(&view); return PyErr_NoMemory(); }

  structure_tensor_3d((const float *)view.buf, out, depth, height, width, (float)deriv_sigma, (float)smooth_sigma);
  PyBuffer_Release(&view);

  PyObject *result = PyBytes_FromStringAndSize((const char *)out, (Py_ssize_t)out_bytes);
  free(out);
  return result;
}

// histogram(data, num_elements, num_bins) -> dict {bins: list[int], min: float, max: float, mean: float}
static PyObject *
core_histogram(PyObject *self, PyObject *args) {
  Py_buffer view;
  Py_ssize_t num_elements;
  int num_bins;
  if (!PyArg_ParseTuple(args, "y*ni", &view, &num_elements, &num_bins)) return NULL;

  size_t expected = (size_t)num_elements * sizeof(float);
  if ((size_t)view.len < expected) {
    PyBuffer_Release(&view);
    PyErr_Format(PyExc_ValueError, "buffer too small: need %zu bytes, got %zd", expected, view.len);
    return NULL;
  }

  histogram *h = histogram_new((const float *)view.buf, (size_t)num_elements, num_bins);
  PyBuffer_Release(&view);
  if (!h) return PyErr_NoMemory();

  // Build bins list
  PyObject *bins_list = PyList_New(num_bins);
  if (!bins_list) { histogram_free(h); return NULL; }
  for (int i = 0; i < num_bins; i++) {
    PyList_SET_ITEM(bins_list, i, PyLong_FromUnsignedLong((unsigned long)h->bins[i]));
  }

  float mean = histogram_mean(h);
  PyObject *result = PyDict_New();
  if (!result) { Py_DECREF(bins_list); histogram_free(h); return NULL; }

  PyDict_SetItemString(result, "bins", bins_list);
  Py_DECREF(bins_list);
  PyDict_SetItemString(result, "min",  PyFloat_FromDouble((double)h->min_val));
  PyDict_SetItemString(result, "max",  PyFloat_FromDouble((double)h->max_val));
  PyDict_SetItemString(result, "mean", PyFloat_FromDouble((double)mean));

  histogram_free(h);
  return result;
}

// window_level(data, num_elements, window, level) -> bytes (uint8)
static PyObject *
core_window_level(PyObject *self, PyObject *args) {
  Py_buffer view;
  Py_ssize_t num_elements;
  double window, level;
  if (!PyArg_ParseTuple(args, "y*ndd", &view, &num_elements, &window, &level)) return NULL;

  size_t expected_in = (size_t)num_elements * sizeof(float);
  if ((size_t)view.len < expected_in) {
    PyBuffer_Release(&view);
    PyErr_Format(PyExc_ValueError, "buffer too small: need %zu bytes, got %zd", expected_in, view.len);
    return NULL;
  }

  uint8_t *out = (uint8_t *)malloc((size_t)num_elements);
  if (!out) { PyBuffer_Release(&view); return PyErr_NoMemory(); }

  window_level((const float *)view.buf, out, (size_t)num_elements, (float)window, (float)level);
  PyBuffer_Release(&view);

  PyObject *result = PyBytes_FromStringAndSize((const char *)out, (Py_ssize_t)num_elements);
  free(out);
  return result;
}

static PyObject *
core_version(PyObject *self, PyObject *args) {
  return PyUnicode_FromString("0.1.0");
}

static PyObject *
core_log_set_level(PyObject *self, PyObject *args) {
  int level;
  if (!PyArg_ParseTuple(args, "i", &level)) {
    return NULL;
  }
  if (level < 0 || level > 4) {
    PyErr_SetString(PyExc_ValueError, "log level must be 0-4");
    return NULL;
  }
  g_log_level = level;
  Py_RETURN_NONE;
}

static PyObject *
core_log_get_level(PyObject *self, PyObject *args) {
  return PyLong_FromLong(g_log_level);
}

static PyMethodDef core_methods[] = {
  {"version",        core_version,        METH_NOARGS,  "Return volatile version string"},
  {"log_set_level",  core_log_set_level,  METH_VARARGS, "Set log level (0-4)"},
  {"log_get_level",  core_log_get_level,  METH_NOARGS,  "Get current log level"},
  {"vol_open",       core_vol_open,       METH_VARARGS, "vol_open(path) -> capsule"},
  {"vol_free",       core_vol_free,       METH_VARARGS, "vol_free(capsule) -> None"},
  {"vol_num_levels", core_vol_num_levels, METH_VARARGS, "vol_num_levels(capsule) -> int"},
  {"vol_shape",      core_vol_shape,      METH_VARARGS, "vol_shape(capsule, level=0) -> tuple"},
  {"vol_sample",          core_vol_sample,          METH_VARARGS, "vol_sample(capsule, level=0, z=0, y=0, x=0) -> float"},
  {"gaussian_blur_2d",   core_gaussian_blur_2d,   METH_VARARGS, "gaussian_blur_2d(data, height, width, sigma) -> bytes"},
  {"structure_tensor_3d",core_structure_tensor_3d, METH_VARARGS, "structure_tensor_3d(data, depth, height, width, deriv_sigma, smooth_sigma) -> bytes"},
  {"histogram",          core_histogram,           METH_VARARGS, "histogram(data, num_elements, num_bins) -> dict"},
  {"window_level",       core_window_level,        METH_VARARGS, "window_level(data, num_elements, window, level) -> bytes"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef core_module = {
  PyModuleDef_HEAD_INIT, "_core", NULL, -1, core_methods
};

PyMODINIT_FUNC PyInit__core(void) {
  return PyModule_Create(&core_module);
}
