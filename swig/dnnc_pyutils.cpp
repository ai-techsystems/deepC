// Copyright 2018 The AITS DNNC Authors.All Rights Reserved.
//
// Licensed to the Apache Software Foundation(ASF) under one
// or more contributor license agreements.See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.See the License for the
// specific language governing permissionsand limitations
// under the License.
//
// This file is part of AITS DNN compiler maintained at
// https://github.com/ai-techsystems/dnnCompiler
//
#include <Python.h>
#include <stdexcept>
#include <vector>

using namespace std;

PyObject *vectorToList_Float(const vector<float> &data) {
  PyObject *listObj = PyList_New(data.size());
  if (!listObj)
    throw logic_error("Unable to allocate memory for Python list");
  for (unsigned int i = 0; i < data.size(); i++) {
    PyObject *num = PyFloat_FromDouble((double)data[i]);
    if (!num) {
      Py_DECREF(listObj);
      throw std::bad_alloc();
    }
    PyList_SET_ITEM(listObj, i, num);
  }
  return listObj;
}

PyObject *vectorToTuple_Float(const vector<float> &data) {
  PyObject *tuple = PyTuple_New(data.size());
  if (!tuple)
    throw std::bad_alloc();
  for (unsigned int i = 0; i < data.size(); i++) {
    PyObject *num = PyFloat_FromDouble((double)data[i]);
    if (!num) {
      Py_DECREF(tuple);
      throw std::bad_alloc();
    }
    PyTuple_SET_ITEM(tuple, i, num);
  }

  return tuple;
}

PyObject *vectorVectorToTuple_Float(const vector<vector<float>> &data) {
  PyObject *tuple = PyTuple_New(data.size());
  if (!tuple)
    throw std::bad_alloc();
  for (unsigned int i = 0; i < data.size(); i++) {
    PyObject *subTuple = NULL;
    try {
      subTuple = vectorToTuple_Float(data[i]);
    } catch (logic_error &e) {
      throw e;
    }
    if (!subTuple) {
      Py_DECREF(tuple);
      throw std::bad_alloc();
    }
    PyTuple_SET_ITEM(tuple, i, subTuple);
  }

  return tuple;
}

vector<float> listTupleToVector_Float(PyObject *incoming) {
  vector<float> data;
  if (PyFloat_Check(incoming)) {
    size_t num = PyFloat_AsDouble(incoming);
    data.push_back(num);
  } else if (PyTuple_Check(incoming)) {
    for (Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
      PyObject *value = PyTuple_GetItem(incoming, i);
      data.push_back(PyFloat_AsDouble(value));
    }
  } else if (PyList_Check(incoming)) {
    for (Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
      PyObject *value = PyList_GetItem(incoming, i);
      data.push_back(PyFloat_AsDouble(value));
    }
  } else {
    throw logic_error("Passed PyObject pointer was not a list or tuple!");
  }
  return data;
}

vector<int> listTupleToVector_Int(PyObject *incoming) {
  vector<int> data;
  if (PyLong_Check(incoming)) {
    size_t num = PyLong_AsLong(incoming);
    data.push_back(num);
  } else if (PyTuple_Check(incoming)) {
    for (Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
      PyObject *value = PyTuple_GetItem(incoming, i);
      data.push_back(PyLong_AsLong(value));
    }
  } else if (PyList_Check(incoming)) {
    for (Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
      PyObject *value = PyList_GetItem(incoming, i);
      data.push_back(PyLong_AsLong(value));
    }
  } else {
    throw logic_error("Passed PyObject pointer was not a list or tuple!");
  }
  return data;
}

vector<size_t> listTupleToVector_SizeT(PyObject *incoming) {
  vector<size_t> data;
  if (PyLong_Check(incoming)) {
    size_t num = PyLong_AsSize_t(incoming);
    data.push_back(num);
  } else if (PyTuple_Check(incoming)) {
    for (Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
      PyObject *value = PyTuple_GetItem(incoming, i);
      data.push_back(PyLong_AsSize_t(value));
    }
  } else if (PyList_Check(incoming)) {
    for (Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
      PyObject *value = PyList_GetItem(incoming, i);
      data.push_back(PyLong_AsSize_t(value));
    }
  } else {
    throw logic_error("Passed PyObject pointer was not a list or tuple!");
  }
  return data;
}
