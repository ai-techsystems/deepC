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

#include "core/tensor.h"
#include "operators/Add.h"
#include "operators/MatMul.h"
#include "operators/ThresholdedRelu.h"

extern std::vector<float> listTupleToVector_Float(PyObject *);

using namespace dnnc;

tensor<float> make_tensor(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
  return tensor<float>(x, y, z, w);
}

tensor<float> empty(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
  return tensor<float>(x, y, z, w, "", dnnc::INIT_NONE);
}

tensor<float> zeros(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
  return tensor<float>(x, y, z, w, "", dnnc::INIT_ZERO);
}

tensor<float> ones(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
  return tensor<float>(x, y, z, w, "", dnnc::INIT_ONE);
}

tensor<float> random(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
  return tensor<float>(x, y, z, w, "", dnnc::INIT_RANDOM);
}

tensor<float> array(PyObject *objects) {
  if (!PyList_Check(objects))
    throw std::logic_error("array expects list to create array.\n");

  Py_ssize_t sz = PyList_Size(objects);
  if (!sz)
    throw std::logic_error("array expects list of non-zero size.\n");

  PyObject *a_list = PyList_GetItem(objects, 0);
  if (PyList_Check(a_list)) {
    // 2D: dc.array(([1,2],[10,20]]);
    Py_ssize_t sz = PyList_Size(a_list);
    std::vector<size_t> vDims;
    std::vector<float> vContents;
    for (Py_ssize_t i = 0; i < sz; i++) {
      std::vector<float> aVec =
          listTupleToVector_Float(PyList_GetItem(objects, i));
      vDims.push_back(aVec.size());
      vContents.insert(vContents.end(), aVec.begin(), aVec.end());
    }
    tensor<float> result(vDims);
    result.load(vContents);
    return result;
  } else {
    // 1D: dc.array([2,3,4])
    std::vector<float> aVec = listTupleToVector_Float(objects);
    tensor<float> result(aVec.size());
    result.load(aVec);
    return result;
  }
  return tensor<float>();
}

tensor<float> arange(size_t stop, size_t start = 0, size_t step = 1) {
  // swap if the range is invalid.
  if (stop < start) {
    size_t tmp = start;
    start = stop;
    stop = tmp;
  }
  if (stop == start)
    throw std::logic_error("arange expects stop arg more than start.\n");

  std::vector<float> vContents;
  for (size_t i = start; i < stop; i = i + step)
    vContents.push_back(static_cast<float>(i));

  tensor<float> result(vContents.size());
  result.load(vContents);
  return result;
}

tensor<float> multiply(tensor<float> &a, tensor<float> &b) {
  MatMul<float> op;
  return op.compute(a, b);
}

tensor<float> add(tensor<float> &a, tensor<float> &b) {
  Add<float> op;
  return op.compute(a, b);
}

tensor<float> thresholded_relu(tensor<float> &input) {
  ThresholdedRelu<float> op;
  return op.compute(input);
}
