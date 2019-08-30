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
#include "operators/DequantizeLinear.h"
#include "operators/Div.h"
#include "operators/Elu.h"
#include "operators/Equal.h"
#include "operators/Erf.h"
#include "operators/Exp.h"
#include "operators/EyeLike.h"
#include "operators/Flatten.h"
#include "operators/Floor.h"
#include "operators/Gemm.h"
#include "operators/GlobalAveragePool.h"
#include "operators/GlobalLpPool.h"
#include "operators/GlobalMaxPool.h"
#include "operators/Greater.h"
#include "operators/HardSigmoid.h"
#include "operators/Hardmax.h"
#include "operators/Identity.h"
#include "operators/InstanceNormalization.h"
#include "operators/IsInf.h"
#include "operators/IsNaN.h"
#include "operators/LRN.h"
#include "operators/LeakyRelu.h"
#include "operators/Less.h"
#include "operators/Log.h"
#include "operators/LogSoftmax.h"
#include "operators/LpNormalization.h"
#include "operators/MatMul.h"
#include "operators/MatMulInteger.h"
#include "operators/Mean.h"
#include "operators/Min.h"
#include "operators/ThresholdedRelu.h"
#include "operators/Transpose.h"

extern std::vector<float> listTupleToVector_Float(PyObject *);
extern std::vector<size_t> listTupleToVector_SizeT(PyObject *);

using namespace dnnc;

tensor<float> array(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
  return tensor<float>(x, y, z, w, "", dnnc::INIT_RANDOM);
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

tensor<float> reshape(tensor<float> &tensorObj, PyObject *newShape) {
  if (PyLong_Check(newShape)) {
    std::vector<size_t> nShape;
    auto newShapeMember = PyLong_AsLong(newShape);
    nShape.push_back(static_cast<size_t>(newShapeMember));
    if (nShape[0] <= 0)
      throw std::logic_error("reshape integer must have positive value.\n");

    tensor<float> result = tensorObj.reshape(nShape);
    return result;
  } else if (PyTuple_Check(newShape)) {
    auto vShape = listTupleToVector_SizeT(newShape);
    for (size_t i = 0; i < vShape.size(); i++)
      if (vShape[i] <= 0)
        throw std::logic_error("reshape tupel must have positive elements.\n");

    tensor<float> result = tensorObj.reshape(vShape);
    return result;
  }
  return tensorObj;
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
    // Py_ssize_t sz = PyList_Size(a_list);
    std::vector<size_t> vDims;
    std::vector<float> vContents;
    for (Py_ssize_t i = 0; i < sz; i++) {
      std::vector<float> aVec =
          listTupleToVector_Float(PyList_GetItem(objects, i));
      vDims.push_back(aVec.size());
      vContents.insert(vContents.end(), aVec.begin(), aVec.end());
    }
    // make sure VDims are consistent.
    for (size_t i = 1; i < vDims.size(); i++) {
      if (vDims[0] != vDims[i])
        throw std::logic_error(
            "2D array expects list of lists with same size.");
    }
    // fill it as row major data.
    tensor<float> result(vDims.size(), vDims[0]);
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

tensor<float> matmul(tensor<float> &a, tensor<float> &b) {
  MatMul<float> op;
  return op.compute(a, b);
}

tensor<float> add(tensor<float> &a, tensor<float> &b) {
  Add<float> op;
  return op.compute(a, b);
}

tensor<float> dequantize_linear(tensor<float> &a, tensor<float> &b,
                                tensor<float> &c) {
  DequantizeLinear<float> op;
  return op.compute(a, b, c);
}

tensor<float> div(tensor<float> &a, tensor<float> &b) {
  Div<float> op;
  return op.compute(a, b);
}

tensor<float> elu(tensor<float> &a, float alpha = 1.0) {
  Elu<float> op("localOpName", alpha);
  return op.compute(a);
}

tensor<bool> equal(tensor<int> &a, tensor<int> &b) {
  Equal<int> op;
  return op.compute(a, b);
}

tensor<bool> equal(tensor<float> &a, tensor<float> &b) {
  Equal<float> op;
  return op.compute(a, b);
}

tensor<float> erf(tensor<float> &a) {
  Erf<float> op;
  return op.compute(a);
}

tensor<float> exp(tensor<float> &a) {
  Exp<float> op;
  return op.compute(a);
}

tensor<float> eye_like(tensor<float> &a, int k = 0) {
  EyeLike<float> op("localOpName", k);
  return op.compute(a);
}

tensor<float> flatten(tensor<float> &a, int axis = 1) {
  Flatten<float> op("localOpName", axis);
  return op.compute(a);
}

tensor<float> floor(tensor<float> &a) {
  Floor<float> op;
  return op.compute(a);
}

tensor<float> gemm(tensor<float> &a, tensor<float> &b, tensor<float> &c,
                   float alpha = 1.0, float beta = 1.0, int transA = 0,
                   int transB = 0) {
  Gemm<float> op("localOpName", alpha, beta, transA, transB);
  return op.compute(a, b, c);
}

tensor<float> thresholded_relu(tensor<float> &input) {
  ThresholdedRelu<float> op;
  return op.compute(input);
}

tensor<bool> less(tensor<float> &a, tensor<float> &b) {
  Less<float> op;
  return op.compute(a, b);
}

tensor<float> log(tensor<float> &input) {
  Log<float> op;
  return op.compute(input);
}

tensor<float> logsoftmax(tensor<float> &input) {
  LogSoftmax<float> op;
  return op.compute(input);
}

tensor<float> lpnormalization(tensor<float> &input) {
  LpNormalization<float> op;
  return op.compute(input);
}

tensor<int> matmulinteger(tensor<int> &a, tensor<int> &b) {
  MatMulInteger<int> op;
  return op.compute(a, b);
}

tensor<float> transpose(tensor<float> &a) {
  dnnc::Transpose<float> op;
  return op.compute(a);
}
tensor<float> global_average_pool(tensor<float> &input) {
  GlobalAveragePool<float> op;
  return op.compute(input);
}

tensor<float> global_lp_pool(tensor<float> &a, int p = 2) {
  GlobalLpPool<float> op("localOpName", p);
  return op.compute(a);
}

tensor<float> global_max_pool(tensor<float> &a) {
  GlobalMaxPool<float> op;
  return op.compute(a);
}

tensor<bool> greater(tensor<float> &a, tensor<float> &b) {
  Greater<float> op;
  return op.compute(a, b);
}

tensor<float> hardmax(tensor<float> &a, int axis = 0) {
  Hardmax<float> op("localOpName", axis);
  return op.compute(a);
}

tensor<float> hardsigmoid(tensor<float> &a, float alpha = 0.2,
                          float beta = 0.5) {
  HardSigmoid<float> op("localOpName", alpha, beta);
  return op.compute(a);
}

tensor<float> identity(tensor<float> &a) {
  Identity<float> op;
  return op.compute(a);
}

tensor<bool> isinf(tensor<float> &a, int detect_positive = 1,
                   int detect_negative = 1) {
  IsInf<float> op("localOpName", detect_positive, detect_negative);
  return op.compute(a);
}

tensor<bool> isnan(tensor<float> &a) {
  IsNaN<float> op;
  return op.compute(a);
}

tensor<float> leakyrelu(tensor<float> &a, float alpha = 0.01) {
  LeakyRelu<float> op("localOpName", alpha);
  return op.compute(a);
}

tensor<float> instancenormalization(tensor<float> &input, tensor<float> &scale,
                                    tensor<float> &B, float epsilon = 1e-5) {
  InstanceNormalization<float> op("localOpName", epsilon);
  return op.compute(input, scale, B);
}
tensor<float> lrn(tensor<float> &input, int size, float alpha = 0.0001,
                  float beta = 0.75, float bias = 1.0) {
  LRN<float> op(size, "localOpName", alpha, beta, bias);
  return op.compute(input);
}

/*tensor<float> mean(std::vector<tensor<float>> inputs) {
  Mean<float> op;
  return op.compute(inputs);
}

/tensor<float> min(std::vector<tensor<float>> inputs) {
  Min<float> op;
/ return op.compute(inputs);
}*/
