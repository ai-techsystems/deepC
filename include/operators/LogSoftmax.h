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
// Eigen cwise unsupported-tensors(written TODO in original doc)
// computing softmax as per axis(https://en.wikipedia.org/wiki/Softmax_function)

#pragma once
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {
/*! The operator computes the logsoftmax (log of softmax) values for each layer
 * in the batch of the given input. */
/*! Input does not need to explicitly be a 2D vector; rather, it will be coerced
 * into one: */
/*! A tensor of N-dimension \f$ [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] \f$
 * where k is a attribute ,will be coerced into 2-D \f$ [a_0 * ... * a_{k-1},
 * a_k * ... * a_{n-1}] \f$ .*/
template <typename To, typename Ti>
class LogSoftmax : public baseOperator<To, Ti, Ti> {
  //  LogSoftmax attributes
protected:
  // default
  int axis =
      1; /*!< Describes the axis of the inputs when coerced to 2D; defaults to
     one because the 0th axis most likely describes the batch_size */
public:
  LogSoftmax(std::string name = "opLogSoftmax", int axis = 1)
      : baseOperator<To, Ti, Ti>(opLogSoftmax, name) {
    this->axis = axis;
  }

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_axis) {
      obj = axis;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_axis) {
      axis = obj;
      return true;
    }
    return false;
  }

  tensor<To> compute(tensor<Ti> a/*< The input tensor that will be coerced into a 2D matrix of size (NxD) as described in operator definition*/) override {
    if (!(this->template type_check<Ti, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<To>;
    }
    if (axis >= int(a.rank()))
      std::invalid_argument("Reshaping failed");
    std::vector<size_t> original_shape = a.shape();
    size_t axis1 = 1;
    size_t axis2 = 1;
    for (int i = 0; i < axis; i++) {
      axis1 *= a.shape()[i];
    }
    if (axis1 > a.length())
      std::invalid_argument("Reshaping failed.Check Axis");
    axis2 = a.length() / axis1;
    std::vector<size_t> shape{axis1, axis2};
    a.reshape(shape);
    tensor<Ti> result(a.shape());

    Eigen::MatrixXf::Index max_index;

    DNNC_EIGEN_MATRIX(eigenMatrix1, Ti, a);
    for (int i = 0; i < int(a.shape()[0]); i++) {
      float sum = 0;
      float e_x = 0;
      float max_x = eigenMatrix1.row(i).maxCoeff(&max_index);

      for (int j = 0; j < int(a.shape()[1]); j++) {
        e_x = exp(eigenMatrix1(i, j) - max_x);
        sum += e_x;
      }
      for (int j = 0; j < int(a.shape()[1]); j++) {
        result(i, j) = eigenMatrix1(i, j) - max_x - log(sum);
      }
    }
    result.reshape(original_shape);

    return result.template asType<To>();
  }
};
} // namespace dnnc
