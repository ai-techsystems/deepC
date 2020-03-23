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

#pragma once
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {
/*! The operator computes the hardmax (1 for the first maximum value, and 0 for
 * all others) values for each layer in the batch of the given input */
/*! Input does not need to explicitly be a 2D vector; rather, it will be coerced
 * into one: */
/*! A tensor of N-dimension \f$ [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] \f$
 * where k is a attribute ,will be coerced into 2-D \f$ [a_0 * ... * a_{k-1},
 * a_k * ... * a_{n-1}] \f$ .*/
template <typename T> class Hardmax : public baseOperator<T, T, T> {
protected:
  int axis =
      1; /*!< Describes the axis of the inputs when coerced to 2D; defaults to
            one because the 0th axis most likely describes the batch_size */
public:
  Hardmax(std::string name = "opHardmax", int axis = 0)
      : baseOperator<T, T, T>(opHardmax, name) {
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
  tensor<T> compute(tensor<T> a/*< The input tensor that will be coerced into a 2D matrix of size (NxD) as described in operator definition*/) {
    if (!(this->template type_check<T, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
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
    tensor<T> result(a.shape());

    Eigen::MatrixXf::Index max_index;

    DNNC_EIGEN_MATRIX(eigenMatrix1, T, a);
    for (int j = 0; j < int(a.shape()[1]); j++) {
      eigenMatrix1.col(j).maxCoeff(&max_index);
      for (int i = 0; i < int(a.shape()[0]); i++) {
        if (i == max_index) {
          result(i, j) = 1;
          continue;
        } else
          result(i, j) = 0;
      }
    }
    result.reshape(original_shape);
    return result;
  }
  /*!<
  \return The output values with the same shape as input tensor (the original
  size without coercion).
  */
};
} // namespace dnnc
