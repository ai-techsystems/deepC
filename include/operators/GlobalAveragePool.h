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
/*! GlobalAveragePool consumes an input tensor X and applies average pooling
 * across the values in the same channel. This is equivalent to AveragePool with
 * kernel size equal to the spatial dimension of input tensor.*/
template <typename T> class GlobalAveragePool : public baseOperator<T> {
public:
  GlobalAveragePool(std::string name = "opGlobalAveragePool")
      : baseOperator<T>(opGlobalAveragePool, name) {}
  /*! Constrain input and output types to float tensors.
   */
  static bool compare() {
    return ((typeid(T) == typeid(float)) || (typeid(T) == typeid(double)));
  }
  tensor<T> compute(
      tensor<T> a /*!< [float,double]: ND tensor of shape ( NxCxD1xD2…Dk ).*/) {
    if (!compare())
      throw std::invalid_argument(
          "Constrain input and output types to float tensors.");
    size_t axis_left = 1;
    for (int i = 2; i < int(a.rank()); i++) {
      axis_left *= a.shape()[i];
    }
    size_t rank = a.rank();
    std::vector<size_t> shape{a.shape()[0], a.shape()[1], axis_left};
    a.reshape(shape);
    shape.pop_back();
    for (int i = 2; i < int(rank); i++) {
      shape.push_back(1);
    }
    int cummulation = axis_left;
    tensor<T> result(shape);
    int j = 0;
    T sum = 0;
    for (size_t i = 0; i < a.length(); i++) {
      sum += a[i];
      if (!((i + 1) % cummulation)) {
        result[j++] = T(sum / cummulation);
        sum = 0;
      }
    }
    return result;
  }
  /*!<
  \return Output data tensor from pooling across the input tensor. The output
  tensor has the same rank as the input. The first two dimensions of output
  shape are the same as the input (N x C), while the other dimensions are all 1.
  */
};
} // namespace dnnc
