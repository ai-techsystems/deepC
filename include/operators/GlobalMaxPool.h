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
template <typename T> class GlobalMaxPool : public baseOperator<T> {
public:
  GlobalMaxPool(std::string name = "opGlobalMaxPool")
      : baseOperator<T>(opGlobalMaxPool, name) {}
  static bool compare() {
    return ((typeid(T) == typeid(float)) || (typeid(T) == typeid(double)));
  }
  tensor<T> compute(tensor<T> a) {
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
    std::cout << a.rank() << "\n";
    for (int i = 2; i < int(rank); i++) {
      shape.push_back(1);
    }
    int cummulation = axis_left;
    tensor<T> result(shape);
    T max = a[0];
    int j = 0;
    for (size_t i = 1; i < a.length(); i++) {
      if (a[i] > max)
        max = a[i];
      if (!((i + 1) % cummulation)) {
        result[j++] = max;
        if ((i + 1) != a.length())
          max = a[i + 1];
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
