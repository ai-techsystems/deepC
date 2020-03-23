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
#include "core/broadcast.h"
#include "operators/baseOperator.h"
#include <string>
#include <vector>

using namespace Eigen;

namespace dnnc {
/*! Returns the tensor resulted
 * from Element-wise mean of each of the input tensors (
 * with Numpy-style broadcasting support).
 */
template <typename T> class Mean : public baseOperator<T, T, T> {

  T meanEl(std::vector<T> &v) {
    T sum = 0;
    if (v.size() == 0) {
      SPDLOG_ERROR("Mean operator requires non-zero size vector.");
      return NULL_TENSOR<T>;
    }

    for (size_t i = 0; i < v.size(); i++)
      sum += v[i];
    return sum / v.size();
  }

public:
  Mean(std::string name = "opMean") : baseOperator<T, T, T>(opMean, name) {}

  tensor<T>
  compute(std::vector<tensor<T>> inputs /*!<[float,double]: ND tensors */) {
    if (!(this->template type_check<T, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }

    if (inputs.size() == 0) {
      SPDLOG_ERROR("Mean operator requires non-zero size input vector.");
      return NULL_TENSOR<T>;
    }

    try {
      std::vector<DIMENSION> resultShape = vecBroadcastReShape(inputs);
      tensor<T> result(resultShape);
      // compute element wise mean
      for (size_t i = 0; i < result.length(); i++) {
        std::vector<T> elVector;
        for (size_t j = 0; j < inputs.size(); j++)
          elVector.push_back(inputs[j][i]);

        result[i] = meanEl(elVector);
      }
      return result;

    } catch (const std::exception &e) {
      SPDLOG_ERROR(
          "operands could not be broadcast together with given shapes!!!");
      return NULL_TENSOR<T>;
    }
  }
};
} // namespace dnnc
