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
/*! \f$ y=scale*\frac{x-mean}{\sqrt{variance+epsilon}} \f$
    where mean and variance are computed per instance per channel (C).
    The formula for Mean is given by:
    \f$ \mu = \frac{1}{n}\sum_{i=1}^{n} x_{i} \f$ */

/*! This can be calculated in a single pass through all the elements.*/

/*! The formula for Variance is given by:
    \f$ var(X) = \frac{1}{n}\sum_{i=1}^{n} (x_{i}-\mu)^{2} \f$ */

/*! According to this Mean of the elements in channel is prerequisite for
   Variance calculation. A little bit of maths will tell you that mean is not
   required for Variance they can be calculated simultaneously.*/

/*! \f$ var(X) = \frac{1}{n}\sum_{i=1}^{n} (x_{i}-\mu)^{2} =
\frac{1}{n}\sum_{i=1}^{n} (x_{i}^{2}-2\mu x_{i}+\mu^{2})=
\frac{1}{n}\sum_{i=1}^{n}x_{i}^{2}-\frac{2\mu}{n}\sum_{i=1}^{n} x_{i}+
\mu\frac{n}{n} = \frac{1}{n}\sum_{i=1}^{n}x_{i}^{2} - \mu^2 \f$ */

/*! And this formulation became part of dnn compiler operator implementation.
 * The operator is O(n) where n = Number of elements in the tensor =
 * N*C*D1…*Dk.*/
template <typename T>
class InstanceNormalization : public baseOperator<T, T, T> {
protected:
  float epsilon = 1e-05; /*!< In case variance goes to zero and to avoid
                            division by zero. */

public:
  InstanceNormalization(std::string name = "opInstanceNormalization",
                        float epsilon = 1e-05)
      : baseOperator<T, T, T>(opInstanceNormalization) {
    this->epsilon = epsilon;
  }
  bool getAttribute(OPATTR attrName, float &obj) override {
    if (attrName == attr_epsilon) {
      obj = epsilon;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, float obj) override {
    if (attrName == attr_epsilon) {
      epsilon = obj;
      return true;
    }
    return false;
  }
  tensor<T>
  compute(tensor<T>
              input /*!< [float,double]: ND tensor of shape ( NxCxD1xD2…Dk ).*/,
          tensor<T> &scale /*!<  1D vector of dimension C.*/,
          tensor<T> &B /*!< : 1D vector of dimension C.*/) {
    if (!(this->template type_check<T, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }

    tensor<T> result(input.shape(), input.name());
    std::vector<size_t> original_shape = input.shape();

    if ((input.shape()[1] != scale.shape()[0]) ||
        (input.shape()[1] != B.shape()[0])) {
      SPDLOG_ERROR("Inappropriate tensor dimenions");
      return NULL_TENSOR<T>;
    }

    size_t size = 1;
    for (size_t i = 2; i < input.rank(); i++) {
      size *= input.shape()[i];
    }

    std::vector<size_t> shape{input.shape()[0], input.shape()[1], size};
    int channel_size = size * input.shape()[0];
    input.reshape(shape);
    result.reshape(shape);
    T sum = 0;
    T sq_sum = 0;
    T mean;
    T var;
    for (size_t i = 0; i < input.shape()[1]; i++) {
      // std::cout << "Current Channel=" << i << "\n";
      for (size_t j = 0; j < input.shape()[0]; j++) {
        for (size_t k = 0; k < size; k++) {
          // std::cout << input(j, i, k) << ',';
          sum += input(j, i, k);
          sq_sum += input(j, i, k) * input(j, i, k);
        }
      }
      // std::cout << "\n";
      mean = sum / channel_size;
      var = (sq_sum / channel_size - mean * mean);
      // std::cout << "Mean= " << mean << ',' << "Variance=" << var <<
      // std::endl;
      for (size_t j = 0; j < input.shape()[0]; j++) {
        for (size_t k = 0; k < size; k++) {

          result(j, i, k) =
              scale[i] * (input(j, i, k) - mean) / sqrt(var + epsilon) + B[i];
        }
      }
      sum = 0;
      sq_sum = 0;
    }
    result.reshape(original_shape);
    return result;
  }
  /*!<
  \return The output tensor of the same shape as input.
  */
};
} // namespace dnnc
