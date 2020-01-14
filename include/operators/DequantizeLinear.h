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
/*! \f$ y=(x-x_{0})*x_{scale} \f$*/
/*! Where \f$ x \f$ is a quantized tensor, \f$x_{0}\f$ is the origin,
 and \f$x_{scale}\f$ is the scale.*/
/*! The formula shows how the Dequantize Linear works.*/
/*! Constraints: \f$x_{scale}\f$ and \f$x_{0}\f$ must have same shape.
 \f$x_{0}\f$ and \f$ x \f$ must have same type (8-bit/32-bit integer tensor)*/

template <typename To, typename Ti>
class DequantizeLinear : public baseOperator<To, Ti, Ti> {
public:
  DequantizeLinear(std::string name = "opDequantizeLinear")
      : baseOperator<To, Ti, Ti>(opDequantizeLinear, name) {}

  /*
  static T dequantize_linear_function (T a, float a_scale, float a_zero_point) {
    return ((a - a_zero_point) * a_scale);
  }

  static bool compare(tensor<T> &a, tensor<T> &x_zero_point) {
    return ((typeid(a) == typeid(int)) && (typeid(x_zero_point) ==
  typeid(int)));
  }
  */

  tensor<To>
  compute(tensor<Ti> &a /*!<N-D quantized input tensor to be de-quantized*/,
          tensor<float> &x_scale /*!<Scalar tensor*/,
          tensor<Ti> &x_zero_point /*!<Scalar tensor*/) {
    if (x_scale.shape() != x_zero_point.shape())
      SPDLOG_ERROR(
          "tensor dimenions not appropriate for DequantizeLinear operator.");
    /*
    if (!compare(a,a_zero_point))
     SPDLOG_ERROR(
         "Constrain input and output types to float tensors.");
   */

    tensor<float> result(a.shape(), a.name());

    for (size_t i = 0; i < a.length(); i++)
      result[i] = (a[i] - x_zero_point[0]) * x_scale[0];

    /*
    float a_zero_point = x_zero_point[0];
    float a_scale = x_scale[0];
    a.flatteninplace();
    DNNC_EIGEN_VECTOR(eigenVector, a);
    DNNC_EIGEN_VECTOR_CTOR(T) eResult;
    auto c0 = std::bind(dequantize_linear_function, std::placeholders::_2,
    a_scale, a_zero_point); eResult.array() = eigenVector.array().unaryExpr(c0);
    result.load(eResult.data());
    */

    return result;
  }
  /*!<
  \return The output tensor as float and of the same shape as input.
  */
};
} // namespace dnnc
