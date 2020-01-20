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

using namespace Eigen;

namespace dnnc {
/*! Returns the tensor resulted from performing the greater than or equal to
 * logical operation elementwise on the input tensors A and B (with Numpy-style
 * broadcasting support).
 */
template <typename To, typename Ti>
class GreaterEqual : public baseOperator<To, Ti, Ti> {
public:
  GreaterEqual(std::string name = "opGreaterEqual")
      : baseOperator<To, Ti, Ti>(opGreaterEqual, name) {}

  tensor<bool>
  compute(tensor<Ti> &a /*!< First input operand for the logical operator.*/,
          tensor<Ti> &b /*!< Second input operand for the logical operator.*/) {
    std::vector<DIMENSION> resultShape = binaryBroadcastReShape(a, b);
    tensor<bool> result(resultShape);

    if (a.shape() != b.shape()) {
      LOG_ERROR(
          "tensor dimenions not appropriate for GreaterEqual operator.");
      return NULL_TENSOR<To>;
    }
    DNNC_EIGEN_ARRAY_MAP(eigenVectorA, Ti, a);
    DNNC_EIGEN_ARRAY_MAP(eigenVectorB, Ti, b);
    DNNC_EIGEN_VECTOR_CTOR(bool) eResult;
    eResult.array() = eigenVectorA.array() >= eigenVectorB.array();
    result.load(eResult.data());
    return result;
  }
};
} // namespace dnnc
