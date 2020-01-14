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
template <typename To, typename Ti>
class Relu : public baseOperator<To, Ti, Ti> {
  //  Relu attributes
  static Ti relu_func(Ti x) {
    Ti zero = 0;
    return x < zero ? zero : x;
  }

public:
  Relu(std::string name = "opRelu") : baseOperator<To, Ti, Ti>(opRelu, name) {}

  tensor<To> compute(tensor<Ti> &input) {
    if (!(this->template type_check<Ti, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<To>;
    }

    tensor<Ti> result(input.shape(), input.name());
    DNNC_EIGEN_ARRAY_MAP(eigenVector, Ti, input);
    DNNC_EIGEN_VECTOR_CTOR(Ti) eResult;
    eResult.array() = eigenVector.array().unaryExpr(&relu_func);
    result.load(eResult.data());

    return result.template asType<To>();
  }
};
} // namespace dnnc
