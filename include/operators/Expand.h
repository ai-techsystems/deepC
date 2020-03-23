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
class Expand : public baseOperator<To, To, Ti> {

public:
  Expand(std::string name = "opExpand")
      : baseOperator<To, To, Ti>(opExpand, name) {}

  tensor<To> compute(tensor<To> a, tensor<Ti> b) {

    if (!(this->template type_check<Ti, int>())) {
      SPDLOG_ERROR("Constrain shape tensor to integer type.");
      return NULL_TENSOR<To>;
    }

    tensor<To> result(b.asTypeULong().data(), b.name());
    tensor<To> temp_b(b.asTypeULong().data(), b.name(), dnnc::INIT_ONE);

    // std::cout << a.shape()[0] << a.shape()[1] << std::endl;
    // std::cout << temp_b.shape()[0] << temp_b.shape()[1] << std::endl;

    DNNC_EIGEN_ARRAY_MAP(eigenVectorA, To, a);
    DNNC_EIGEN_ARRAY_MAP(eigenVectorB, To, temp_b);

    DNNC_EIGEN_VECTOR_CTOR(To) eResult;
    eResult = eigenVectorA * eigenVectorB;
    result.load(eResult.data());

    return result;
  }
};
} // namespace dnnc
