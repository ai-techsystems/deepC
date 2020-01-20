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
#include <typeinfo>

using namespace Eigen;

namespace dnnc {
template <typename T> class Clip : public baseOperator<T, T, T> {
  //  Clip attributes
public:
  Clip(std::string name = "opClip") : baseOperator<T, T, T>(opClip, name) {}

  static T clipper(T x, T min, T max) {
    if (x > max)
      return max;
    else if (x < min)
      return min;
    else
      return x;
  }

  tensor<T> compute(tensor<T> a, T &min, T &max) {
    if (!(this->template type_check<T, float, double>())) {
      LOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }

    tensor<T> result(a.shape(), a.name());
    DNNC_EIGEN_ARRAY_MAP(eigenVector, T, a);
    DNNC_EIGEN_VECTOR_CTOR(T) eResult;
    auto c0 = std::bind(clipper, std::placeholders::_1, min, max);
    eResult.array() = eigenVector.array().unaryExpr(c0);
    result.load(eResult.data());
    return result;
  }
};
} // namespace dnnc
