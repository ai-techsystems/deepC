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
template <typename T> class PRelu : public baseOperator<T, T, T> {
  //  PRelu attributes
public:
  PRelu(std::string name = "opPRelu") : baseOperator<T, T, T>(opPRelu, name) {}

  // bool getAttribute<int>(OPATTR attrName, int& obj) ;

  static T prelu(T x, T slope) {
    // f(x) = slope * x for x < 0, f(x) = x for x >= 0
    if (x < 0)
      return (T)slope * x;
    else
      return (T)x;
  }

  tensor<T> compute(tensor<T> &x /*!< : N D tensor input*/,
                    tensor<T> &slope /*!< : N D tensor input*/) {

    if (!(this->template type_check<T, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }

    std::vector<DIMENSION> resultShape = binaryBroadcastReShape(x, slope);
    tensor<T> result(resultShape);

    if (x.shape() != slope.shape()) {
      SPDLOG_ERROR("tensor dimenions not appropriate for PRelu operator.");
      return NULL_TENSOR<T>;
    }

    DNNC_EIGEN_ARRAY_MAP(eigen_x, T, x);
    DNNC_EIGEN_ARRAY_MAP(eigen_slope, T, slope);

    DNNC_EIGEN_VECTOR_CTOR(T) eigen_result;

    eigen_result.array() =
        eigen_x.array().binaryExpr(eigen_slope.array(), &prelu);

    result.load(eigen_result.data());

    return result;
  }
};
} // namespace dnnc
