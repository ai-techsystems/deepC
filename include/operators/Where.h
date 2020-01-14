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
template <typename T> class Where : public baseOperator<T, T, T> {
  //  Where attributes
public:
  Where(std::string name = "opWhere") : baseOperator<T, T, T>(opWhere, name) {}

  // bool getAttribute<int>(OPATTR attrName, int& obj) ;

  tensor<T> compute(tensor<bool> &B, tensor<T> &X, tensor<T> &Y) {

    if (X.shape() != Y.shape() || X.shape() != B.shape() ||
        Y.shape() != B.shape()) {
      SPDLOG_ERROR("tensor dimenions not appropriate for Where operator.");
      return NULL_TENSOR<T>;
    }

    tensor<T> result(X.shape(), X.name());
    for (size_t i = 0; i < X.length(); i++)
      result[i] = B[i] ? X[i] : Y[i];

    return result;
  }
};
} // namespace dnnc
