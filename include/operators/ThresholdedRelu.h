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
template <typename T> class ThresholdedRelu : public baseOperator<T, T, T> {
  //  ThresholdedRelu attributes
  float alpha = 1.0; // alpha's default value.
public:
  ThresholdedRelu(std::string name = "opThresholdedRelu")
      : baseOperator<T, T, T>(opThresholdedRelu, name) {}

  bool getAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_alpha) {
      obj = alpha;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) {
    if (attrName == attr_alpha) {
      alpha = obj;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> &input) {
    // create a new copy of input.
    tensor<T> result(input.shape(), input.name());
    for (size_t i = 0; i < input.length(); i++)
      result[i] = input[i] > alpha ? input[i] : 0.0;

    return result;
  }
};
} // namespace dnnc
