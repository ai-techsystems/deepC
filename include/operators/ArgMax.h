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
class ArgMax : public baseOperator<To, Ti, Ti> {

  int _axis = 0;
  int _keepdims = 1;

public:
  ArgMax(std::string name = "opArgMax")
      : baseOperator<To, Ti, Ti>(opArgMax, name) {}

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_axis) {
      obj = _axis;
      return true;
    } else if (attrName == attr_keepdims) {
      obj = _keepdims;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_axis) {
      _axis = obj;
      return true;
    } else if (attrName == attr_keepdims) {
      _keepdims = obj;
      return true;
    }
    return false;
  }

  tensor<To> compute(tensor<Ti> input) {

    if (!(this->template type_check<short int, int, long int>(typeid(To))))
      throw std::invalid_argument("Constrain output tensor type to int type.");

    if (_axis < -input.rank() || _axis > input.rank() - 1)
      throw std::invalid_argument("axis is out of bounds for tensor.");

    if (input.rank() == 1 || input.rank() == 2) {
    } else if (input.rank() == 3) {
    } else if (input.rank() == 4) {
    } else {
    }
    return NULL_TENSOR<To>;
  }
};
} // namespace dnnc
