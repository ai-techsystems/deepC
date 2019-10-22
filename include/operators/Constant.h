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
template <typename To> class Constant : public baseOperator<To, To, To> {
  //  Constant attributes
  tensor<To> _data;

public:
  Constant(std::string name = "opConstant", tensor<To> data = NULL_TENSOR<To>)
      : baseOperator<To, To, To>(opConstant, name), _data(data) {}

  bool getAttribute(OPATTR attrName, tensor<To> &obj) override {
    if (attrName == attr_value) {
      obj = _data;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, tensor<To> obj) override {
    if (attrName == attr_value) {
      _data = obj;
      return true;
    }
    return false;
  }

  tensor<To> compute(void) { return _data; }
};
} // namespace dnnc
