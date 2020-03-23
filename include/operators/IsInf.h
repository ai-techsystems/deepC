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
/*! Map infinity to true and other values to false.*/
template <typename To, typename Ti>
class IsInf : public baseOperator<To, Ti, Ti> {
protected:
  int detect_negative =
      1; /*!< (Optional) Whether map negative infinity to true. Default to 1 so
            that negative infinity induces true. Set this attribute to 0 if
            negative infinity should be mapped to false.*/
  int detect_positive =
      1; /*!< (Optional) Whether map positive infinity to true. Default to 1 so
            that positive infinity induces true. Set this attribute to 0 if
            positive infinity should be mapped to false */

public:
  IsInf(std::string name = "opIsInf", int detect_positive = 1,
        int detect_negative = 1)
      : baseOperator<To, Ti, Ti>(opIsInf, name) {
    this->detect_positive = detect_positive;
    this->detect_negative = detect_negative;
  }
  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_detect_positive) {
      obj = detect_positive;
      return true;
    } else if (attrName == attr_detect_negative) {
      obj = detect_negative;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_detect_positive) {
      detect_positive = obj;
      return true;
    } else if (attrName == attr_detect_negative) {
      detect_negative = obj;
      return true;
    }
    return false;
  }

  static bool Is_INF(Ti x, int detect_negative, int detect_positive) {
    if (std::isinf(x)) {
      if ((x < 0) && (detect_negative))
        return true;
      else if ((x > 0) && (detect_positive))
        return true;
      else
        return false;
    } else
      return false;
  }
  // NOT GOOD to return by value
  tensor<To> compute(tensor<Ti> &a) {
    if (!(this->template type_check<Ti, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<To>;
    }
    tensor<bool> result(a.shape(), a.name());
    auto c0 = std::bind(Is_INF, std::placeholders::_1, detect_negative,
                        detect_positive);

    for (size_t i = 0; i < a.length(); i++) {
      result[i] = c0(a[i]);
    }
    return result;
  }
};
} // namespace dnnc
