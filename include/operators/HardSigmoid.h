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
/*! \f$ \max (0,\min(1,alpha*x+beta)) \f$
 */
template <typename T> class HardSigmoid : public baseOperator<T> {
protected:
  float alpha = 0.2;
  float beta = 0.5;

public:
  HardSigmoid(std::string name = "opHardSigmoid", float alpha = 0.2,
              float beta = 0.5)
      : baseOperator<T>(opHardSigmoid, name) {
    this->alpha = alpha;
    this->beta = beta;
  }
  static bool compare() {
    return ((typeid(T) == typeid(float)) || (typeid(T) == typeid(double)));
  }
  bool getAttribute(OPATTR attrName, float &obj) {
    if (attrName == attr_alpha) {
      obj = alpha;
      return true;
    } else if (attrName == attr_beta) {
      obj = beta;
      return true;
    }
    return false;
  }
  static T Hard_Sigmoid(T x, float alpha, float beta) {
    T temp = T(alpha * x + beta);
    temp = (1 < temp) ? 1 : temp;
    temp = (0 > temp) ? 0 : temp;
    return temp;
  }

  // NOT GOOD to return by value
  tensor<T> compute(tensor<T> &a) {
    if (!compare())
      throw std::invalid_argument(
          "Constrain input and output types to float tensors.");
    tensor<T> result(a.shape(), a.name());
    // max(0, min(1, alpha * x + beta))
    a.flatteninplace();
    DNNC_EIGEN_VECTOR(eigenVector, a);
    DNNC_EIGEN_VECTOR_CTOR(T) eResult;
    auto c0 = std::bind(Hard_Sigmoid, std::placeholders::_1, alpha, beta);
    eResult.array() = eigenVector.array().unaryExpr(c0);
    result.load(eResult.data());
    return result;
  }
};
} // namespace dnnc
