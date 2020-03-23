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
/*! Applies LeakyRelu function \f$ f(x)=\left\{\begin{matrix}
\alpha x & x<0 \\
x  & x\geq 0
\end{matrix}\right.\f$ elementwise */
template <typename T> class LeakyRelu : public baseOperator<T, T, T> {
protected:
  float alpha = 0.01;

public:
  LeakyRelu(std::string name = "opLeakyRelu", float alpha = 0.01)
      : baseOperator<T, T, T>(opLeakyRelu, name) {
    this->alpha = alpha;
  }
  bool getAttribute(OPATTR attrName, float &obj) override {
    if (attrName == attr_alpha) {
      obj = alpha;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, float obj) override {
    if (attrName == attr_alpha) {
      alpha = obj;
      return true;
    }
    return false;
  }
  static T Leaky_Relu(T x, float alpha) {
    if (x < 0)
      return T(alpha * x);
    else
      return x;
  }

  tensor<T> compute(tensor<T> &a /*!<[float,double]: ND tensor*/) {
    if (!(this->template type_check<T, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }
    tensor<T> result(a.shape(), a.name());
    DNNC_EIGEN_ARRAY_MAP(eigenVector, T, a);
    DNNC_EIGEN_VECTOR_CTOR(T) eResult;
    auto c0 = std::bind(Leaky_Relu, std::placeholders::_1, alpha);
    eResult.array() = eigenVector.array().unaryExpr(c0);

    result.load(eResult.data());
    return result;
  }
};
} // namespace dnnc
