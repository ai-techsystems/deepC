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

/*! \f$
 * f(x)=\alpha\times(e^{x}-1),\;\;\;for\;x<0\;;\\f(x)=x,\;\;\;for\;x\geq0\;;
 * \f$*/
/*! The formula shows how the Elu operator works.*/
/*! And this formulation became part of dnn compiler operator implementation.
 * The operator is O(n) where n = Number of elements in the tensor*/

template <typename T> class Elu : public baseOperator<T, T, T> {
protected:
  float alpha = 1.0; /*!< Coefficient of ELU. */

public:
  Elu(std::string name = "opElu", float alpha = 1.0)
      : baseOperator<T, T, T>(opElu, name) {
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
  /*! Element wise Elu-Function*/
  static T elu_function(T x, float alpha) {
    return (x < 0) ? (alpha * (exp(x) - 1.)) : x;
  }

  tensor<T> compute(tensor<T> &a /*!<[float,double]: ND tensor*/) {

    if (!(this->template type_check<T, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }

    if (a.rank() != 1) {
      SPDLOG_ERROR("tensor dimenions not appropriate for Elu operator.");
      return NULL_TENSOR<T>;
    }

    tensor<T> result(a.shape(), a.name());

    DNNC_EIGEN_ARRAY_MAP(eigenVector, T, a);
    DNNC_EIGEN_VECTOR_CTOR(T) eResult;
    auto c0 = std::bind(elu_function, std::placeholders::_1, alpha);
    eResult.array() = eigenVector.array().unaryExpr(c0);
    result.load(eResult.data());

    return result;
  }
  /*!<
  \return The output tensor of the same shape as input.
  */
};
} // namespace dnnc
