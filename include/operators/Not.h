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
template <typename T> class Not : public baseOperator<T> {
  //  Not attributes
protected:

  //Functions that defines the operation to be performed for each element
  static T not_function(T x) {
    return !x;
  }

public:
  Not(std::string name = "opNot") : baseOperator<T>(opNot, name) {
    if (!(this->template type_check<bool>()))
      throw std::invalid_argument(
          "Constrain input and output types to bool tensors.");
  }

  // bool getAttribute<int>(OPATTR attrName, int& obj) ;

  tensor<T> compute(tensor<T> a) {

    //Result vector declaration
	  tensor<T> result(a.shape(), a.name());

    DNNC_EIGEN_VECTOR(input_eigen, a);
    DNNC_EIGEN_VECTOR_CTOR(T) eigen_result;

    auto not_bind = std::bind(not_function, std::placeholders::_1);

    eigen_result.array() = input_eigen.array().unaryExpr(not_bind);
    result.load(eigen_result.data());

    return result;
  }
};
} // namespace dnnc