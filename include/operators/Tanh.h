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
template <typename T> class Tanh : public baseOperator<T, T, T> {
  //  Tanh attributes
public:
  Tanh(std::string name = "opTanh") : baseOperator<T, T, T>(opTanh, name) {}

  // bool getAttribute<int>(OPATTR attrName, int& obj) ;

  // float Tanh_func(T x) { return tanh(x); }

  // NOT GOOD to return by value
  tensor<T> compute(tensor<T> &a) {
    if (!(this->template type_check<T, float, double>())) {
      LOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }

    tensor<T> result(a.shape(), a.name());

    DNNC_EIGEN_ARRAY_MAP(eigenVector, T, a);
    DNNC_EIGEN_VECTOR_CTOR(T) eResult;

    eResult.array() = tanh(eigenVector.array());

    result.load(eResult.data());
    return result;
    /*if (a.rank() == 2) {
      tensor<T> result(a.shape()[0], a.shape()[1]);
      Matrix<T, Dynamic, Dynamic> eResult = eigenMatrixA.unaryExpr(&Tanh_func);
      result.load(eResult.data());
      return result;
    } else if (a.rank() == 3) {
      tensor<T> result(a.shape()[0], a.shape()[1], a.shape()[2]);
      Matrix<T, Dynamic, Dynamic> eResult = eigenMatrixA.unaryExpr(&Tanh_func);
      result.load(eResult.data());
      return result;
    }*/
  }
};
} // namespace dnnc
