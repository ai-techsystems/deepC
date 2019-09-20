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
template <typename T> class Transpose : public baseOperator<T, T, T> {
  //  Transpose attributes
public:
  Transpose(std::string name = "opTranspose")
      : baseOperator<T, T, T>(opTranspose, name) {}

  // bool getAttribute<int>(OPATTR attrName, int& obj) ;

  tensor<T> compute(tensor<T> &a) {

    tensor<T> result(a.shape(), a.name());
    DNNC_EIGEN_MATRIX(eigenMatrix, T, a);
    // DNNC_EIGEN_VECTOR_CTOR(T) eResult;
    Matrix<T, Dynamic, Dynamic, RowMajor> eResult(a.shape()[0], a.shape()[1]);
    eResult = eigenMatrix.transpose();

    result.load(eResult.data());
    return result;

    /*if (a.rank() == 2) {
      tensor<T> result(a.shape()[1], a.shape()[0]);
      Matrix<T, Dynamic, Dynamic> eResult = eigenMatrixA.transpose();
      result.load(eResult.data());
      return result;
    } else if (a.rank() == 3) {
      tensor<T> result(a.shape()[2], a.shape()[0], a.shape()[1]);
      Matrix<T, Dynamic, Dynamic> eResult = eigenMatrixA.transpose();
      result.load(eResult.data());
      return result;
    }

    else if (a.rank() == 4) {
      tensor<T> result(a.shape()[3], a.shape()[2], a.shape()[1], a.shape()[0]);
      Matrix<T, Dynamic, Dynamic> eResult = eigenMatrixA.transpose();
      result.load(eResult.data());
      return result;
    }
    return a;*/
  }
};
} // namespace dnnc
