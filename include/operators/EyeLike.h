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
template <typename T> class EyeLike : public baseOperator<T> {
protected:
  int k = 0;

public:
  EyeLike(std::string name = "opEyeLike", int k = 0)
      : baseOperator<T>(opEyeLike, name) {
    this->k = k;
  }

  bool getAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_k) {
      obj = k;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> &a) {
    if (a.rank() != 2)
      throw std::invalid_argument(
          "tensor dimenions not appropriate for EyeLike operator.");

    int row = a.shape()[0];
    int col = a.shape()[1];
    tensor<T> result(row, col);

    DNNC_EIGEN_MATRIX(eigenMatrixA, a);

    Matrix<T, Dynamic, Dynamic> eResult(row, col);

    // std::cout << std::endl << "row" << " " << "col" << std::endl ;

    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {

        if (i == (j - k)){
          std::cout << i << " " << (j) << " -> " << "1" << std::endl;
          eResult(i, j) = 1;
        }
        else{
          std::cout << i << " " << (j) << " -> " << "0" << std::endl;
          eResult(i, j) = 0;
        }
        // eResult(i, j) = (i == (j - k)) ? 1. : 0.;
      }
    }

    result.load(eResult.data());

    return result;
  }
};
} // namespace dnnc