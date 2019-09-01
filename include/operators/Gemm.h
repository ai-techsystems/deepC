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
template <typename T> class Gemm : public baseOperator<T> {
protected:
  float alpha = 1.0;
  float beta = 1.0;
  int transA = 0;
  int transB = 0;

public:
  Gemm(std::string name = "opGemm", float alpha = 1.0, float beta = 1.0,
       int transA = 0, int transB = 0)
      : baseOperator<T>(opGemm, name) {
    this->alpha = alpha;
    this->beta = beta;
    this->transA = transA;
    this->transB = transB;
  }

  bool getAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_transA) {
      obj = transA;
      return true;
    } else if (attrName == attr_transB) {
      obj = transB;
      return true;
    }
    return false;
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

  tensor<T> compute(tensor<T> &a, tensor<T> &b, tensor<T> &c) {
    if (a.rank() != 2 || b.rank() != 2 || c.rank() != 2)
      throw std::invalid_argument(
          "tensor dimenions not appropriate for Gemm operator.");
    tensor<T> result(c.shape()[0], c.shape()[1]);
    DNNC_EIGEN_MATRIX(eigenMatrixA, a);
    DNNC_EIGEN_MATRIX(eigenMatrixB, b);
    DNNC_EIGEN_MATRIX(eigenMatrixC, c);
    Matrix<T, Dynamic, Dynamic> eResult(c.shape()[0], c.shape()[1]);

    // if (transA==1)
    // 	eigenMatrixA.transposeInPlace();
    // if (transB==1)
    // 	eigenMatrixB.transposeInPlace();

    try {
      // eResult = alpha*(eigenMatrixA * eigenMatrixB) + beta * eigenMatrixC;
      if (transA == 0 && transB == 0) {
        eResult = alpha * (eigenMatrixA * eigenMatrixB) + beta * eigenMatrixC;
      } else if (transA == 1 && transB == 0) {
        eResult = alpha * ((eigenMatrixA.transpose()) * eigenMatrixB) +
                  beta * eigenMatrixC;
      } else if (transA == 0 && transB == 1) {
        eResult = alpha * (eigenMatrixA * (eigenMatrixB.transpose())) +
                  beta * eigenMatrixC;
      } else if (transA == 1 && transB == 1) {
        eResult =
            alpha * ((eigenMatrixA.transpose()) * (eigenMatrixB.transpose())) +
            beta * eigenMatrixC;
      }
    } catch (...) {
      throw std::invalid_argument(
          "tensor dimenions not appropriate for Gemm operator.");
    }
    result.load(eResult.data());
    return result;
  }
};
} // namespace dnnc