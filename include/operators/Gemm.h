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

/*! The General Matrix Multiplication formula is*/
/*! \f$ Y=\alpha\;A'\;B'+\beta\;C\f$ */
/*! Where */
/*! \f$ A'=transpose(A)\;,\;\;if\;A_{trans}=1\;;\\A'=A\;,\;\;if\;A_{trans}=0 \f$
 */
/*! And */
/*! \f$ B'=transpose(B)\;,\;\;if\;B_{trans}=1\;;\\B'=B\;,\;\;if\;B_{trans}=0 \f$
 */
/*! Input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N)
or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y
has shape (M, N). A will be transposed before doing the computation if attribute
transA is non-zero, same for B and transB.\n This operator supports
unidirectional broadcasting (tensor C should be
unidirectional broadcastable to tensor A * B)*/

template <typename T> class Gemm : public baseOperator<T> {
protected:
  float alpha =
      1.0; /*!< Scalar multiplier for the product of input tensors A * B */
  float beta = 1.0; /*!< Scalar multiplier for input tensor C */
  int transA = 0;   /*!< Whether A should be transposed */
  int transB = 0;   /*!< Whether B should be transposed */

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

  tensor<T> compute(tensor<T> &a/*!<Input tensor A. The shape of A should be (M, K) 
                                  if \f$ A_{trans} \f$ is 0, or (K, M) if \f$ A_{trans} \f$ 
                                  is non-zero.*/,
                   tensor<T> &b/*!<Input tensor B. The shape of B should be (K, N) 
                                  if \f$ B_{trans} \f$ is 0, or (N, K) if \f$ B_{trans} \f$ 
                                  is non-zero.*/,
                   tensor<T> &c/*!<Input tensor C. The shape of C should be
                                   unidirectional broadcastable to (M, N)*/) {

    if (a.rank() != 2 || b.rank() != 2 || c.rank() != 2)
      throw std::invalid_argument(
          "tensor dimenions not appropriate for Gemm operator.");

    if (!(this->template type_check<float, double, int>()))
      throw std::invalid_argument(
          "Constrain input and output types to float and int tensors.");

    tensor<T> result(c.shape(), c.name());
    DNNC_EIGEN_MATRIX(eigenMatrixA, a);
    DNNC_EIGEN_MATRIX(eigenMatrixB, b);
    DNNC_EIGEN_MATRIX(eigenMatrixC, c);
    Matrix<T, Dynamic, Dynamic, RowMajor> eResult(c.shape()[0], c.shape()[1]);
    // DNNC_EIGEN_VECTOR_CTOR(T) eResult;

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
  /*!<
  \return The output tensor of the same shape and type as tensor C.
  */
};
} // namespace dnnc