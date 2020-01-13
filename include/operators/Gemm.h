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

template <typename To, typename Ti1, typename Ti2>
class Gemm : public baseOperator<To, Ti1, Ti2> {
protected:
  float alpha =
      1.0; /*!< Scalar multiplier for the product of input tensors A * B */
  float beta = 1.0; /*!< Scalar multiplier for input tensor C */
  int transA = 0;   /*!< Whether A should be transposed */
  int transB = 0;   /*!< Whether B should be transposed */

  inline std::vector<size_t> swap0And1(std::vector<size_t> v) {
    return (v.size() == 2) ? std::vector<size_t>({v[1], v[0]}) : v;
  }

public:
  Gemm(std::string name = "opGemm", float alpha = 1.0, float beta = 1.0,
       int transA = 0, int transB = 0)
      : baseOperator<To, Ti1, Ti2>(opGemm, name) {
    this->alpha = alpha;
    this->beta = beta;
    this->transA = transA;
    this->transB = transB;
  }

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_transA) {
      obj = transA;
      return true;
    } else if (attrName == attr_transB) {
      obj = transB;
      return true;
    }
    return false;
  }
  bool getAttribute(OPATTR attrName, float &obj) override {
    if (attrName == attr_alpha) {
      obj = alpha;
      return true;
    } else if (attrName == attr_beta) {
      obj = beta;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_transA) {
      transA = obj;
      return true;
    } else if (attrName == attr_transB) {
      transB = obj;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, float obj) override {
    if (attrName == attr_alpha) {
      alpha = obj;
      return true;
    } else if (attrName == attr_beta) {
      beta = obj;
      return true;
    }
    return false;
  }

  tensor<To> compute(tensor<Ti1> a/*!<Input tensor A. The shape of A should be (M, K)
                                  if \f$ A_{trans} \f$ is 0, or (K, M) if \f$ A_{trans} \f$
                                  is non-zero.*/,
                   tensor<Ti1> b/*!<Input tensor B. The shape of B should be (K, N)
                                  if \f$ B_{trans} \f$ is 0, or (N, K) if \f$ B_{trans} \f$
                                  is non-zero.*/,
                   tensor<Ti1> c/*!<Input tensor C. The shape of C should be
                                   unidirectional broadcastable to (M, N)*/) override {

    std::vector<size_t> aShape = transA ? swap0And1(a.shape()) : a.shape();
    std::vector<size_t> bShape = transB ? swap0And1(b.shape()) : b.shape();

    if (a.rank() == 1 && b.rank() == 2) {
      if (aShape[0] == bShape[0])
        aShape = {1, aShape[0]};
      else if (bShape[0] == 1)
        aShape = {aShape[0], 1};
      aShape = transA ? swap0And1(aShape) : aShape;
      a.reshape(aShape);
    }
    if (a.rank() == 2 && b.rank() == 1) {
      if (aShape[1] == bShape[0])
        bShape = {bShape[0], 1};
      else if (aShape[1] == 1)
        bShape = {1, bShape[0]};
      bShape = transB ? swap0And1(bShape) : bShape;
      b.reshape(bShape);
    }

    if (a.rank() != 2 || b.rank() != 2) {
      SPDLOG_ERROR("tensor dimenions not appropriate for Gemm operator.");
      return NULL_TENSOR<To>;
    }

    if (!(this->template type_check<Ti1, float, double, int>())) {
      SPDLOG_ERROR(
          "Constrain input and output types to float and int tensors.");
      return NULL_TENSOR<To>;
    }

    std::vector<size_t> targetShape = {aShape[0], bShape[1]};
    tensor<Ti1> broadcastedC = broadcast(c, targetShape);

    tensor<Ti1> result(broadcastedC.shape(), broadcastedC.name());
    DNNC_EIGEN_MATRIX(eigenMatrixA, Ti1, a);
    DNNC_EIGEN_MATRIX(eigenMatrixB, Ti1, b);
    DNNC_EIGEN_MATRIX(eigenMatrixC, Ti1, broadcastedC);
    Matrix<Ti1, Dynamic, Dynamic, RowMajor> eResult(broadcastedC.shape()[0],
                                                    broadcastedC.shape()[1]);

    try {
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
      SPDLOG_ERROR("tensor dimenions not appropriate for Gemm operator.");
      return NULL_TENSOR<To>;
    }
    result.load(eResult.data());

    // perform type conversion
    if (!(this->template type_check<To, Ti1>()))
      return result.template asType<To>();

    return result;
  }
  /*!<
  \return The output tensor of the same shape and type as tensor C.
  */
};
} // namespace dnnc
