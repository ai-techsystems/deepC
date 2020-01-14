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

/*! Generate a 2D tensor (matrix) with ones on the diagonal and zeros
everywhere else. Only 2D tensors are supported, i.e. input T1 must
be of rank 2. The shape of the output tensor is the same as the input
tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By
default, the main diagonal is populated with ones, but attribute
'k' can be used to populate upper or lower diagonals*/
/*! The formula for fillung up Eye Like matrix is*/
/*! \f$ A_{(i,j)}=1\;,\;\;\;for\;\;i=j-k\;;\\A_{(i,j)}=0\;,\;\;\;for\;\;i\neq
 * j-k \f$*/

template <typename T> class EyeLike : public baseOperator<T, T, T> {
protected:
  int k = 0; /*!< (Optional) Index of the diagonal to be populated
               with ones. Default is 0. If T2 is the output, this
               op sets T2[i, i+k] = 1. k = 0 populates the main
               diagonal, k > 0 populates an upper diagonal, and
                k < 0 populates a lower diagonal.*/

public:
  EyeLike(std::string name = "opEyeLike", int k = 0)
      : baseOperator<T, T, T>(opEyeLike, name) {
    this->k = k;
  }

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_k) {
      obj = k;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_k) {
      k = obj;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> &a /*!< D input tensor to copy shape, and
                                 optionally, type information from*/) {
    if (a.rank() != 2) {
      SPDLOG_ERROR("tensor dimenions not appropriate for EyeLike operator.");
      return NULL_TENSOR<T>;
    }

    int row = a.shape()[0];
    int col = a.shape()[1];
    tensor<T> result(a.shape(), a.name());

    // DNNC_EIGEN_VECTOR_CTOR(T) eResult(row,col);
    Matrix<T, Dynamic, Dynamic, RowMajor> eResult(row, col);

    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        eResult(i, j) = (i == (j - k)) ? 1. : 0.;
      }
    }

    result.load(eResult.data());

    return result;
  }
  /*!<
  \return The output tensor of the same shape as input.
  */
};
} // namespace dnnc
