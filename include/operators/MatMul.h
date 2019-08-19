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
template <typename T> class MatMul : public baseOperator<T> {
protected:
  //  MatMul attributes
  //  NONE
public:
  MatMul(std::string name = "opMatMul") : baseOperator<T>(opMatMul, name) {}
  tensor<T> compute(tensor<T> &a, tensor<T> &b) {

    if ((a.rank() == 1) || (a.rank() == 2)) {
      if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument(
            "tensor dimenions not appropriate for multiplication operator.");

      tensor<T> result(a.shape()[0], b.shape()[1]);

      DNNC_EIGEN_MATRIX(eigenMatrixA, a);
      DNNC_EIGEN_MATRIX(eigenMatrixB, b);

      Matrix<T, Dynamic, Dynamic> eResult = eigenMatrixA * eigenMatrixB;

      result.load(eResult.data());
      return result;
    } else if ((a.rank() == 3)) {
      // dnnc (& eigen) are column major whereas numpy is row major and requires:
      // a.shape()[0] == b.shape()[0] && a.shape()[2] == b.shape()[1]
      if ((a.shape()[1] != b.shape()[0]) || (a.shape()[2] != b.shape()[2])) {
        throw std::invalid_argument(
            "tensor dimenions not appropriate for multiplication operator.");
      }

      tensor<T> result(a.shape()[0], b.shape()[1], b.shape()[2]);

      DNNC_EIGEN_TENSOR_MAP(eigenTensorA, a);
      DNNC_EIGEN_TENSOR_MAP(eigenTensorB, b);

      Tensor<T,3> eResult(a.shape()[0], b.shape()[1], b.shape()[2]);

      for (size_t i = 0; i < a.shape()[2]; i++) {
	Tensor<T,2> eigenTensorChipA = eigenTensorA.chip(i,2);
	Tensor<T,2> eigenTensorChipB = eigenTensorB.chip(i,2);
	
	auto eigenMatrixA  = Map<Matrix<T, Dynamic, Dynamic>>(eigenTensorChipA.data(), 
								 a.shape()[0], 
								 a.shape()[1]);
	auto eigenMatrixB  = Map<Matrix<T, Dynamic, Dynamic>>(eigenTensorChipB.data(), 
								 b.shape()[0], 
								 b.shape()[1]);
        Matrix<T, Dynamic, Dynamic> eigenMatMulAB = eigenMatrixA * eigenMatrixB;

	eResult.chip(i, 2) = TensorMap<Tensor<T, 2>>(eigenMatMulAB.data(), 
							    a.shape()[0], 							    
							    b.shape()[1]);
      }

      result.load(eResult.data());
      return result;

    } else if ((a.rank() == 4)) {
#ifdef DNNC_HIGHRANK_SUPPORT
      if ((a.shape()[1] != b.shape()[0]) || (a.shape()[2] != b.shape()[1]) ||
          (a.shape()[3] != b.shape()[2])) {
        throw std::invalid_argument(
            "tensor dimenions not appropriate for multiplication operator.");
      }

      tensor<T> result(a.shape()[0], a.shape()[1], a.shape()[2], b.shape()[3]);

      DNNC_EIGEN_TENSOR4D_MAP(eigenTensorA, a);
      DNNC_EIGEN_TENSOR4D_MAP(eigenTensorB, b);

      array<IndexPair<long>, 3> dims = {
          IndexPair<long>(1, 0), IndexPair<long>(2, 1), IndexPair<long>(3, 2)};

      auto eResult = static_cast<DNNC_EIGEN_TENSOR4D>(
          eigenTensorA.contract(eigenTensorB, dims));

      result.load(eResult.data());
      return result;
#endif

    } else {
      throw std::invalid_argument("invalid tensor rank.");
    }

    return tensor<T>();
  }
};
} // namespace dnnc
