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
template <typename T> class MatMul : public baseOperator<T, T, T> {
protected:
  //  MatMul attributes
  //  NONE
public:
  MatMul(std::string name = "opMatMul")
      : baseOperator<T, T, T>(opMatMul, name) {}
  tensor<T> compute(tensor<T> &a, tensor<T> &b) {

    if ((a.rank() == 1 && b.rank() == 1)) {
      if (a.length() != b.length()) {
        SPDLOG_ERROR(
            "vector dimensions not appropriate for multiplication operator.");
        return NULL_TENSOR<T>;
      }

      tensor<T> result({1});
      result[0] = 0.0;
      for (size_t i = 0; i < a.length(); i++)
        result[0] += a[i] * b[i];

      return result;
    } else if (a.rank() == 2 && b.rank() == 2) {
      if (a.shape()[1] != b.shape()[0]) {
        SPDLOG_ERROR("matrix dimensions not appropriate for 2D "
                     "multiplication operator.");
        return NULL_TENSOR<T>;
      }

      tensor<T> result({a.shape()[0], b.shape()[1]});

      DNNC_EIGEN_MATRIX(eigenMatrixA, T, a);
      DNNC_EIGEN_MATRIX(eigenMatrixB, T, b);

      Matrix<T, Dynamic, Dynamic, RowMajor> eResult =
          eigenMatrixA * eigenMatrixB;

      result.load(eResult.data());
      return result;
    } else if ((a.rank() == 3)) {
      if ((a.shape()[2] != b.shape()[1]) || (a.shape()[0] != b.shape()[0])) {
        SPDLOG_ERROR("tensor dimensions not appropriate for 3D "
                     "multiplication operator.");
        return NULL_TENSOR<T>;
      }

      tensor<T> result({a.shape()[0], a.shape()[1], b.shape()[2]});

      DNNC_EIGEN_TENSOR_MAP(eigenTensorA, T, a);
      DNNC_EIGEN_TENSOR_MAP(eigenTensorB, T, b);

      Tensor<T, 3, RowMajor> eResult(a.shape()[0], a.shape()[1], b.shape()[2]);

      for (size_t i = 0; i < a.shape()[0]; i++) {
        Tensor<T, 2, RowMajor> eigenTensorChipA = eigenTensorA.chip(i, 0);
        Tensor<T, 2, RowMajor> eigenTensorChipB = eigenTensorB.chip(i, 0);

        auto eigenMatrixA = Map<Matrix<T, Dynamic, Dynamic, RowMajor>>(
            eigenTensorChipA.data(), a.shape()[1], a.shape()[2]);
        auto eigenMatrixB = Map<Matrix<T, Dynamic, Dynamic, RowMajor>>(
            eigenTensorChipB.data(), b.shape()[1], b.shape()[2]);
        Matrix<T, Dynamic, Dynamic, RowMajor> eigenMatMulAB =
            eigenMatrixA * eigenMatrixB;

        eResult.chip(i, 0) = TensorMap<Tensor<T, 2, RowMajor>>(
            eigenMatMulAB.data(), a.shape()[1], b.shape()[2]);
      }

      result.load(eResult.data());
      return result;
#ifdef DNNC_HIGHRANK_SUPPORT
    } else if ((a.rank() == 4)) {
      if ((a.shape()[1] != b.shape()[0]) || (a.shape()[2] != b.shape()[1]) ||
          (a.shape()[3] != b.shape()[2])) {
        SPDLOG_ERROR(
            "tensor dimenions not appropriate for multiplication operator.");
        return NULL_TENSOR<T>;
      }

      tensor<T> result(
          {a.shape()[0], a.shape()[1], a.shape()[2], b.shape()[3]});

      DNNC_EIGEN_TENSOR4D_MAP(eigenTensorA, T, a);
      DNNC_EIGEN_TENSOR4D_MAP(eigenTensorB, T, b);

      array<IndexPair<long>, 3> dims = {
          IndexPair<long>(1, 0), IndexPair<long>(2, 1), IndexPair<long>(3, 2)};

      auto eResult = static_cast<DNNC_EIGEN_TENSOR4D>(
          eigenTensorA.contract(eigenTensorB, dims));

      result.load(eResult.data());
      return result;
#endif

    } else {
      SPDLOG_ERROR("invalid tensor rank.");
      return NULL_TENSOR<T>;
    }

    return tensor<T>();
  }
};
} // namespace dnnc
