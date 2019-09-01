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
#include "core/broadcast.h"
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {
template <typename T> class Add : public baseOperator<T> {
protected:
public:
  Add(std::string name = "opAdd") : baseOperator<T>(opAdd, name) {}
  tensor<T> compute(tensor<T> a, tensor<T> b) {

    std::vector<DIMENSION> resultShape = binaryBroadcastReShape(a, b);
    tensor<T> result(resultShape);

    if (a.rank() == 1) {

      DNNC_EIGEN_VECTOR(eigenMatrixA, a);
      DNNC_EIGEN_VECTOR(eigenMatrixB, b);

      Matrix<T, 1, Dynamic, RowMajor> eResult = eigenMatrixA + eigenMatrixB;

      result.load(eResult.data());
      return result;

    } else if (a.rank() == 2) {

      DNNC_EIGEN_MATRIX(eigenMatrixA, a);
      DNNC_EIGEN_MATRIX(eigenMatrixB, b);

      Matrix<T, Dynamic, Dynamic, RowMajor> eResult =
          eigenMatrixA + eigenMatrixB;

      result.load(eResult.data());
      return result;

    } else if (a.rank() == 3) {

      DNNC_EIGEN_TENSOR_MAP(eigenTensorA, a);
      DNNC_EIGEN_TENSOR_MAP(eigenTensorB, b);

      Tensor<T, 3, RowMajor> eResult = eigenTensorA + eigenTensorB;

      result.load(eResult.data());
      return result;

    } else {
      std::cout << "Not yet supported!" << std::endl;
      return dnnc::NULL_TENSOR<T>;
    }
    return result;
  }
};
} // namespace dnnc
