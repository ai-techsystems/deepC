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
#include <typeinfo>

using namespace Eigen;

namespace dnnc {
template <typename T> class Chip : public baseOperator<T, T, T> {
protected:
  int value;
  int axis;

public:
  Chip(int value, int axis, std::string name = "opChip")
      : baseOperator<T, T, T>(opChip, name) {
    this->value = value;
    this->axis = axis;
  }

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_axis) {
      obj = axis;
      return true;
    } else if (attrName == attr_value) {
      obj = value;
      return true;
    }
    return false;
  }

  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_axis) {
      obj = axis;
      return true;
    } else if (attrName == attr_value) {
      obj = value;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> a) {

    int rank = a.rank();
    // constexpr int rank = a.rank();
    std::vector<size_t> resultShape(rank - 1);

    for (int i = 0, j = 0; i < rank; i++) {
      if (i != axis) {
        resultShape[j] = a.shape()[i];
        j++;
      }
    }
    // for (auto i: resultShape)
    //   std::cout << i << ' ';

    tensor<T> result(resultShape);
    std::cout << a << std::endl << std::endl;
    DNNC_EIGEN_TENSOR_MAP(eigenTensorA, T, a);
    std::cout << eigenTensorA << std::endl;

    // Eigen::Tensor<T, (rank-1), RowMajor> eResult = eigenTensorA.chip(value,
    // axis);
    std::cout << eigenTensorA.chip(value, axis) << std::endl << std::endl;
    // result.load(eResult.data());
    // return result;
    return a;
  }
};
} // namespace dnnc
