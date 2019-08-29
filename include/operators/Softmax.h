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
template <typename T> class Softmax : public baseOperator<T> {
  //  Softmax attributes

protected:
  int axis = 1;

public:
  Softmax(std::string name = "opSoftmax") : baseOperator<T>(opSoftmax, name) {}

  bool getAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_axis) {
      obj = axis;
      return true;
    }
    return false;
  }
  void setAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_axis) {
      axis = obj;
    }
  }

  static bool compare() {
    return ((typeid(T) == typeid(float)) || (typeid(T) == typeid(double)));
  }

  tensor<T> compute(tensor<T> &a) {
    if (!compare())
      throw std::invalid_argument(
          "Constrain a and output types to float tensors.");

    // For 2D
    if (a.rank() == 2) {

      tensor<T> result(a.shape()[0], a.shape()[1]);

      DNNC_EIGEN_MATRIX(eigenMatrixA, a);

      if (axis == 1) {
        int i, j;
        for (i = 0; i < int(a.shape()[0]); i++) {
          float sum = 0;
          for (j = 0; j < int(a.shape()[1]); j++) {
            sum += exp(eigenMatrixA(i, j));
          }
          for (j = 0; j < int(a.shape()[1]); j++) {
            eigenMatrixA(i, j) = exp(eigenMatrixA(i, j)) / (sum);
          }
        }
      }
      if (axis == 0) {
        int i, j;
        for (i = 0; i < int(a.shape()[1]); i++) {
          float sum = 0;
          for (j = 0; j < int(a.shape()[0]); j++) {
            sum += exp(eigenMatrixA(j, i));
          }
          for (j = 0; j < int(a.shape()[0]); j++) {
            eigenMatrixA(j, i) = exp(eigenMatrixA(j, i)) / (sum);
          }
        }
      }

      Matrix<T, Dynamic, Dynamic> eResult = eigenMatrixA;
      result.load(eResult.data());

      return result;
    } else
      throw std::invalid_argument(
          "tensor dimensions not appropriate for softmax operator.");
  }
};
} // namespace dnnc
