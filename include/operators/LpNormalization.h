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
template <typename T> class LpNormalization : public baseOperator<T> {
  //  LpNormalization attributes

  // since axis is int it can be 0 or 1
  // only L1 norm and L2 norm is supported to
  // normalize(https://en.wikipedia.org/wiki/Norm_(mathematics)) Eigen cwise
  // unsupported-tensors(written TODO in original doc)
protected:
  int p = 2;
  int axis = 1;

public:
  LpNormalization(std::string name = "opLpNormalization")
      : baseOperator<T>(opLpNormalization, name) {}

  // bool getAttribute<int>(OPATTR attrName, int& obj) ;
  bool getAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_p) {
      obj = p;
      return true;
    } else if (attrName == attr_axis) {
      obj = axis;
      return true;
    }
    return false;
  }

  void setAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_axis) {
      axis = obj;
    } else if (attrName == attr_p)
      p = obj;
  }

  static bool compare() {
    return ((typeid(T) == typeid(float)) || (typeid(T) == typeid(double)));
  }

  tensor<T> compute(tensor<T> &input) {
    if (!compare())
      throw std::invalid_argument(
          "Constrain input and output types to float tensors.");
    if (p != 2 && p != 1) {
      throw std::invalid_argument("Constrain input not supported.");
      return input;
    }

    tensor<T> result(input.shape(), input.name());

    DNNC_EIGEN_MATRIX(eigenMatrixA, input);

    if (axis == 0 && p == 1) {
      int i, j;
      for (i = 0; i < int(input.shape()[1]); i++) {
        float sum = 0;
        for (j = 0; j < int(input.shape()[0]); j++) {
          sum += abs(eigenMatrixA(j, i));
        }

        for (j = 0; j < int(input.shape()[0]); j++) {
          result(j, i) = eigenMatrixA(j, i) / sum;
        }
      }
    }

    if (axis == 1 && p == 1) {
      int i, j;
      for (i = 0; i < int(input.shape()[0]); i++) {
        float sum = 0;
        for (j = 0; j < int(input.shape()[1]); j++) {
          sum += abs(eigenMatrixA(i, j));
        }

        for (j = 0; j < int(input.shape()[1]); j++) {
          result(i, j) = eigenMatrixA(i, j) / sum;
        }
      }
    }

    if (axis == 0 && p == 2) {
      int i, j;
      for (i = 0; i < int(input.shape()[1]); i++) {
        float sum = 0;
        for (j = 0; j < int(input.shape()[0]); j++) {
          sum += (eigenMatrixA(j, i) * eigenMatrixA(j, i));
        }
        for (j = 0; j < int(input.shape()[0]); j++) {
          result(j, i) = eigenMatrixA(j, i) / sqrt(sum);
        }
      }
    }

    // default cases
    if (axis == 1 && p == 2) {
      int i, j;
      for (i = 0; i < int(input.shape()[0]); i++) {
        float sum = 0;
        for (j = 0; j < int(input.shape()[1]); j++) {
          sum += (eigenMatrixA(i, j) * eigenMatrixA(i, j));
        }
        for (j = 0; j < int(input.shape()[1]); j++) {
          result(i, j) = eigenMatrixA(i, j) / sqrt(sum);
          // std::cout<<eigenMatrixA(i,j)<<" ";
        }
        // std::cout<<"\n";
      }
    }

    // result=eigenMatrixA;
    // Matrix<T, Dynamic, Dynamic> eResult=eigenMatrixA;

    // result.load( eResult.data() );
    return result;
  }
};
} // namespace dnnc