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
// Eigen cwise unsupported-tensors(written TODO in original doc)
// computing softmax as per axis(https://en.wikipedia.org/wiki/Softmax_function)

#pragma once
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {
  /*! The operator computes the logsoftmax (log of softmax) values for each 
  layer in the batch of the given input. The input is a 2-D tensor (Tensor)
  of size (batch_size x input_feature_dimensions). 
  The output tensor has the same shape and contains the logsoftmax
  values of the corresponding input. */
template <typename T> class LogSoftmax : public baseOperator<T> {
  //  LogSoftmax attributes
protected:
  // default
  int axis = 1; /*!< axis of the input. */

public:
  LogSoftmax(std::string name = "opLogSoftmax", int axis=1)
      : baseOperator<T>(opLogSoftmax, name) {
        this-> axis = axis;
      }

 bool getAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_axis) {
      obj = axis;
      return true;
    }
    return false;
  }

 static bool compare() {
    return ((typeid(T) == typeid(float)) || (typeid(T) == typeid(double)));
  }

  tensor<T> compute(tensor<T> &a) {
    if (!compare())
      throw std::invalid_argument(
          "Constrain input and output types to float tensors.");

    tensor<T> result(a.shape(), a.name());

    DNNC_EIGEN_MATRIX(eigenMatrixA, a);

    std::vector<size_t> shape{a.shape()[0], a.shape()[1]};

    if (a.rank() != 2) {
          a.reshape(shape);
    }    
      // default cases
      if (axis == 1) {
        int i, j;
        for (i = 0; i < int(a.shape()[0]); i++) {
          float sum = 0;
          for (j = 0; j < int(a.shape()[1]); j++) {
            sum += exp(eigenMatrixA(i, j)); 
          }
          for (j = 0; j < int(a.shape()[1]); j++) {
            eigenMatrixA(i, j) = log(exp(eigenMatrixA(i, j)) /sum);
          }
        }
      }

      else if (axis == 0) {
        int i, j;
        for (i = 0; i < int(a.shape()[1]); i++) {
          float sum = 0;
          for (j = 0; j < int(a.shape()[0]); j++) {
            sum += exp(eigenMatrixA(j, i));
          }
          for (j = 0; j < int(a.shape()[0]); j++) {
            eigenMatrixA(j, i) = log(exp(eigenMatrixA(j, i)) / (sum));
          }
        }
      }

      Matrix<T, Dynamic, Dynamic> eResult = eigenMatrixA;
      result.load(eResult.data());

      return result;
     
  }
};
} // namespace dnnc
