
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

/*! Flattens the input tensor into a 2D matrix. If input tensor has shape (d_0,
d_1, ... d_n) then
the output will have shape (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X
dn)*/

template <typename T> class Flatten : public baseOperator<T, T, T> {
protected:
  int axis = 1; /*!< Indicate up to which input dimensions (exclusive) should be
    flattened to the outer dimension of the output. The value for axis must be
    in the range [0, R], where R is the rank of the input tensor. When axis = 0,
    the shape of the output tensor is (1, (d_0 X d_1 ... d_n), where the shape
    of the input tensor is (d_0, d_1, ... d_n).*/
public:
  Flatten(std::string name = "opFlatten", int axis = 1)
      : baseOperator<T, T, T>(opFlatten, name) {
    this->axis = axis;
  }

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_axis) {
      obj = axis;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_axis) {
      axis = obj;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> a /*!< : N D tensor input of rank >= axis.*/) {

    if (a.rank() < (size_t)axis) {
      SPDLOG_ERROR("tensor rank or axis not appropriate for Flatten operator.");
      return NULL_TENSOR<T>;
    }

    size_t row = 1;
    size_t col = 1;
    size_t i;

    for (i = 0; i < (size_t)axis; i++) {
      row *= a.shape()[i];
    }
    for (i = axis; i < (size_t)a.rank(); i++) {
      col *= a.shape()[i];
    }

    std::vector<size_t> two_dimension{row, col};
    a.reshape(two_dimension);
    return a;
  }
  /*!<
  \returns a 2D tensor with the contents of the input tensor, with input
  dimensions up to axis flattened to the outer dimension of the output and
  remaining input dimensions flattened into the inner dimension of the output.
   */
};
} // namespace dnnc
