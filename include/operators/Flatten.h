
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
template <typename T> class Flatten : public baseOperator<T> {
protected:
  int axis = 1;
  //  Flatten attributes
public:
  Flatten(std::string name = "opFlatten", int axis = 1)
      : baseOperator<T>(opFlatten, name) {
    this->axis = axis;
  }

  bool getAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_axis) {
      obj = axis;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> &a) {
    if (a.rank() < (size_t)axis)
      throw std::invalid_argument(
          "tensor rank or axis not appropriate for Flatten operator.");

    size_t row = 1;
    size_t col = 1;
    size_t i;

    for (i = 0; i < (size_t)axis; i++) {
      row *= a.shape()[i];
    }
    for (i = axis; i < (size_t)a.rank(); i++) {
      col *= a.shape()[i];
    }
    // std::cout<<a.shape()[0]<<" , "<<a.shape()[1]<<" , "<<a.shape()[2]<<" ,
    // "<<a.shape()[3]<<std::endl;

    std::vector<size_t> two_dimension{row, col};
    a.reshape(two_dimension);
    tensor<T> result = a;
    // std::cout<<result.shape()[0]<<" , "<<result.shape()[1]<<std::endl;

    return result;
  }
};
} // namespace dnnc