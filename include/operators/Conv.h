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
template <typename T> class Conv : public baseOperator<T, T, T> {
  //  Conv attributes
protected:
  std::string auto_pad;
  std::vector<int> dilations;
  int group;
  std::vector<DIMENSION> kernal_shape;
  std::vector<size_t> pads;
  std::vector<size_t> strides;

public:
  Conv(std::string name = "opConv", std::string auto_pad = "NOTSET",
       std::vector<int> dilations = {}, int group = 1,
       std::vector<DIMENSION> kernal_shape = {}, std::vector<size_t> pads = {},
       std::vector<size_t> strides = {1})
      : baseOperator<T, T, T>(opConv, "opConv") {}

  /*bool getAttribute<int>(OPATTR attrName, std::vector<size_t> &obj) {
  if (attrName == attr_kernal_shape) {
    obj = kernal_shape;
    return true;
  } else if (attrName == attr_pads) {
    obj = pads;
    return true;
  } else if (attrName == attr_strides) {
    obj = strides;
    return true;
  }
  return false;
}

bool getAttribute<int>(OPATTR attrName, std::vector<int> &obj) {
  if (attrName == attr_dilations) {
    obj = dilations;
    return true;
  }
  return false;
}

bool getAttribute<int>(OPATTR attrName, int &obj) {
  if (attrName == attr_group) {
    obj = group;
    return true;
  }
  return false;
}

bool getAttribute<int>(OPATTR attrName, string &obj) {
  if (attrName == attr_auto_pad) {
    obj = auto_pad;
    return true;
  }
  return false;
  } */

  tensor<T> compute(tensor<T> &X, tensor<T> &W, tensor<T> &B) {}

}; // template class
} // namespace dnnc
