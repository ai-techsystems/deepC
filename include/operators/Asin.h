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
template <typename T> class Asin : public baseOperator<T, T, T> {
public:
  Asin(std::string name = "opAsin") : baseOperator<T, T, T>(opAsin, name) {}

  tensor<T> compute(tensor<T> &a) {

    tensor<T> result(a.shape());

    for (size_t i = 0; i < a.length(); i++) {
      float x = a[i];
      if (x < -1 && x > 1) {
        LOG_ERROR("Error : the value of tensor element is not "
                     "lying in the domain of arc sine ");
        return NULL_TENSOR<T>;
      }
      result[i] = asin(x);
    }

    return result;
  }
};
} // namespace dnnc
