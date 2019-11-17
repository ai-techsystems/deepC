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
template <typename To, typename Ti1, typename Ti2>
class Reshape : public baseOperator<To, Ti1, Ti2> {
protected:
  int64_t shape_length(tensor<int64_t> &shape) {
    int64_t new_length = 1;
    for (size_t i = 0; i < shape.length(); i++)
      new_length = new_length * shape[i];
    return new_length;
  }

public:
  Reshape(std::string name = "opReshape")
      : baseOperator<To, Ti1, Ti2>(opReshape, name) {}

  tensor<To> compute(tensor<Ti1> input, tensor<int64_t> shape) {

    // A dimension could also be 0, in which case
    // the actual dimension value is unchanged,
    // i.e. taken from the input tensor
    for (size_t i = 0; i < shape.shape().size(); i++) {
      if (shape[i] == 0 && input.rank() > i)
        shape[i] = input.shape()[i];
    }

    // At most one dimension of the new shape can be -1.
    // In this case, the value is inferred from the
    // size of the tensor and the remaining dimensions.
    for (size_t i = 0; i < shape.shape().size(); i++) {
      if (shape[i] == -1) {
        shape[i] = 1;
        shape[i] = static_cast<int64_t>(input.length() / shape_length(shape));
      }
    }

    tensor<To> newTensor = input.template asType<To>();
    std::vector<size_t> shape_vec = shape.asType<size_t>();
    newTensor.reshape(shape_vec);
    return newTensor;
  }
};
} // namespace dnnc
