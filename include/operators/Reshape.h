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
template <typename To> class Reshape : public baseOperator<To, To, To> {
protected:
  long int shape_length(tensor<long int> &shape) {
    long int new_length = 1;
    for (size_t i = 0; i < shape.length(); i++)
      new_length = new_length * shape[i];
    return new_length;
  }

public:
  Reshape(std::string name = "opReshape")
      : baseOperator<To, To, To>(opReshape, name) {}

  tensor<To> compute(tensor<To> input, tensor<long int> shape) {

    // A dimension could also be 0, in which case
    // the actual dimension value is unchanged,
    // i.e. taken from the input tensor
    for (size_t i = 0; i < shape.length(); i++) {
      if (shape[i] == 0 && input.rank() > i)
        shape[i] = input.shape()[i];
    }

    // At most one dimension of the new shape can be -1.
    // In this case, the value is inferred from the
    // size of the tensor and the remaining dimensions.
    for (size_t i = 0; i < shape.length(); i++) {
      if (shape[i] == -1) {
        shape[i] = 1;
        shape[i] = static_cast<long int>(input.length() / shape_length(shape));
      }
    }

    tensor<To> newTensor = input.copy();
    std::vector<size_t> shape_vec = shape.asType<size_t>();
    newTensor.reshape(shape_vec);
    return newTensor;
  }
};
} // namespace dnnc
