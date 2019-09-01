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

#include "core/broadcast.h"

using namespace dnnc;

// reference: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
template <typename T>
tensor<T> broadcast(const tensor<T> a,
                    const std::vector<DIMENSION> targetShape) {
  // TODO:
  // 1. uni-directional broadcasting
  // 2. multi-directional broadcasting
  tensor<T> null;

  // multi-directional broadcasting
  if (a.shape() == targetShape) {
    // nothing to do
    return a;
  } else if ((a.rank() == 1) && (a.shape[0] == 1)) {
    // a is a scalar
    tensor<T> result(targetShape);
    size_t num_elems = std::accumulate(begin(targetShape), end(targetShape), 1,
                                       std::multiplies<>());
    T mem_data = malloc(num_elems * sizeof(T));
    mem_set(mem_data, a, num_elems);
    result.load(mem_data);
    return result;
  } else {
    std::cout << "Not supported";
  }

  return null;
}
