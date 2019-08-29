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
#include "tensor.h"
#include <string.h>

// reference: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md

namespace dnnc {
template <typename T>
tensor<T> broadcast(const tensor<T> a,
                    const std::vector<DIMENSION> targetShape) {
  // TODO:
  // 1. uni-directional broadcasting
  // 2. multi-directional broadcasting

  DIMENSION aNumDims = a.shape().size();
  DIMENSION targetNumDims = targetShape.size();

  // multi-directional broadcasting
  if (aNumDims > targetNumDims) {
    // Can't broadcast to fewer dimensions!
    return dnnc::NULL_TENSOR<T>;
  }
  if (a.shape() == targetShape) {
    // nothing to do
    return a;
  } else if ((a.rank() == 1) && (a.shape()[0] == 1)) {
    // a is a scalar
    size_t num_elems = std::accumulate(begin(targetShape), end(targetShape), 1,
                                       std::multiplies<>());
    T *mem_data = (T *)malloc(num_elems * sizeof(T));
    for (size_t i = 0; i < num_elems; i++) {
      mem_data[i] = a.data()[0];
    }
    tensor<T> result(mem_data, targetShape);
    return result;
  }
  if (aNumDims == targetNumDims) {

    std::vector<size_t> resultShape(targetNumDims);
    // Determine broadcast result shape
    for (size_t i = 0; i < targetNumDims; i++) {
      if ((a.shape()[i] == targetShape[i]) || (a.shape()[i] == 1)) {
        resultShape[i] = targetShape[i];
      } else {
        // Can't broadcast unless a's dimensions is 1
        return dnnc::NULL_TENSOR<T>;
      }
    }

    tensor<T> result(resultShape);
    // Determine broadcast result values
    DIMENSION d0 = targetShape[0];
    DIMENSION d1 = targetShape[1];
    DIMENSION d2 = targetShape[2];
    if (targetNumDims == 3) {
      for (size_t i = 0; i < d0; i++) {
        for (size_t j = 0; j < d1; j++) {
          for (size_t k = 0; k < d2; k++) {
            size_t i1 = i, j1 = j, k1 = k;
            if (a.shape()[0] != d0) {
              i1 = 0;
            }
            if (a.shape()[1] != d1) {
              j1 = 0;
            }
            if (a.shape()[2] != d2) {
              k1 = 0;
            }
            result(i, j, k) = a(i1, j1, k1);
          }
        }
      }
    } else if (targetNumDims == 2) {
      for (size_t i = 0; i < d0; i++) {
        for (size_t j = 0; j < d1; j++) {
          size_t i1 = i, j1 = j;
          if (a.shape()[0] != d0) {
            i1 = 0;
          }
          if (a.shape()[1] != d1) {
            j1 = 0;
          }
          result(i, j) = a(i1, j1);
        }
      }
    } else {
      std::cout << "Unsupported!\n";
    }

    return result;

  } else if (aNumDims < targetNumDims) {

    std::vector<size_t> aReShape(targetNumDims);

    size_t diffNumDims = targetNumDims - aNumDims;

    for (size_t i = 0; i < targetNumDims; i++) {
      if (i < diffNumDims) {
        aReShape[i] = 1;
      } else {
        aReShape[i] = a.shape()[i - diffNumDims];
      }
    }
    tensor<T> aReShaped(aReShape);

    aReShaped.load(a.data());

    return broadcast<T>(aReShaped, targetShape);

  } else {
    std::cout << "Not supported";
  }

  return dnnc::NULL_TENSOR<T>;
}
} // namespace dnnc
