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
#include <cmath>
#include <string>

using namespace Eigen;

namespace dnnc {
template <typename T> class LRN : public baseOperator<T> {
protected:
  float alpha = 0.0001;
  float beta = 0.75;
  float bias = 1.0;
  int size;

public:
  LRN(int size, std::string name = "opLRN", float alpha = 0.0001,
      float beta = 0.75, float bias = 1.0)
      : baseOperator<T>(opLRN, name) {
    this->alpha = alpha;
    this->beta = beta;
    this->bias = bias;
    this->size = size;
  }

  static bool compare() {
    return ((typeid(T) == typeid(float)) || (typeid(T) == typeid(double)));
  }
  bool getAttribute(OPATTR attrName, float &obj) {
    if (attrName == attr_alpha) {
      obj = alpha;
      return true;
    } else if (attrName == attr_beta) {
      obj = beta;
      return true;
    } else if (attrName == attr_bias) {
      obj = bias;
      return true;
    }
    return false;
  }
  bool getAttribute(OPATTR attrName, int &obj) {
    if (attrName == attr_size) {
      obj = size;
      return true;
    }
    return false;
  }
  tensor<T> compute(tensor<T> &input) {
    if (!compare())
      throw std::invalid_argument(
          "Constrain input and output types to float tensors.");

    tensor<T> result(input.shape(), input.name());
    std::vector<size_t> original_shape = input.shape();

    size_t size = 1;
    for (size_t i = 2; i < input.rank(); i++) {
      size *= input.shape()[i];
    }

    std::vector<size_t> shape{input.shape()[0], input.shape()[1], size};
    input.reshape(shape);
    result.reshape(shape);
    T sq_sum = 0;
    for (size_t c = 0; c < input.shape()[1]; c++) {
      int temp1 = c - floor((size - 1) / 2);
      int lower = (0 > temp1) ? 0 : temp1;
      int temp2 = c + ceil((size - 1) / 2);
      int upper = ((int(input.shape()[1]) - 1) < temp2)
                      ? (int(input.shape()[1]) - 1)
                      : temp2;
      /*std::cout << "Current Channel=" << c << "\n";
      std::cout << "Upper=" << upper << "Lower" << lower << "\n";*/
      for (int i = lower; i <= upper; i++) {
        for (size_t j = 0; j < input.shape()[0]; j++) {
          for (size_t k = 0; k < size; k++) {
            /*std::cout << input(j, i, k) << ',';*/
            sq_sum += input(j, i, k) * input(j, i, k);
          }
        }
        /*std::cout << "\n";
        std::cout << "sq_sum= " << sq_sum << std::endl;*/
        for (size_t j = 0; j < input.shape()[0]; j++) {
          for (size_t k = 0; k < size; k++) {

            result(j, i, k) =
                input(j, i, k) / pow((bias + alpha / size * sq_sum), beta);
          }
        }
        sq_sum = 0;
      }
    }
    result.reshape(original_shape);
    return result;
  }
};
} // namespace dnnc
