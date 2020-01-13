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
/*! Local response normalization. This Normalizes over local input regions. The
 * local region is defined across the channels.*/
/*! For an element \f$X[n, c, d1, ..., dk]\f$ in a tensor of shape \f$ (N * C *
 * D1 * D2* ...* Dk)\f$, its region is \f$\left \{ X[n, i, d1, ..., dk] \mid
 * max(0, c - floor((size - 1) / 2)) \leq  i \leq  min(C - 1, c + ceil((size -
 * 1) / 2))  \right \}\f$ */
/*! Define \f$square\_sum[n, c, d1, ..., dk] = \sum_{i=max(0, c - floor((size -
 * 1) / 2))}^{min(C - 1, c + ceil((size - 1) / 2))}(X[n, i, d1, ..., dk] ^ 2)\f$
 */
/*! and then apply \f$ Y[n, c, d1, ..., dk] = X[n, c, d1, ...,
 * dk]/\begin{pmatrix}bias + \alpha*square\_sum[n, c, d1, ..., dk]
 * \end{pmatrix}^{beta} \f$ */
template <typename T> class LRN : public baseOperator<T, T, T> {
protected:
  float alpha = 0.0001; /*!< Scaling parameter.*/
  float beta = 0.75;    /*!< The exponent.*/
  float bias = 1.0;
  int size; /*!< (Required) The number of channels to sum over.*/

public:
  LRN(int size, std::string name = "opLRN", float alpha = 0.0001,
      float beta = 0.75, float bias = 1.0)
      : baseOperator<T, T, T>(opLRN, name) {
    this->alpha = alpha;
    this->beta = beta;
    this->bias = bias;
    this->size = size;
  }
  bool getAttribute(OPATTR attrName, float &obj) override {
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
  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_size) {
      obj = size;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, float obj) override {
    if (attrName == attr_alpha) {
      alpha = obj;
      return true;
    } else if (attrName == attr_beta) {
      beta = obj;
      return true;
    } else if (attrName == attr_bias) {
      bias = obj;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_size) {
      size = obj;
      return true;
    }
    return false;
  }
  tensor<T> compute(tensor<T> input/*!< Input data tensor from the previous operator;
    dimensions for image case are \f$(N * C * H * W)\f$, where N is the batch size,
    C is the number of channels, and H and W are the height and the width of the data.
    For non image case, the dimensions are in the form of  \f$(N * C * D1 * D2* ...* Dn)\f$,
     where N is the batch size.*/) {
    if (!(this->template type_check<T, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }

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
  /*!<
   \return Output tensor, which has the shape and type as input tensor
 */
};
} // namespace dnnc
