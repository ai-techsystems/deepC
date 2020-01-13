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

/*! Dropout takes one input floating tensor and produces two tensor outputs,
    output (floating tensor) and mask (Tensor<bool>). Depending on whether
    it is in test mode or not, the output Y will either be a random dropout
    or a simple copy of the input. Note that our implementation of Dropout
    does scaling in the training phase, so during testing nothing needs to be
   done.*/

template <typename T> class Dropout : public baseOperator<T, T, T> {
protected:
  float ratio = 0.5; /*!< The ratio of random dropout. */
  //  Dropout attributes
public:
  Dropout(std::string name = "opDropout", float ratio = 0.5)
      : baseOperator<T, T, T>(opDropout, name) {
    this->ratio = ratio;
  }

  bool getAttribute(OPATTR attrName, float &obj) override {
    if (attrName == attr_ratio) {
      obj = ratio;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, float obj) override {
    if (attrName == attr_ratio) {
      ratio = obj;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> &a /*!<[float,double]: ND tensor*/) {

    if (!(this->template type_check<T, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }

    // Dropout is a NOOP for compiler. During training, it zeros
    // a fraction (attribute ratio) of the tensor a.
    return a;
  }
  /*!<
  \return The output tensor of the same shape and dtype as input.
  */
};
} // namespace dnnc
