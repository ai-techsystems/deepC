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

/*! This does element wise binary not operation of given N D tensor.*/

template <typename To, typename Ti>
class Not : public baseOperator<To, Ti, Ti> {
  //  Not attributes
public:
  Not(std::string name = "opNot") : baseOperator<To, Ti, Ti>(opNot, name) {}

  tensor<To> compute(tensor<Ti> &a /*!< [bool]: N D tensor input*/) {

    // This check is for ONNX standard
    // if (!(this->template type_check<T, bool>() ))
    //   SPDLOG_ERROR("Constrain input tensors to bool types.");

    tensor<To> result(a.shape(), a.name());

    DNNC_EIGEN_ARRAY_MAP(eigenVector, Ti, a);
    DNNC_EIGEN_VECTOR_CTOR(To) eResult;
    eResult.array() = !eigenVector.template cast<bool>().array();
    result.load(eResult.data());

    return result;
  }
  /*!<
  \return The output tensor of the same shape as input with dtype bool.
  */
};
} // namespace dnnc
