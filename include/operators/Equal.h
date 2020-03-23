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
#include "core/broadcast.h"
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {

/*! Returns the tensor resulted from performing the equal logical operation
 * elementwise on the input tensors A and B*/
/*! This operator supports multidirectional (i.e., Numpy-style) broadcasting.*/

template <typename To, typename Ti>
class Equal : public baseOperator<To, Ti, Ti> {
public:
  Equal(std::string name = "opEqual")
      : baseOperator<To, Ti, Ti>(opEqual, name) {}

  tensor<To> compute(tensor<Ti> a /*!< : N D tensor input*/,
                     tensor<Ti> b /*!< : N D tensor input*/) {

    std::vector<DIMENSION> resultShape = binaryBroadcastReShape(a, b);
    tensor<To> result(resultShape);

    if (a.shape() != b.shape()) {
      SPDLOG_ERROR("tensor dimenions not appropriate for Equal operator.");
      return NULL_TENSOR<To>;
    }

    DNNC_EIGEN_ARRAY_MAP(eigenVectorA, Ti, a);
    DNNC_EIGEN_ARRAY_MAP(eigenVectorB, Ti, b);

    DNNC_EIGEN_VECTOR_CTOR(To) eResult;

    eResult.array() = eigenVectorA.array() == eigenVectorB.array();
    result.load(eResult.data());

    return result;
  }
  /*!<
  \return The output tensor of type bool and the same shape as input.
  */
};
} // namespace dnnc
