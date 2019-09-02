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

/*! This does element wise binary addition operation of two given N D tensors of same size.
    This operator supports multidirectional (i.e., Numpy-style) broadcasting.*/

template <typename T> class Add : public baseOperator<T> {
public:
  Add(std::string name = "opAdd") : baseOperator<T>(opAdd, name) {}
  tensor<T> compute(tensor<T> a /*!< : N D tensor input*/,
                   tensor<T> b /*!< : N D tensor input*/) {

    std::vector<DIMENSION> resultShape = binaryBroadcastReShape(a, b);
    tensor<T> result(resultShape);
    
    if (a.shape() != b.shape())
      throw std::invalid_argument(
          "tensor dimenions not appropriate for Add operator.");
    // Written for arbitrary Dimension.
    a.flatteninplace();
    b.flatteninplace();
    
    DNNC_EIGEN_VECTOR(eigenVectorA, a);
    DNNC_EIGEN_VECTOR(eigenVectorB, b);
    
    DNNC_EIGEN_VECTOR_CTOR(T) eResult;
    
    eResult.array() = eigenVectorA.array() + eigenVectorB.array();
    result.load(eResult.data());
    
    return result;
  }
  /*!<
  \return The output tensor of the same shape and type as input.
  */
};
} // namespace dnnc
