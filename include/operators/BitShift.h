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

/*! This does element wise binary left shift and right shift operation of two
   given N D tensors of same size. This operator supports multidirectional
   (i.e., Numpy-style) broadcasting.*/

template <typename T> class BitShift : public baseOperator<T, T, T> {
protected:
  std::string direction = ""; /*!< Direction of BitShift. */

  // Eigen does not support bitshift
  // So binaryExpr is needed to work around that limitation.
  // https://stackoverflow.com/questions/29127497/bitwise-operations-in-eigen

  /*! Element wise Left-Shift-Function*/
  static T left_shift_func(T x, T y) { return (x << y); }

  /*! Element wise Right-Shift-Function*/
  static T right_shift_func(T x, T y) { return (x >> y); }

  template <typename Scalar>
  inline DNNC_EIGEN_VECTOR_CTOR(Scalar)
      eigenArrayLeftShift(Map<DNNC_EIGEN_VECTOR_CTOR(Scalar)> &a,
                          Map<DNNC_EIGEN_VECTOR_CTOR(Scalar)> &b) {
    return (a.array().binaryExpr(b.array(), &left_shift_func));
  }

  template <typename Scalar>
  inline DNNC_EIGEN_VECTOR_CTOR(Scalar)
      eigenArrayRightShift(Map<DNNC_EIGEN_VECTOR_CTOR(Scalar)> &a,
                           Map<DNNC_EIGEN_VECTOR_CTOR(Scalar)> &b) {
    return (a.array().binaryExpr(b.array(), &right_shift_func));
  }

public:
  BitShift(std::string name = "opBitShift", std::string direction = "")
      : baseOperator<T, T, T>(opBitShift, name) {
    this->direction = direction;
  }

  bool getAttribute(OPATTR attrName, std::string &obj) override {
    if (attrName == attr_direction) {
      obj = direction;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, std::string obj) override {
    if (attrName == attr_direction) {
      direction = obj;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> a /*!<[int]: ND tensor*/,
                    tensor<T> b /*!<[int]: ND tensor*/) override {

    std::vector<DIMENSION> resultShape = binaryBroadcastReShape(a, b);
    tensor<T> result(resultShape);

    if (a.shape() != b.shape()) {
      SPDLOG_ERROR("tensor dimenions not appropriate for BitShift operator.");
      return NULL_TENSOR<T>;
    }

    if ((direction != "LEFT") && (direction != "RIGHT")) {
      SPDLOG_ERROR("Specify direction to 'LEFT' or 'RIGHT'");
      return NULL_TENSOR<T>;
    }

    if (!(this->template type_check<T, int>())) {
      SPDLOG_ERROR("Constrain input and output types to int tensors.");
      return NULL_TENSOR<T>;
    }

    DNNC_EIGEN_ARRAY_MAP(eigenVectorA, T, a);
    DNNC_EIGEN_ARRAY_MAP(eigenVectorB, T, b);

    DNNC_EIGEN_VECTOR_CTOR(T) eResult;

    if (direction == "LEFT")
      eResult.array() = eigenVectorA.array().binaryExpr(eigenVectorB.array(),
                                                        &left_shift_func);
    else if (direction == "RIGHT")
      eResult.array() = eigenVectorA.array().binaryExpr(eigenVectorB.array(),
                                                        &right_shift_func);

    result.load(eResult.data());

    return result;
  }
  /*!<
\return The output tensor of the same shape and type as input.
*/
};
} // namespace dnnc
