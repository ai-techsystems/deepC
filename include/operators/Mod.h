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
#include <math.h>
#include <string>

using namespace Eigen;

namespace dnnc {
template <typename T> class Mod : public baseOperator<T, T, T> {
  //  Mod attributes
protected:
  int fmod = 0;

public:
  Mod(std::string name = "opMod", int fmod = 0)
      : baseOperator<T, T, T>(opMod, name) {
    this->fmod = fmod;

    // Check for fmod or not
    if ((fmod == 0) && ((this->template type_check<T, float, double>()))) {
      SPDLOG_ERROR("Set fmod to 1 to pass float values.");
      return;
    }
  }

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_mode) {
      obj = fmod;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_mode) {
      fmod = obj;
      return true;
    }
    return false;
  }

  static T mod_function(T x, T y) { return (T)((int)x % (int)y); }

  static T fmod_function(T x, T y) { return ::fmod(x, y); }

  tensor<T> compute(tensor<T> &a /*!< : N D tensor input*/,
                    tensor<T> &b /*!< : N D tensor input*/) {

    std::vector<DIMENSION> resultShape = binaryBroadcastReShape(a, b);
    tensor<T> result(resultShape);

    if (a.shape() != b.shape()) {
      SPDLOG_ERROR("tensor dimenions not appropriate for Mod operator.");
      return NULL_TENSOR<T>;
    }

    DNNC_EIGEN_ARRAY_MAP(eigenVectorA, T, a);
    DNNC_EIGEN_ARRAY_MAP(eigenVectorB, T, b);

    DNNC_EIGEN_VECTOR_CTOR(T) eigen_result;

    if (fmod) {
      eigen_result.array() =
          eigenVectorA.array().binaryExpr(eigenVectorB.array(), &fmod_function);
    } else {
      eigen_result.array() =
          eigenVectorA.array().binaryExpr(eigenVectorB.array(), &mod_function);
    }

    result.load(eigen_result.data());

    return result;
  }
};
} // namespace dnnc
