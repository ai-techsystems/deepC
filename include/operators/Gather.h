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
template <typename To, typename Ti>
class Gather : public baseOperator<To, To, Ti> {
protected:
  int axis = 0; /*!< Which axis to gather on. Negative value means
          counting dimensions from the back. Accepted
          range is [-r, r-1] where r = rank(data). */
  //  Gather attributes
public:
  Gather(std::string name = "opGather", int axis = 0)
      : baseOperator<To, To, Ti>(opGather, name) {
    this->axis = axis;
  }

  bool getAttribute(OPATTR attrName, float &obj) override {
    if (attrName == attr_axis) {
      obj = axis;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, float obj) override {
    if (attrName == attr_axis) {
      axis = obj;
      return true;
    }
    return false;
  }

  // duplicate of numpy.take(), link:
  // "https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html"

  tensor<To> compute(tensor<To> &a /*!<[float,double]: ND tensor*/,
                     tensor<Ti> &indices /*!<[int]: ND tensor*/) {

    if (!(this->template type_check<Ti, int>())) {
      SPDLOG_ERROR("Constrain axis tensor to integer type.");
      return NULL_TENSOR<To>;
    }

    if (a.rank() < 1) {
      SPDLOG_ERROR("Constrain input tensor rank greater than 0.");
      return NULL_TENSOR<To>;
    }

    if (axis < -a.rank() || axis > a.rank() - 1) {
      SPDLOG_ERROR("Constrain axis in range [-r,r-1] where r = rank(data)");
      return NULL_TENSOR<To>;
    }

    std::vector<size_t> Ni, Nj, Nk;
    tensor<To> result(a);

    for (int i = 0; i < axis; i++) {
      Ni.push_back(a.shape()[i]);
    }

    for (int i = axis; i < a.rank(); i++) {
      Nk.push_back(a.shape()[i]);
    }

    Nj = indices.shape();

    for (int i = 0; i < size(Ni); i++) {
      for (int j = 0; j < size(Nj); j++) {
        for (int k = 0; k < size(Nk); k++) {
          // out[ii + jj + kk] = a[ii + (indices[jj],) + kk]
          // result[Ni[i] + Nj[j] + Nk[k]] = a[Ni[i] + ]
        }
      }
    }

    return result;
  }
};
} // namespace dnnc
