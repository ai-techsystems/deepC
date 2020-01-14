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
template <typename To, typename Ti1, typename Ti2>
class OneHot : public baseOperator<To, Ti1, Ti2> {
  //  OneHot attributes
  int _axis;

public:
  OneHot(std::string name = "opOneHot", int xis = -1)
      : baseOperator<To, Ti1, Ti2>(opOneHot, name), _axis(xis) {}

  bool getAttribute(OPATTR attrName, int &xis) override {
    if (attrName == attr_axis) {
      xis = _axis;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int xis) override {
    if (attrName == attr_axis) {
      _axis = xis;
      return true;
    }
    return false;
  }

  tensor<To> compute(tensor<Ti1> indices, Ti2 depth, tensor<To> values) {
    if (values.rank() != 1 || values.length() != 2) {
      SPDLOG_ERROR("invalid values rank or length.");
      return NULL_TENSOR<To>;
    }

    To off_value = values[0];
    To on_value = values[1];

    if (!(this->template type_check<Ti1, int, float, double>())) {
      SPDLOG_ERROR("Constrain input tensor indices to numeric tensors.");
      return NULL_TENSOR<To>;
    }

    if (!(this->template type_check<Ti2, int, float, double>())) {
      SPDLOG_ERROR("Constrain scalar depth to numeric values.");
      return NULL_TENSOR<To>;
    }

    size_t axis = _axis < 0 || static_cast<size_t>(_axis) > indices.rank()
                      ? indices.rank()
                      : _axis;

    std::vector<size_t> rshape = indices.shape();
    rshape.insert(rshape.begin() + axis, static_cast<size_t>(depth));

    tensor<To> result(rshape, indices.name() + "_1hot", INIT_VALUE, off_value);

    // TODO: Now add on_value selectively.
    if (indices.rank() == 1) {
    } else if (indices.rank() == 2) {
      size_t i0 = 0, sz0 = indices.shape()[0];
      size_t i1 = 0, sz1 = indices.shape()[1];
      for (i0 = 0; i0 < sz0; i0++) {
        for (i1 = 0; i1 < sz1; i1++) {
          std::vector<size_t> new_index{i0, i1};
          Ti1 idx = indices(new_index);
          // Any entries in the 'indices' input tensor
          // with values outside the range [-depth, depth-1]
          // will result in one-hot representation with all
          // 'off_value' values in the output tensor.
          if (idx < -depth || idx > depth - 1)
            continue;
          int idepth = idx < 0 ? depth + idx : idx;
          new_index.insert(new_index.begin() + axis,
                           static_cast<size_t>(idepth));
          result.load(on_value, new_index);
        }
      }
    } else if (indices.rank() == 3) {
      size_t i0 = 0, sz0 = indices.shape()[0];
      size_t i1 = 0, sz1 = indices.shape()[1];
      size_t i2 = 0, sz2 = indices.shape()[2];
      for (i0 = 0; i0 < sz0; i0++) {
        for (i1 = 0; i1 < sz1; i1++) {
          for (i2 = 0; i2 < sz2; i2++) {
            std::vector<size_t> new_index{i0, i1, i2};
            Ti1 idx = indices(new_index);
            // Any entries in the 'indices' input tensor
            // with values outside the range [-depth, depth-1]
            // will result in one-hot representation with all
            // 'off_value' values in the output tensor.
            if (idx < -depth || idx > depth - 1)
              continue;
            int idepth = idx < 0 ? depth + idx : idx;
            new_index.insert(new_index.begin() + axis,
                             static_cast<size_t>(idepth));
            result.load(on_value, new_index);
          }
        }
      }
    } else if (indices.rank() == 4) {
      size_t i0 = 0, sz0 = indices.shape()[0];
      size_t i1 = 0, sz1 = indices.shape()[1];
      size_t i2 = 0, sz2 = indices.shape()[2];
      size_t i3 = 0, sz3 = indices.shape()[3];
      for (i0 = 0; i0 < sz0; i0++) {
        for (i1 = 0; i1 < sz1; i1++) {
          for (i2 = 0; i2 < sz2; i2++) {
            for (i3 = 0; i3 < sz3; i3++) {
              std::vector<size_t> new_index{i0, i1, i2, i3};
              Ti1 idx = indices(new_index);
              // Any entries in the 'indices' input tensor
              // with values outside the range [-depth, depth-1]
              // will result in one-hot representation with all
              // 'off_value' values in the output tensor.
              if (idx < -depth || idx > depth - 1)
                continue;
              int idepth = idx < 0 ? depth + idx : idx;
              new_index.insert(new_index.begin() + axis,
                               static_cast<size_t>(idepth));
              result.load(on_value, new_index);
            }
          }
        }
      }
    }

    return result;
  }
};
} // namespace dnnc
