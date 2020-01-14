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
class ArgMax : public baseOperator<To, Ti, Ti> {

  int _axis = 0;
  int _keepdims = 1;

  void updateMax(To index, Ti value, To &maxIndex, Ti &maxValue) {
    if (value > maxValue) {
      maxValue = value;
      maxIndex = index;
    }
  }

public:
  ArgMax(std::string name = "opArgMax")
      : baseOperator<To, Ti, Ti>(opArgMax, name) {}

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_axis) {
      obj = _axis;
      return true;
    } else if (attrName == attr_keepdims) {
      obj = _keepdims;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_axis) {
      _axis = obj;
      return true;
    } else if (attrName == attr_keepdims) {
      _keepdims = obj;
      return true;
    }
    return false;
  }

  tensor<To> compute(tensor<Ti> input) override {

    if (!(this->template type_check<To, short int, int, long int>())) {
      SPDLOG_ERROR("Constrain output tensor type to int type.");
      return NULL_TENSOR<To>;
    }

    int rank = input.rank();

    if (_axis < -rank || _axis > rank - 1) {
      SPDLOG_ERROR("axis " + std::to_string(_axis) +
                   " is out of bounds for tensor.");
      return NULL_TENSOR<To>;
    }

    size_t axis = _axis + (_axis < 0 ? rank : 0); // ascertain positive number.

    std::vector<DIMENSION> axes = input.shape();
    size_t axis0 = rank > 0 ? axes[0] : 0;
    size_t axis1 = rank > 1 ? axes[1] : 0;
    size_t axis2 = rank > 2 ? axes[2] : 0;
    size_t axis3 = rank > 3 ? axes[3] : 0;
    size_t axis4 = rank > 4 ? axes[4] : 0;

    std::vector<DIMENSION> new_shape;
    if (_keepdims) {
      new_shape = axes;
    } else {
      if (input.rank() == 1) {
        new_shape.push_back(1);
      } else {
        for (size_t x = 0; x < axes.size(); x++) {
          if (x != axis) {
            new_shape.push_back(axes[x]);
          }
        }
      }
    }
    tensor<To> result(new_shape);

    if (axis == 0) {
      for (size_t j = 0; j == 0 || j < axis1; j++) {
        for (size_t k = 0; k == 0 || k < axis2; k++) {
          for (size_t l = 0; l == 0 || l < axis3; l++) {
            for (size_t m = 0; m == 0 || m < axis4; m++) {
              Ti maxValue;
              To maxIndex;
              for (size_t i = 0; i == 0 || i < axis0; i++) {
                if (i == 0) {
                  maxValue = input(i, j, k, l, m);
                  maxIndex = 0;
                } else {
                  Ti value =
                      input(i, j, k, l,
                            m); // TODO: use input.operator[] for performance
                  updateMax(i, value, maxIndex, maxValue);
                }
              }
              if (_keepdims) {
                for (size_t i = 0; i < axis0; i++)
                  result.load(maxIndex, i, j, k, l, m);
              } else {
                result.load(maxIndex, j, k, l, m);
              }
            }
          }
        }
      }
      return result;
    } else if (axis == 1) {
      for (size_t i = 0; i == 0 || i < axis0; i++) {
        for (size_t k = 0; k == 0 || k < axis2; k++) {
          for (size_t l = 0; l == 0 || l < axis3; l++) {
            for (size_t m = 0; m == 0 || m < axis4; m++) {
              Ti maxValue;
              To maxIndex;
              for (size_t j = 0; j == 0 || j < axis1; j++) {
                if (j == 0) {
                  maxValue = input(i, j, k, l, m);
                  maxIndex = 0;
                } else {
                  Ti value =
                      input(i, j, k, l,
                            m); // TODO: use input.operator[] for performance
                  updateMax(j, value, maxIndex, maxValue);
                }
              }
              if (_keepdims) {
                for (size_t j = 0; j < axis1; j++)
                  result.load(maxIndex, i, j, k, l, m);
              } else {
                result.load(maxIndex, i, k, l, m);
              }
            }
          }
        }
      }
      return result;
    } else if (axis == 2) {
      for (size_t i = 0; i == 0 || i < axis0; i++) {
        for (size_t j = 0; j == 0 || j < axis1; j++) {
          for (size_t l = 0; l == 0 || l < axis3; l++) {
            for (size_t m = 0; m == 0 || m < axis4; m++) {
              Ti maxValue;
              To maxIndex;
              for (size_t k = 0; k == 0 || k < axis2; k++) {
                if (k == 0) {
                  maxValue = input(i, j, k, l, m);
                  maxIndex = 0;
                } else {
                  Ti value =
                      input(i, j, k, l,
                            m); // TODO: use input.operator[] for performance
                  updateMax(k, value, maxIndex, maxValue);
                }
              }
              if (_keepdims) {
                for (size_t k = 0; k < axis2; k++)
                  result.load(maxIndex, i, j, k, l, m);
              } else {
                result.load(maxIndex, i, j, l, m);
              }
            }
          }
        }
      }
      return result;
    } else if (axis == 3) {
      for (size_t i = 0; i == 0 || i < axis0; i++) {
        for (size_t j = 0; j == 0 || j < axis1; j++) {
          for (size_t k = 0; k == 0 || k < axis2; k++) {
            for (size_t m = 0; m == 0 || m < axis4; m++) {
              Ti maxValue;
              To maxIndex;
              for (size_t l = 0; l == 0 || l < axis3; l++) {
                if (l == 0) {
                  maxValue = input(i, j, k, l, m);
                  maxIndex = 0;
                } else {
                  Ti value =
                      input(i, j, k, l,
                            m); // TODO: use input.operator[] for performance
                  updateMax(l, value, maxIndex, maxValue);
                }
              }
              if (_keepdims) {
                for (size_t l = 0; l < axis3; l++)
                  result.load(maxIndex, i, j, k, l, m);
              } else {
                result.load(maxIndex, i, j, k, m);
              }
            }
          }
        }
      }
      return result;
    } else if (axis == 4) {
      for (size_t i = 0; i == 0 || i < axis0; i++) {
        for (size_t j = 0; j == 0 || j < axis1; j++) {
          for (size_t k = 0; k == 0 || k < axis2; k++) {
            for (size_t l = 0; l == 0 || l < axis3; l++) {
              Ti maxValue;
              To maxIndex;
              for (size_t m = 0; m == 0 || m < axis4; m++) {
                if (m == 0) {
                  maxValue = input(i, j, k, l, m);
                  maxIndex = 0;
                } else {
                  Ti value =
                      input(i, j, k, l,
                            m); // TODO: use input.operator[] for performance
                  updateMax(m, value, maxIndex, maxValue);
                }
              }
              if (_keepdims) {
                for (size_t m = 0; m < axis4; m++)
                  result.load(maxIndex, i, j, k, l, m);
              } else {
                result.load(maxIndex, i, j, k, l);
              }
            }
          }
        }
      }
      return result;
    } else {
      SPDLOG_ERROR("axis " + std::to_string(_axis) +
                   " more than 5 for ArgMax is not supported.");
    }

    return NULL_TENSOR<To>;
  }
};
} // namespace dnnc
