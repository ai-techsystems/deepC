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
class ReduceSum : public baseOperator<To, Ti, Ti> {
  std::vector<int> _axis = {};
  int _keepdims = 1;

public:
  ReduceSum(std::string name = "opReduceSum")
      : baseOperator<To, Ti, Ti>(opReduceSum, name) {}

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_keepdims) {
      obj = _keepdims;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_keepdims) {
      _keepdims = obj;
      return true;
    }
    return false;
  }
  bool getAttribute(OPATTR attrName, std::vector<int> &obj) override {
    if (attrName == attr_axis) {
      obj = _axis;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, std::vector<int> obj) override {
    if (attrName == attr_axis) {
      _axis = obj;
      return true;
    }
    return false;
  }

  tensor<To> compute(tensor<Ti> input) override {
    int rank = input.rank();

    // check that values in _axis are less than or equal to the rank
    for (int axis : _axis) {
      if (axis > rank - 1) {
        SPDLOG_ERROR("Axis is larger than input tensor rank.");
        return NULL_TENSOR<To>;
      }
      if (axis < 0) {
        SPDLOG_ERROR("Axis value is negative.");
        return NULL_TENSOR<To>;
      }
    }

    if (rank == 0) {
      return input.template asType<To>();
    }

    if (rank == 1 || _axis.empty()) {
      tensor<To> result({1});
      To in_sum = input.sum();
      result.load({in_sum});
      return result;
    }

    std::vector<DIMENSION> shape = input.shape();

    std::vector<DIMENSION> new_shape = input.shape();

    // sort axes from greatest to least
    sort(_axis.begin(), _axis.end(), std::greater<int>());

    // new shape of the tensor will be composed of the original tensor shape
    // with values in _axis removed
    for (int axis : _axis) {
      new_shape.erase(new_shape.begin() + axis);
    }

    // resulting tensor has the shape new_shape
    tensor<To> result(new_shape, "", INIT_ZERO);

    if (rank == 2) {
      for (size_t l0 = 0; l0 < shape[0]; l0++) {
        for (size_t l1 = 0; l1 < shape[1]; l1++) {
          std::vector<size_t> k;

          k.push_back(l0);
          k.push_back(l1);

          for (int axis : _axis) {
            k.erase(k.begin() + axis);
          }
          result(k) += static_cast<To>(input(l0, l1));
        }
      }
      if (_keepdims) {
        for (size_t l0 = 0; l0 < shape[0]; l0++) {
          for (size_t l1 = 0; l1 < shape[1]; l1++) {
            std::vector<size_t> k;

            k.push_back(l0);
            k.push_back(l1);

            for (int axis : _axis) {
              k.erase(k.begin() + axis);
            }

            input(l0, l1) = result(k);
          }
        }
        result = input.template asType<To>();
      }
    }
    if (rank == 3) {
      for (size_t l0 = 0; l0 < shape[0]; l0++) {
        for (size_t l1 = 0; l1 < shape[1]; l1++) {
          for (size_t l2 = 0; l2 < shape[2]; l2++) {
            std::vector<size_t> k;

            k.push_back(l0);
            k.push_back(l1);
            k.push_back(l2);

            for (int axis : _axis) {
              k.erase(k.begin() + axis);
            }

            result(k) += static_cast<To>(input(l0, l1, l2));
          }
        }
      }
      if (_keepdims) {
        for (size_t l0 = 0; l0 < shape[0]; l0++) {
          for (size_t l1 = 0; l1 < shape[1]; l1++) {
            for (size_t l2 = 0; l2 < shape[2]; l2++) {
              std::vector<size_t> k;

              k.push_back(l0);
              k.push_back(l1);
              k.push_back(l2);

              for (int axis : _axis) {
                k.erase(k.begin() + axis);
              }

              input(l0, l1, l2) = result(k);
            }
          }
        }
        result = input.template asType<To>();
      }
    }
    if (rank == 4) {
      for (size_t l0 = 0; l0 < shape[0]; l0++) {
        for (size_t l1 = 0; l1 < shape[1]; l1++) {
          for (size_t l2 = 0; l2 < shape[2]; l2++) {
            for (size_t l3 = 0; l3 < shape[3]; l3++) {
              std::vector<size_t> k;

              k.push_back(l0);
              k.push_back(l1);
              k.push_back(l2);
              k.push_back(l3);

              for (int axis : _axis) {
                k.erase(k.begin() + axis);
              }

              result(k) += static_cast<To>(input(l0, l1, l2, l3));
            }
          }
        }
      }
      if (_keepdims) {
        for (size_t l0 = 0; l0 < shape[0]; l0++) {
          for (size_t l1 = 0; l1 < shape[1]; l1++) {
            for (size_t l2 = 0; l2 < shape[2]; l2++) {
              for (size_t l3 = 0; l3 < shape[3]; l3++) {
                std::vector<size_t> k;

                k.push_back(l0);
                k.push_back(l1);
                k.push_back(l2);
                k.push_back(l3);

                for (int axis : _axis) {
                  k.erase(k.begin() + axis);
                }

                input(l0, l1, l2, l3) = result(k);
              }
            }
          }
        }
        result = input.template asType<To>();
      }
    }
    if (rank == 5) {
      for (size_t l0 = 0; l0 < shape[0]; l0++) {
        for (size_t l1 = 0; l1 < shape[1]; l1++) {
          for (size_t l2 = 0; l2 < shape[2]; l2++) {
            for (size_t l3 = 0; l3 < shape[3]; l3++) {
              for (size_t l4 = 0; l4 < shape[4]; l4++) {
                std::vector<size_t> k;

                k.push_back(l0);
                k.push_back(l1);
                k.push_back(l2);
                k.push_back(l3);
                k.push_back(l4);

                for (int axis : _axis) {
                  k.erase(k.begin() + axis);
                }

                result(k) += static_cast<To>(input(l0, l1, l2, l3, l4));
              }
            }
          }
        }
      }
      if (_keepdims) {
        for (size_t l0 = 0; l0 < shape[0]; l0++) {
          for (size_t l1 = 0; l1 < shape[1]; l1++) {
            for (size_t l2 = 0; l2 < shape[2]; l2++) {
              for (size_t l3 = 0; l3 < shape[3]; l3++) {
                for (size_t l4 = 0; l4 < shape[4]; l4++) {
                  std::vector<size_t> k;

                  k.push_back(l0);
                  k.push_back(l1);
                  k.push_back(l2);
                  k.push_back(l3);
                  k.push_back(l4);

                  for (int axis : _axis) {
                    k.erase(k.begin() + axis);
                  }

                  input(l0, l1, l2, l3, l4) = result(k);
                }
              }
            }
          }
        }
        result = input.template asType<To>();
      }
    }
    /*
    if keepdims
      for every element in the orignal matrix, set it equal to the corresponding
    element in the reduced return input.template asType<To>();
    */
    return result;
  }
};
} // namespace dnnc
