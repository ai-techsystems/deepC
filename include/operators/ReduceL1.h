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
class ReduceL1 : public baseOperator<To, Ti, Ti> {
  //  ReduceL1 attributes
protected:
  std::vector<int> axes = {};
  int keepdims = 1;

public:
  ReduceL1(std::string name = "opReduceL1")
      : baseOperator<To, Ti, Ti>(opReduceL1, name) {
    this->axes = axes;
    this->keepdims = keepdims;
  }
  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_keepdims) {
      obj = keepdims;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_keepdims) {
      keepdims = obj;
      return true;
    }
    return false;
  }
  bool getAttribute(OPATTR attrName, std::vector<int> &obj) override {
    if (attrName == attr_axis) {
      obj = axes;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, std::vector<int> obj) override {
    if (attrName == attr_axis) {
      axes = obj;
      return true;
    }
    return false;
  }

  tensor<To> compute(tensor<Ti> a /*!< : N D tensor input*/) override {

    int rank = a.rank();
    int reductions = axes.size();

    std::vector<int> arr(rank, 0);
    for (int i = 0; i < axes.size(); i++) {
      if (axes.at(i) >= rank || arr.at(axes.at(i)) >= 1) {
        SPDLOG_ERROR("Inputted axes not appropriate for Reduce operator.");
        return NULL_TENSOR<To>;
      } else {
        arr.at(axes.at(i))++;
      }
    }

    if (reductions == 0) {
      for (int i = 0; i < arr.size(); i++) {
        arr.at(i) = 1;
      }
    }

    std::vector<unsigned long> dimensions;

    for (int j = 0; j < arr.size(); j++) {
      if (arr.at(j) == 0) {
        dimensions.push_back(a.shape()[j]);
      } else if (keepdims) {
        dimensions.push_back(1);
      }
    }

    if (dimensions.size() == 0) {
      dimensions.push_back(1);
    }

    if (rank < reductions) {
      SPDLOG_ERROR("tensor dimenions not appropriate for Reduce operator.");
      return NULL_TENSOR<To>;
    }

    if (rank == 4) {

      tensor<To> result(dimensions);
      DNNC_EIGEN_TENSOR4D_MAP(tensor4D, Ti, a);
      // tensor4D = tensor4D.abs();

      if (reductions == 0) {
        std::array<int, 4> dims = {0, 1, 2, 3};
        Tensor<To, 0, RowMajor> b = tensor4D.abs().sum(dims);
        result.load(b.data());
      } else if (reductions == 1) {
        std::array<int, 1> dims;
        std::copy_n(axes.begin(), reductions, dims.begin());
        Tensor<To, 3, RowMajor> b = tensor4D.abs().sum(dims);
        result.load(b.data());
      } else if (reductions == 2) {
        std::array<int, 2> dims;
        std::copy_n(axes.begin(), reductions, dims.begin());
        Tensor<To, 2, RowMajor> b = tensor4D.abs().sum(dims);
        result.load(b.data());
      } else if (reductions == 3) {
        std::array<int, 3> dims;
        std::copy_n(axes.begin(), reductions, dims.begin());
        Tensor<To, 1, RowMajor> b = tensor4D.abs().sum(dims);
        result.load(b.data());
      } else if (reductions == 4) {
        std::array<int, 4> dims;
        std::copy_n(axes.begin(), reductions, dims.begin());
        Tensor<To, 0, RowMajor> b = tensor4D.abs().sum(dims);
        result.load(b.data());
      }

      return result;

    } else if (rank == 3) {

      tensor<To> result(dimensions);
      DNNC_EIGEN_TENSOR_MAP(tensor, Ti, a);
      // tensor = tensor.abs();

      if (reductions == 0) {
        std::array<int, 3> dims = {0, 1, 2};
        Tensor<To, 0, RowMajor> b = tensor.abs().sum(dims);
        result.load(b.data());
      } else if (reductions == 1) {
        std::array<int, 1> dims;
        std::copy_n(axes.begin(), reductions, dims.begin());
        Tensor<To, 2, RowMajor> b = tensor.abs().sum(dims);
        result.load(b.data());
      } else if (reductions == 2) {
        std::array<int, 2> dims;
        std::copy_n(axes.begin(), reductions, dims.begin());
        Tensor<To, 1, RowMajor> b = tensor.abs().sum(dims);
        result.load(b.data());
      } else if (reductions == 3) {
        std::array<int, 3> dims;
        std::copy_n(axes.begin(), reductions, dims.begin());
        Tensor<To, 0, RowMajor> b = tensor.abs().sum(dims);
        result.load(b.data());
      }
      return result;
    }
    if (rank == 2) {
      DNNC_EIGEN_MATRIX(matrix, Ti, a);
      tensor<To> result(dimensions);

      if (reductions == 2 || reductions == 0) {
        Matrix<To, 1, Dynamic, RowMajor> colReduced =
            matrix.cwiseAbs().colwise().sum();
        Matrix<To, 1, RowMajor> fullReduced =
            colReduced.cwiseAbs().rowwise().sum();
        result.load(fullReduced.data());
      } else if (axes[0] == 0) {
        Matrix<To, 1, Dynamic, RowMajor> colReduced =
            matrix.cwiseAbs().colwise().sum();
        result.load(colReduced.data());
      } else if (axes[0] == 1) {
        Matrix<To, 1, Dynamic, RowMajor> rowReduced =
            matrix.cwiseAbs().rowwise().sum();
        result.load(rowReduced.data());
      }
      return result;
    }
    if (rank == 1) {
      DNNC_EIGEN_VECTOR(vector, Ti, a);
      tensor<To> result(dimensions);

      Matrix<To, 1, RowMajor> b = vector.cwiseAbs().rowwise().sum();
      result.load(b.data());

      return result;
    }

    return a;
    // CHANGE return-type and args
  }
};
} // namespace dnnc