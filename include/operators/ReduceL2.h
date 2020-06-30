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
template <typename T> class ReduceL2 : public baseOperator<T, T, T> {
  //  ReduceL2 attributes
protected:
    
    std::vector<int> axes;
    int keepdims = 1;

public:
  ReduceL2(std::string name = "opReduceL2", std::vector<int> axes = {}, int keepdims = 1)
      : baseOperator<T, T, T>(opReduceL2, name) {
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
    if (attrName == attr_axes) {
      obj = axes;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, std::vector<int> obj) override {
    if (attrName == attr_axes) {
      axes = obj;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> a /*!< : N D tensor input*/) override {

    //Todo: return array with dimension depending on keepdims

    int rank = a.rank();
    int reductions = axes.size();

    std::vector<int> arr(rank, 0);
    for (int i = 0; i < axes.size(); i++) {
      if (axes.at(i) < 0 || axes.at(i) >= rank || arr.at(axes.at(i)) >= 1) {
        //TODO: decide what to return
        SPDLOG_ERROR("Inputted axes not appropriate for Reduce operator.");
        return NULL_TENSOR<T>;
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
        } else if(keepdims) {
          dimensions.push_back(1);
        }
    }

    if (dimensions.size() == 0) {
      dimensions.push_back(1);
    }

    //CANT RETURN A NUMBER SO ARRAY OF DIMENSION 1 IS MINIMUM

    std::cout << dimensions.size() << std::endl;
    
    //TODO: decide what to return
    if (rank < reductions){
      SPDLOG_ERROR("tensor dimenions not appropriate for Reduce operator.");
      return NULL_TENSOR<T>;
    } 
   
    if (rank == 4) {

        tensor<T> result(dimensions);

        DNNC_EIGEN_TENSOR4D_MAP(tensor4D, T, a);
        tensor4D = tensor4D.abs().square();

        if (reductions == 0) { 
          array<int, 4> dims = {0, 1, 2, 3};
          Tensor<T, 0, RowMajor> b = tensor4D.sum(dims);
          result.load(b.data());
        } else if (reductions == 1) {
          array<int, 1> dims;
          std::copy_n(axes.begin(), reductions, dims.begin());      
          Tensor<T, 3, RowMajor> b = tensor4D.sum(dims);
          result.load(b.data());
        } else if (reductions == 2) {
          array<int, 2> dims;
          std::copy_n(axes.begin(), reductions, dims.begin());      
          Tensor<T, 2, RowMajor> b = tensor4D.sum(dims);
          result.load(b.data());
        } else if (reductions == 3) {
          array<int, 3> dims;
          std::copy_n(axes.begin(), reductions, dims.begin());      
          Tensor<T, 1, RowMajor> b = tensor4D.sum(dims);
          result.load(b.data());
        } else if (reductions == 4) {
          array<int, 4> dims;
          std::copy_n(axes.begin(), reductions, dims.begin());   
          Tensor<T, 0, RowMajor> b = tensor4D.sum(dims);
          result.load(b.data());
        }

        return result;

    } else if (rank == 3 ) {      

        tensor<T> result(dimensions);
        DNNC_EIGEN_TENSOR_MAP(tensor, T, a);
        tensor = tensor.abs().square();

        if (reductions == 0) { 
          array<int, 3> dims = {0, 1, 2};
          Tensor<T, 0, RowMajor> b = tensor.sum(dims);
          result.load(b.data());          
        } else if (reductions == 1) {
          array<int, 1> dims;
          std::copy_n(axes.begin(), reductions, dims.begin());      
          Tensor<T, 2, RowMajor> b = tensor.sum(dims);
          result.load(b.data());
        } else if (reductions == 2) {
          array<int, 2> dims;
          std::copy_n(axes.begin(), reductions, dims.begin());      
          Tensor<T, 1, RowMajor> b = tensor.sum(dims);
          result.load(b.data());
        } else if (reductions == 3) {
          array<int, 3> dims;
          std::copy_n(axes.begin(), reductions, dims.begin());   
          Tensor<T, 0, RowMajor> b = tensor.sum(dims);
          result.load(b.data());
        }
        return result;

    } if (rank == 2) {
        DNNC_EIGEN_MATRIX(matrix, T, a);
        matrix = matrix.array().square().matrix();
        tensor<T> result(dimensions);

        if (reductions == 2 || reductions == 0) {
          Matrix<T, 1, Dynamic, RowMajor> colReduced = matrix.cwiseAbs().colwise().sum();
          Matrix<T, 1, RowMajor> fullReduced = colReduced.cwiseAbs().rowwise().sum();
          result.load(fullReduced.data());
        } else if (axes[0] == 0) {
          Matrix<T, 1, Dynamic, RowMajor> colReduced = matrix.cwiseAbs().colwise().sum();
          result.load(colReduced.data());
        } else if (axes[0] == 1) {
          Matrix<T, 1, Dynamic, RowMajor> rowReduced = matrix.cwiseAbs().rowwise().sum();
          result.load(rowReduced.data());
        }
        return result;

    } if (rank == 1) {
        DNNC_EIGEN_VECTOR(vector, T, a);
        vector = vector.array().square().matrix();
        tensor<T> result(dimensions);

        Matrix<T, 1, RowMajor> b = vector.cwiseAbs().rowwise().sum();
        result.load(b.data());

        return result;
    } 
    
    return a;
    // CHANGE return-type and args

  }
};
}