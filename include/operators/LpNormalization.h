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
// normalize(https://en.wikipedia.org/wiki/Norm_(mathematics))
// Eigen cwise unsupported-tensors(written TODO in original doc)
//

#pragma once
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {
/*! Given a matrix, apply Lp-normalization along the provided axis.*/
/*! The formula for Lp-norm is given by:
    \f$ \left \| x \right \|_{1} = \sum_{i=1}^{n}\left | x_{i} \right | \f$ */
/* \f$ \left \| x \right \|_{2} = \sum_{i=1}^{n}\sqrt{\left ( x_{i} \right
 * )^{2}} \f$ */

template <typename T> class LpNormalization : public baseOperator<T, T, T> {
  //  LpNormalization attributes
protected:
  int p = 2; /*!< p value of the Lp norm used to pool over the input data.Only
                L1 norm and L2 norm are supported */
  int axis = -1; /*!< axis to apply normalization.
                  * Since axis is int it can be 0 or 1(-1 indicates last axis
                  * i.e. 1). */
public:
  LpNormalization(std::string name = "opLpNormalization", int p = 2,
                  int axis = -1)
      : baseOperator<T, T, T>(opLpNormalization, name) {
    this->p = p;
    this->axis = axis;
  }

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_p) {
      obj = p;
      return true;
    } else if (attrName == attr_axis) {
      obj = axis;
      return true;
    }
    return false;
  }
  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_p) {
      p = obj;
      return true;
    } else if (attrName == attr_axis) {
      axis = obj;
      return true;
    }
    return false;
  }

  tensor<T> compute(tensor<T> &a /*!<[float,double]: 2D tensor*/) {
    if (!(this->template type_check<T, float, double>())) {
      SPDLOG_ERROR("Constrain input and output types to float tensors.");
      return NULL_TENSOR<T>;
    }
    if (a.rank() != 2) {
      SPDLOG_ERROR("Constrain input and output types should be matrix.");
      return NULL_TENSOR<T>;
    }
    if (p != 2 && p != 1) {
      SPDLOG_ERROR("Constrain input(norm) not supported.");
      return NULL_TENSOR<T>;
    }

    tensor<T> result(a.shape(), a.name());

    DNNC_EIGEN_MATRIX(eigenMatrixA, T, a);

    if (axis == 0 && p == 1) {
      int i, j;
      for (i = 0; i < int(a.shape()[1]); i++) {
        float sum = 0;
        for (j = 0; j < int(a.shape()[0]); j++) {
          sum += abs(eigenMatrixA(j, i));
        }

        for (j = 0; j < int(a.shape()[0]); j++) {
          result(j, i) = eigenMatrixA(j, i) / sum;
        }
      }
    }

    else if ((axis == 1 || axis == -1) && p == 1) {
      int i, j;
      for (i = 0; i < int(a.shape()[0]); i++) {
        float sum = 0;
        for (j = 0; j < int(a.shape()[1]); j++) {
          sum += abs(eigenMatrixA(i, j));
        }

        for (j = 0; j < int(a.shape()[1]); j++) {
          result(i, j) = eigenMatrixA(i, j) / sum;
        }
      }
    }

    else if (axis == 0 && p == 2) {
      int i, j;
      for (i = 0; i < int(a.shape()[1]); i++) {
        float sum = 0;
        for (j = 0; j < int(a.shape()[0]); j++) {
          sum += (eigenMatrixA(j, i) * eigenMatrixA(j, i));
        }
        for (j = 0; j < int(a.shape()[0]); j++) {
          result(j, i) = eigenMatrixA(j, i) / sqrt(sum);
        }
      }
    }

    // default cases
    else if ((axis == 1 || axis == -1) && p == 2) {
      int i, j;
      for (i = 0; i < int(a.shape()[0]); i++) {
        float sum = 0;
        for (j = 0; j < int(a.shape()[1]); j++) {
          sum += (eigenMatrixA(i, j) * eigenMatrixA(i, j));
        }
        for (j = 0; j < int(a.shape()[1]); j++) {
          result(i, j) = eigenMatrixA(i, j) / sqrt(sum);
        }
      }
    }
    return result;
  }
  /*!<
 \return The output matrix after normalization.
 */
};
} // namespace dnnc
