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

#if defined(ARDUINO)
#include "Eigen.h"
#else
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#endif

using namespace Eigen;

// These macros are used to create a EigenMap object on
// the top of memory allocated in class tensor.
// They help in keeping the memory footprint small.

// check out Eigen documentation
// "Interfacing with raw buffers: the Map class"
// https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html

// NOTE: These macros are valid only on classes inherited
// from baseOperator.
#define DNNC_EIGEN_ARRAY_MAP(var, T, t)                                        \
  Map<Matrix<T, 1, Dynamic, RowMajor>> var(this->tensorMem(t), t.length());

#define DNNC_EIGEN_VECTOR_CTOR(T) Matrix<T, 1, Dynamic, RowMajor>

#define DNNC_EIGEN_MATRIX_CTOR(T) Matrix<T, Dynamic, Dynamic, RowMajor>

#define DNNC_EIGEN_VECTOR(var, T, t)                                           \
  Map<Matrix<T, 1, Dynamic, RowMajor>> var(this->tensorMem(t), t.shape()[0]);

#define DNNC_EIGEN_MATRIX(var, T, t)                                           \
  Map<Matrix<T, Dynamic, Dynamic, RowMajor>> var(this->tensorMem(t),           \
                                                 t.shape()[0], t.shape()[1]);

#define DNNC_EIGEN_TENSOR Tensor<T, 3, RowMajor>

#define DNNC_EIGEN_TENSOR_MAP(var, T, t)                                       \
  TensorMap<DNNC_EIGEN_TENSOR> var(this->tensorMem(t), t.shape()[0],           \
                                   t.shape()[1], t.shape()[2]);

#define DNNC_EIGEN_TENSOR4D Tensor<T, 4, RowMajor>

#define DNNC_EIGEN_TENSOR4D_MAP(var, T, t)                                     \
  TensorMap<DNNC_EIGEN_TENSOR4D> var(this->tensorMem(t), t.shape()[0],         \
                                     t.shape()[1], t.shape()[2],               \
                                     t.shape()[3]);
