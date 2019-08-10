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
#include "core/tensor.h"
#include "operators/Add.h"
#include "operators/MatMul.h"

using namespace dnnc;

tensor<float> make_tensor(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
  return tensor<float>(x, y, z, w);
}

tensor<float> multiply(tensor<float> &a, tensor<float> &b) {
  MatMul<float> op;
  return op.compute(a, b);
}

tensor<float> add(tensor<float> &a, tensor<float> &b) {
  Add<float> op;
  return op.compute(a, b);
}