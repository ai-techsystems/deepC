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

#include "operators/Conv.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_CONV_TEST
#include <iostream>
int main() {
  // ADD YOUR TEST CODE HERE
  tensor<float> X({1,1,3,3});
  X.load({2,2,2,2,2,2,2,2,2});
  tensor<float> W({1,1,2,2});
  W.load({1,1,1,1});
  tensor<float> B({1});
  B.load({0});
  std::string auto_pad = "VALID";
  std::vector<int> dilations = std::vector<int>(); 
  int group = 1;
  std::vector<int> kernal_shape = std::vector<int>();
  std::vector<int> pads = std::vector<int>();
  std::vector<int> strides = std::vector<int>();

  Conv<float, float, float> op("opConv", auto_pad, dilations, group,
                               kernal_shape, pads, strides);
  tensor<float> result = op.compute(X, W, B);
  return 0;
}
#endif
