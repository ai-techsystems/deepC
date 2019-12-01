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

#include "operators/ReduceSum.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_REDUCESUM_TEST
#include <iostream>
int main() {
  float f1[6] = {1., 2., 3., 4., 5., 6};
  float f2[9] = {6., 1., 0., 5., 7., 4., 2., 3., 8};

  tensor<float> tf1({2, 3});
  tf1.load(f1);
  tensor<float> tf2({9});
  tf2.load(f2);

  std::vector<int> axes({0});

  ReduceSum<int, float> sum("floatsum");

  sum.setAttribute(attr_axis, axes);
  sum.setAttribute(attr_keepdims, 0);

  std::cout << sum.compute(tf1) << std::endl;
  std::cout << sum.compute(tf2) << std::endl;
}
#endif
