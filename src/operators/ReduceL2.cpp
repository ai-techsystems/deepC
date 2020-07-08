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

#include "operators/ReduceL2.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_REDUCEL2_TEST
#include <iostream>
int main() {
  float d1[6] = {1, 2, 3, 6, 5, 4};
  float d2[12] = {1., 4., 3., 9., 5., 6., 1., 5., 3., 6., 7., 4.};
  float d3[16] = {1., 4., 3., 9., 5., 6., 1., 5.,
                  3., 6., 7., 4., 3., 8., 4., 5.};

  tensor<float> a({2, 2, 3});
  a.load(d2);

  tensor<float> b({12});
  b.load(d2);

  tensor<float> c({2, 6});
  c.load(d2);

  tensor<float> d({2, 2, 2, 2});
  d.load(d3);

  std::vector<int> axes{0};

  ReduceL2<float, float> m("localOpName");

  m.setAttribute(attr_axis, axes);
  m.setAttribute(attr_keepdims, 0);

  auto result = m.compute(c);

  std::cout << result << std::endl;
  // ADD YOUR TEST CODE HERE
}
#endif
