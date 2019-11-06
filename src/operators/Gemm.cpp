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

#include "operators/Gemm.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_GEMM_TEST
#include <iostream>
int main() {
  float d1[6] = {1., 2., 3., 4., 5., 6.};
  float d2[6] = {6., 1., 5., 10., 2., 4.};
  float d3[9] = {6., 1., 5., 10., 2., 4., 1., 2., 3.};
  tensor<float> a({2, 3});
  a.load(d1);
  tensor<float> b({3, 2});
  b.load(d2);
  tensor<float> c({3, 3});
  c.load(d3);
  float alpha = 2.2;
  float beta = 1.8;
  int transA = 1;
  int transB = 1;

  Gemm<float, float, float> m("localOpName", alpha, beta, transA, transB);

  auto result = m.compute(a, b, c);

  std::cout << result;
  std::cout << "\n";

  return 0;
}
#endif
