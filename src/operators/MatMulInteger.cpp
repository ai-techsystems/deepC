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

#include "operators/MatMulInteger.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_MATMULINTEGER_TEST
#include <iostream>
int main() {
  int d1[6] = {1, 2, 3, 4, 5, 6};
  int d2[6] = {1, 2, 3, 4, 5, 6};
  int d3[1] = {0};
  int d4[1] = {0};
  tensor<int> a(2, 3);
  a.load(d1);
  tensor<int> b(3, 2);
  b.load(d2);
  tensor<int> c(1);
  c.load(d3);
  tensor<int> d(1);
  d.load(d4);

  MatMulInteger<int> m("localOpName");
  auto result = m.compute(a, b, c, d);

  std::cout << result;
  std::cout << "\n";

#define DNNC_HIGHRANK_SUPPORT
  int i1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int i2[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int i3[1] = {0};
  int i4[1] = {0};

  dnnc::tensor<int> e(2, 2, 2);
  e.load(i1);
  dnnc::tensor<int> f(2, 2, 2);
  f.load(i2);

  dnnc::tensor<int> g(1);
  g.load(i3);
  dnnc::tensor<int> h(1);
  h.load(i4);

  MatMulInteger<int> m1("localint");
  dnnc::tensor<int> iresult = m1.compute(e, f, g, h);

  std::cout << iresult << "\n";

  return 0;
}

#endif
