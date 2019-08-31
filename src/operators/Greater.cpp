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
#include "operators/Greater.h"

using namespace dnnc;
using namespace Eigen;

//#define DNNC_ADD_TEST 1
#ifdef DNNC_GREATER_TEST
#include <iostream>

int main() {
  int d1[6] = {6, 2, 4, 4, 3, 6};
  int d2[6] = {1, 2, 3, 4, 5, 6};
  tensor<int> a(2, 1, 3);
  a.load(d1);
  tensor<int> b(2, 1, 3);
  b.load(d2);
  std::cout << a;
  std::cout << b;
  Greater<int> m("localOpName");
  auto result = m.compute(a, b);
  std::cout << result;
  std::cout << "\n";

  return 0;
}

#endif
