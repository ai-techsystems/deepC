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

#include "operators/BitShift.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_BITSHIFT_TEST
#include <iostream>
int main() {

  int d1[8] = {0, -1, 2, 3, 4, 5, 6, 7};
  int d2[8] = {0, -1, 2, 3, 4, 5, 6, 7};
  tensor<int> a({2, 4, 1});
  tensor<int> b({2, 4, 1});
  a.load(d1);
  b.load(d2);
  // std::string direction = "LEFT";
  std::string direction = "RIGHT";
  BitShift<int> m("localOpName", direction);
  // std::cout << a << "\n";
  auto result = m.compute(a, b);

  std::cout << result;
  std::cout << "\n";
}
#endif
