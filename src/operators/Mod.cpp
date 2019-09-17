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

#include "operators/Mod.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_MOD_TEST
#include <iostream>
#include <math.h>

int main() {

  float d1[4] = {21., 22., 23., 24.};
  float d2[4] = {20., 20., 20., 20.};
  tensor<float> a(4);
  a.load(d1);
  tensor<float> b(4);
  b.load(d1);
  int fmod_flag = 1;
  // std::cout << fmod(21.0, 21.0);

  Mod<float> m("localOpName", fmod_flag);
  auto result = m.compute(a, b);

  std::cout << result;
  std::cout << "\n";

  int d1_int[4] = {21, 22, 23, 24};
  int d2_int[4] = {20};
  tensor<int> a_int(4);
  a_int.load(d1_int);
  tensor<int> b_int(1);
  b_int.load(d2_int);
  // int fmod_flag = 0;

  Mod<int> m_int("localOpName");
  auto result_int = m_int.compute(a_int, b_int);

  std::cout << result_int;
  std::cout << "\n";

  return 0;
}
#endif
