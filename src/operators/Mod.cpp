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

// template <typename T>
// void mod_test(T* a, T* b) {

//   tensor<T> tensor_a(4);
//   tensor_a.load(a);

//   tensor<T> tensor_b(1);
//   tensor_b.load(b);

//   // Mod<T> m("localOpName");
//   // if (typeid(a) != typeid(float))
//   std::cout << "Typeid = " << typeid(a).name() << "\n" << std::endl;
//   Mod<T> m("localOpName", 1);
//   auto result = m.compute(tensor_a, tensor_b);

//   std::cout << result << std::endl;
// }

int main() {

  // int a[4] = {-21, -22, -23, -24};
  // int b[1] = {20};
  // mod_test(a, b);

  float d1[4] = {21., 22., 23., 24.};
  float d2[4] = {20., 21., 22., 23.};
  tensor<float> a(4);
  a.load(d1);
  tensor<float> b(4);
  a.load(d2);
  // mod_test(c, d);
  int fmod_flag = 1;

  Mod<float> m("localOpName", fmod_flag);
  auto result = m.compute(a, b);

  std::cout << result;
  std::cout << "\n";

  return 0;
  // double e[4] = {21., 22., 23., 24.};
  // double f[1] = {20};
  // mod_test(e, f);
}
#endif
