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

#include "operators/Not.h"

using namespace dnnc;
using namespace Eigen;

#define DNNC_NOT_TEST 1
#ifdef DNNC_NOT_TEST
#include <iostream>

int main() {
  std::cout << "Tests for different dimensions" << std::endl;
  
  bool a[8] = {true, false, false, true, false, true};
  // bool b[8] = {1, 0, 0, 1, 0, 1};

  tensor<bool> tensor_a(8);
  tensor_a.load(a);
  std::cout << "Original Vector : " << tensor_a << std::endl;
  std::cout << "Original Vector Shape: 8, 1" << std::endl;
  std::cout << "------------------------------------------" << std::endl;


  tensor<bool> tensor_b(2, 4);
  tensor_b.load(a);

  tensor<bool> tensor_c(2, 2, 2);
  tensor_c.load(a);

  Not<bool> n("localOpName");
  auto result = n.compute(tensor_a);

  std::cout << "Changed Vector - 8, 1" << std::endl;
  std::cout << result;
  std::cout << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  result = n.compute(tensor_b);

  std::cout << "Changed Vector - 2, 4" << std::endl;
  std::cout << result;
  std::cout << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  result = n.compute(tensor_c);

  std::cout << "Changed Vector - 2, 2, 2" << std::endl;
  std::cout << result;
  std::cout << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  std::cout << "Tests for different types" << std::endl;
  Not<int> x("localOpName");
  Not<float> y("localOpName");
  Not<double> z("localOpName");


}
#endif
