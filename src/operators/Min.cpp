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

#include "operators/Min.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_MIN_TEST
#include <iostream>
int main() {
  float data1[12] = {0.521434,   0.57921106,  -2.0741816, -0.34119776,
                     -1.0926818, -0.11949139, 0.47781935, -0.32272545,
                     -2.029931,  -0.59571075, 0.00333933, -1.2404536};
  float data2[12] = {-0.72546995, -0.66292864, 0.03466121, 1.1469446,
                     0.62084216,  1.8464565,   2.229277,   0.6344861,
                     0.28158414,  -0.9978712,  0.10784209, -0.4692914};
  tensor<float> fTensor1(1, 12);
  fTensor1.load(data1);
  tensor<float> fTensor2(1, 12);
  fTensor2.load(data2);

  // std::cout << fTensor1 << std::endl;
  // std::cout << fTensor2 << std::endl;

  std::vector<tensor<float>> vt;
  vt.push_back(fTensor1);
  vt.push_back(fTensor2);

  Min<float> m("localOpName");
  auto result = m.compute(vt);

  std::cout << result << "\n";
}
#endif
