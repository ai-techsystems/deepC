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

#include "operators/Mean.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_MEAN_TEST
#include <iostream>
int main() {
  float data1[9] = {0.1, 0.02, 1.3, 4.05, 0.5, 0.06, 1.2, 1.03, 1.4};
  float data2[9] = {0.01, 0.2, 1.03, 4.5, 0.05, 0.6, 1.02, 1.3, 1.4};
  float data3[9] = {0.12, 0.032, 1.34, 4.056, 0.25, 0.062, 1.82, 2.03, 3.4};
  tensor<float> fTensor1(3, 3);
  fTensor1.load(data1);
  tensor<float> fTensor2(3, 3);
  fTensor2.load(data2);
  tensor<float> fTensor3(3, 3);
  fTensor3.load(data3);

  std::vector<tensor<float>> vt;
  vt.push_back(fTensor1);
  vt.push_back(fTensor2);
  vt.push_back(fTensor3);

  Mean<float> m("localOpName");
  auto result = m.compute(vt);

  std::cout << result << "\n";
}
#endif
