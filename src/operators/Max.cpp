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

#include "operators/Max.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_MAX_TEST
#include <iostream>
int main() {
  float data1[6] = {1, 1, 1, 2,3, 1};
  float data2[4] = {1, 2,3, 4};
  tensor<float> fTensor1(3, 2);
  fTensor1.load(data1);
  tensor<float> fTensor2(2,2);
  fTensor2.load(data2);

 // std::cout<<fTensor1<<"\n";
  //std::cout<<"\n";
  //std::cout<<fTensor2<<"\n";
  //std::cout<<"\n";

  std::vector<tensor<float>> vt;
  vt.push_back(fTensor1);
  vt.push_back(fTensor2);

  Max<float> m("localOpName");
  auto result = m.compute(vt);

  std::cout << result << "\n";
}
#endif
