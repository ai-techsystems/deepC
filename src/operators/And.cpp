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

#include "operators/And.h"

using namespace dnnc;
using namespace Eigen;

#ifdef DNNC_AND_TEST
#include <iostream>
int main() {
  int i1[6] = {1, 2, 3, 4, 5, 6};
  int i2[6] = {0, 2, 3, 4, 5, 6};
  float f1[6] = {1., 2., 3., 4., 5., 6.};
  float f2[6] = {0., 2., 3., 4., 5., 6.};
  double d1[6] = {1., 2., 3., 4., 5., 6.};
  double d2[6] = {0., 2., 3., 4., 5., 6.};
  bool b1[6] = {true, false, true, true, true, false};
  bool b2[6] = {false, true, true, true, false, true};

  tensor<bool> tb1(2, 3);
  tb1.load(b1);
  tensor<bool> tb2(2, 3);
  tb2.load(b2);
  tensor<int> ti1(2, 3);
  ti1.load(i1);
  tensor<int> ti2(2, 3);
  ti2.load(i2);
  tensor<float> tf1(2, 3);
  tf1.load(f1);
  tensor<float> tf2(2, 3);
  tf2.load(f2);
  tensor<double> td1(2, 3);
  td1.load(d1);
  tensor<double> td2(2, 3);
  td2.load(d2);

  And<bool, bool> ANDbb("localOpName");
  std::cout << ANDbb.compute(tb1, tb2) << std::endl;

  And<bool, int> ANDbi("localOpName");
  std::cout << ANDbi.compute(ti1, ti2) << std::endl;

  And<bool, float> ANDbf("localOpName");
  std::cout << ANDbf.compute(tf1, tf2) << std::endl;

  And<bool, double> ANDbd("localOpName");
  std::cout << ANDbd.compute(td1, td2) << std::endl;

  // And<double, float> AND3("localOpName");
  // std::cout << AND3.compute(a, b) << std::endl ;

  // And<bool, double> AND3("name4");
  // std::cout << AND3.compute(a, b) << std::endl ;

  return 0;
}
#endif
