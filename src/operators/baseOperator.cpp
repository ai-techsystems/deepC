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
#include "operators/baseOperator.h"

//#define DNNC_OPERATOR_TEST 1
#ifdef DNNC_OPERATOR_TEST
#include <iostream>

namespace dnnc {
template <typename T> class fakeOperatorTest : public baseOperator<T> {
public:
  fakeOperatorTest() : baseOperator<T>(opAbs) {}
  void testEigenMatrix(tensor<T> &t) {
    if (t.rank() == 1) {
      DNNC_EIGEN_VECTOR(eigenVector, t);
      std::cout << eigenVector << "\n";
    } else if (t.rank() == 2) {
      DNNC_EIGEN_MATRIX(eigenMatrix, t);
      std::cout << eigenMatrix << "\n";
    } else if (t.rank() == 3) {
      DNNC_EIGEN_TENSOR(eigenTensor, t);
      // std::cout << eigenTensor << "\n";
    } else if (t.rank() == 4) {
      DNNC_EIGEN_TENSOR4D(eigenTensor4D, t);
      // std::cout << eigenTensor4D << "\n";
    }

    return;
  }
};
} // namespace dnnc

using namespace dnnc;

int main() {
  tensor<float> tf({3, 4});
  fakeOperatorTest<float> fotf;
  fotf.testEigenMatrix(tf);

  tensor<double> td({3, 4});
  fakeOperatorTest<double> fotd;
  fotd.testEigenMatrix(td);

  tensor<int> ti({3, 4});
  fakeOperatorTest<int> foti;
  foti.testEigenMatrix(ti);

  tensor<float> tf1({2, 3, 4, 5});
  fakeOperatorTest<float> fotf1;
  fotf1.testEigenMatrix(tf1);

  tensor<double> td1({3, 4, 6, 7});
  fakeOperatorTest<double> fotd1;
  fotd1.testEigenMatrix(td1);

  return 0;
}
#endif
