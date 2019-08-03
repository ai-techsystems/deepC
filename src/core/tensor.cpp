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
#include "core/tensor.h"
#include <iostream>
#include <stdint.h>
#include <float.h>

//#define DNNC_TENSOR_TEST 1
#ifdef DNNC_TENSOR_TEST
using namespace dnnc;

template <typename T>
void print_tensor(tensor<T>& v) {
	for (size_t i = 0; i < v.size(); i++) 
		std::cout << v[i] << "@" << i << " ";
	std::cout << "\n";
}

template <typename T>
void type_test()
{
	size_t x1=2, x2=3, x3=4, x4=5;
	tensor<T> t1(x1, x2);

	size_t sz = x1*x2;
	for (size_t i = 0; i < sz; i++)
		t1[i] = static_cast<T>(i + sz);

	std::cout << "size " << t1.size() << std::endl;
	const std::vector<DIMENSION> shape = t1.shape();
	std::cout << "shape (";
	for (size_t i = 0; i < shape.size(); i++)
		std::cout << shape[i] << ", ";
	std::cout << ")\n";

	std::cout << t1[0] << std::endl; // print first element
	std::cout << t1(1, 2, 3, 4) << std::endl; // print last element
	print_tensor(t1);
}

int main()
{
	std::cout << "short------\n"; type_test<short>();
	std::cout << "int8_t------\n"; type_test <int8_t>();
	std::cout << "int16_t------\n"; type_test <int16_t>();
	std::cout << "int32_t------\n"; type_test <int32_t>();
	std::cout << "int64_t------\n"; type_test <int64_t>();
	std::cout << "float------\n"; type_test<float>();
	std::cout << "double------\n"; type_test<double>();
	// need tests for float11_t, float16_t and float64_t
	return 0;
}
#endif
