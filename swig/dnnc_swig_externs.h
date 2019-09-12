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
//	 http://www.apache.org/licenses/LICENSE-2.0
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

namespace dnnc {
	extern tensor<float> \
		 array(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
	extern tensor<float> \
		 empty(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
	extern tensor<float> \
		 zeros(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
	extern tensor<float> \
		 ones(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
	extern tensor<float> \
		 random(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
	extern tensor<float> \
		 array(PyObject *objects) ;
	extern tensor<float> \
		 arange(size_t stop, size_t start = 0, size_t step = 1) ;
	extern tensor<double> \
		 reshape(tensor<double> &tens_obj, PyObject *newShape) ;
	extern tensor<float> \
		 reshape(tensor<float> &tens_obj, PyObject *newShape) ;
	extern tensor<int> \
		 reshape(tensor<int> &tens_obj, PyObject *newShape) ;
	extern tensor<bool> \
		 reshape(tensor<bool> &tens_obj, PyObject *newShape) ;
	extern tensor<double> \
		 matmul(tensor<double> &a, tensor<double> &b) ;
	extern tensor<float> \
		 matmul(tensor<float> &a, tensor<float> &b) ;
	extern tensor<int> \
		 matmul(tensor<int> &a, tensor<int> &b) ;
	extern tensor<double> \
		 add(tensor<double> &a, tensor<double> &b) ;
	extern tensor<float> \
		 add(tensor<float> &a, tensor<float> &b) ;
	extern tensor<int> \
		 add(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		 logical_and(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		 logical_and(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		 logical_and(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		 logical_and(tensor<int> &a, tensor<int> &b) ;
	extern tensor<float> \
		 dequantize_linear(tensor<int> &a, tensor<float> &b, tensor<int> &c) ;
	extern tensor<float> \
		 div(tensor<double> &a, tensor<double> &b) ;
	extern tensor<float> \
		 div(tensor<float> &a, tensor<float> &b) ;
	extern tensor<float> \
		 div(tensor<int> &a, tensor<int> &b) ;
	extern tensor<int> \
		 floor_div(tensor<double> &a, tensor<double> &b) ;
	extern tensor<int> \
		 floor_div(tensor<float> &a, tensor<float> &b) ;
	extern tensor<int> \
		 floor_div(tensor<int> &a, tensor<int> &b) ;
	extern tensor<double> \
		 elu(tensor<double> &a, float alpha = 1.0) ;
	extern tensor<float> \
		 elu(tensor<float> &a, float alpha = 1.0) ;
	extern tensor<bool> \
		 equal(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		 equal(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		 equal(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		 equal(tensor<int> &a, tensor<int> &b) ;
	extern tensor<double> \
		 erf(tensor<double> &a) ;
	extern tensor<float> \
		 erf(tensor<float> &a) ;
	extern tensor<double> \
		 exp(tensor<double> &a) ;
	extern tensor<float> \
		 exp(tensor<float> &a) ;
	extern tensor<double> \
		 eye_like(tensor<double> &a, int k = 0) ;
	extern tensor<float> \
		 eye_like(tensor<float> &a, int k = 0) ;
	extern tensor<int> \
		 eye_like(tensor<int> &a, int k = 0) ;
	extern tensor<bool> \
		 eye_like(tensor<bool> &a, int k = 0) ;
	extern tensor<double> \
		 flatten(tensor<double> &a, int axis = 1) ;
	extern tensor<float> \
		 flatten(tensor<float> &a, int axis = 1) ;
	extern tensor<int> \
		 flatten(tensor<int> &a, int axis = 1) ;
	extern tensor<bool> \
		 flatten(tensor<bool> &a, int axis = 1) ;
	extern tensor<double> \
		 floor(tensor<double> &a) ;
	extern tensor<float> \
		 floor(tensor<float> &a) ;
	extern tensor<double> \
		 gemm(tensor<double> &a, tensor<double> &b, tensor<double> &c, float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0) ;
	extern tensor<float> \
		 gemm(tensor<float> &a, tensor<float> &b, tensor<float> &c, float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0) ;
	extern tensor<float> \
		 global_average_pool(tensor<float> &a) ;
	extern tensor<double> \
		 global_average_pool(tensor<double> &a) ;
	extern tensor<float> \
		 global_lp_pool(tensor<float> &a, int p = 2) ;
	extern tensor<double> \
		 global_lp_pool(tensor<double> &a, int p = 2) ;
	extern tensor<float> \
		 global_max_pool(tensor<float> &a) ;
	extern tensor<double> \
		 global_max_pool(tensor<double> &a) ;
	extern tensor<bool> \
		 greater(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		 greater(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		 greater(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		 greater_equal(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		 greater_equal(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		 greater_equal(tensor<double> &a, tensor<double> &b) ;
	extern tensor<float> \
		 hardmax(tensor<float> &a, int axis = 0) ;
	extern tensor<double> \
		 hardmax(tensor<double> &a, int axis = 0) ;
	extern tensor<float> \
		 hardsigmoid(tensor<float> &a, float alpha = 0.2,float beta = 0.5) ;
	extern tensor<double> \
		 hardsigmoid(tensor<double> &a, float alpha = 0.2,float beta = 0.5) ;
	extern tensor<bool> \
		 identity(tensor<bool> &a) ;
	extern tensor<int> \
		 identity(tensor<int> &a) ;
	extern tensor<float> \
		 identity(tensor<float> &a) ;
	extern tensor<double> \
		 identity(tensor<double> &a) ;
	extern tensor<float> \
		 instancenormalization(tensor<float> &a, tensor<float> &scale,tensor<float> &B, float epsilon = 1e-5) ;
	extern tensor<double> \
		 instancenormalization(tensor<double> &a, tensor<double> &scale,tensor<double> &B, float epsilon = 1e-5) ;
	extern tensor<bool> \
		 isinf(tensor<float> &a, int detect_positive = 1,int detect_negative = 1) ;
	extern tensor<bool> \
		 isinf(tensor<double> &a, int detect_positive = 1,int detect_negative = 1) ;
	extern tensor<bool> \
		 isnan(tensor<float> &a) ;
	extern tensor<bool> \
		 isnan(tensor<double> &a) ;
	extern tensor<float> \
		 lrn(tensor<float> &a, int size, float alpha = 0.0001,float beta = 0.75, float bias = 1.0) ;
	extern tensor<double> \
		 lrn(tensor<double> &a, int size, float alpha = 0.0001,float beta = 0.75, float bias = 1.0) ;
	extern tensor<float> \
		 leakyrelu(tensor<float> &a, float alpha = 0.01) ;
	extern tensor<double> \
		 leakyrelu(tensor<double> &a, float alpha = 0.01) ;
	extern tensor<bool> \
		 less(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		 less(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		 less(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		 less_equal(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		 less_equal(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		 less_equal(tensor<double> &a, tensor<double> &b) ;
	extern tensor<float> \
		 thresholded_relu(tensor<float> &a) ;
	extern tensor<float> \
		 log(tensor<float> &a) ;
	extern tensor<float> \
		 logsoftmax(tensor<float> &a) ;
	extern tensor<float> \
		 lpnormalization(tensor<float> &a) ;
	extern tensor<int> \
		 matmulinteger(tensor<int> &a, tensor<int> &b) ;
	extern tensor<float> \
		 max(std::vector<tensor<float>> a) ;
	extern tensor<int> \
		 mul(tensor<int> &a, tensor<int> &b) ;
	extern tensor<float> \
		 mul(tensor<float> &a, tensor<float> &b) ;
	extern tensor<double> \
		 mul(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		 not_equal(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		 not_equal(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		 not_equal(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		 not_equal(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		 logical_not(tensor<double> &a) ;
	extern tensor<bool> \
		 logical_not(tensor<float> &a) ;
	extern tensor<bool> \
		 logical_not(tensor<bool> &a) ;
	extern tensor<bool> \
		 logical_not(tensor<int> &a) ;
	extern tensor<bool> \
		 logical_or(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		 logical_or(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		 logical_or(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		 logical_or(tensor<int> &a, tensor<int> &b) ;
	extern tensor<float> \
		 pow(tensor<float> &a, tensor<float> &b) ;
	extern tensor<double> \
		 pow(tensor<double> &a, tensor<double> &b) ;
	extern tensor<double> \
		 sub(tensor<double> &a, tensor<double> &b) ;
	extern tensor<float> \
		 sub(tensor<float> &a, tensor<float> &b) ;
	extern tensor<int> \
		 sub(tensor<int> &a, tensor<int> &b) ;
	extern tensor<float> \
		 tan(tensor<float> &a) ;
	extern tensor<double> \
		 tan(tensor<double> &a) ;
	extern tensor<float> \
		 tanh(tensor<float> &a) ;
	extern tensor<double> \
		 tanh(tensor<double> &a) ;
	extern tensor<bool> \
		 logical_xor(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		 logical_xor(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		 logical_xor(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		 logical_xor(tensor<int> &a, tensor<int> &b) ;
	extern tensor<double> \
		 transpose(tensor<double> &a) ;
	extern tensor<float> \
		 transpose(tensor<float> &a) ;
	extern tensor<int> \
		 transpose(tensor<int> &a) ;
	extern tensor<bool> \
		 transpose(tensor<bool> &a) ;

}
