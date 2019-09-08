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

extern dnnc::tensor<float> \
		 array(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
extern dnnc::tensor<float> \
		 empty(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
extern dnnc::tensor<float> \
		 zeros(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
extern dnnc::tensor<float> \
		 ones(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
extern dnnc::tensor<float> \
		 random(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) ;
extern dnnc::tensor<float> \
		 array(PyObject *objects) ;
extern dnnc::tensor<float> \
		 arange(size_t stop, size_t start = 0, size_t step = 1) ;
extern dnnc::tensor<double> \
		 reshape(dnnc::tensor<double> &tens_obj, PyObject *newShape) ;
extern dnnc::tensor<float> \
		 reshape(dnnc::tensor<float> &tens_obj, PyObject *newShape) ;
extern dnnc::tensor<int> \
		 reshape(dnnc::tensor<int> &tens_obj, PyObject *newShape) ;
extern dnnc::tensor<bool> \
		 reshape(dnnc::tensor<bool> &tens_obj, PyObject *newShape) ;
extern dnnc::tensor<double> \
		 matmul(dnnc::tensor<double> &a, dnnc::tensor<double> &b) ;
extern dnnc::tensor<float> \
		 matmul(dnnc::tensor<float> &a, dnnc::tensor<float> &b) ;
extern dnnc::tensor<int> \
		 matmul(dnnc::tensor<int> &a, dnnc::tensor<int> &b) ;
extern dnnc::tensor<double> \
		 add(dnnc::tensor<double> &a, dnnc::tensor<double> &b) ;
extern dnnc::tensor<float> \
		 add(dnnc::tensor<float> &a, dnnc::tensor<float> &b) ;
extern dnnc::tensor<int> \
		 add(dnnc::tensor<int> &a, dnnc::tensor<int> &b) ;
extern dnnc::tensor<float> \
		 dequantize_linear(dnnc::tensor<float> &a, dnnc::tensor<float> &b, dnnc::tensor<float> &c) ;
extern dnnc::tensor<double> \
		 div(dnnc::tensor<double> &a, dnnc::tensor<double> &b) ;
extern dnnc::tensor<float> \
		 div(dnnc::tensor<float> &a, dnnc::tensor<float> &b) ;
extern dnnc::tensor<int> \
		 div(dnnc::tensor<int> &a, dnnc::tensor<int> &b) ;
extern dnnc::tensor<double> \
		 elu(dnnc::tensor<double> &a, float alpha = 1.0) ;
extern dnnc::tensor<float> \
		 elu(dnnc::tensor<float> &a, float alpha = 1.0) ;
extern dnnc::tensor<bool> \
		 equal(dnnc::tensor<double> &a, dnnc::tensor<double> &b) ;
extern dnnc::tensor<bool> \
		 equal(dnnc::tensor<float> &a, dnnc::tensor<float> &b) ;
extern dnnc::tensor<bool> \
		 equal(dnnc::tensor<bool> &a, dnnc::tensor<bool> &b) ;
extern dnnc::tensor<bool> \
		 equal(dnnc::tensor<int> &a, dnnc::tensor<int> &b) ;
extern dnnc::tensor<double> \
		 erf(dnnc::tensor<double> &a) ;
extern dnnc::tensor<float> \
		 erf(dnnc::tensor<float> &a) ;
extern dnnc::tensor<double> \
		 exp(dnnc::tensor<double> &a) ;
extern dnnc::tensor<float> \
		 exp(dnnc::tensor<float> &a) ;
extern dnnc::tensor<double> \
		 eye_like(dnnc::tensor<double> &a, int k = 0) ;
extern dnnc::tensor<float> \
		 eye_like(dnnc::tensor<float> &a, int k = 0) ;
extern dnnc::tensor<int> \
		 eye_like(dnnc::tensor<int> &a, int k = 0) ;
extern dnnc::tensor<bool> \
		 eye_like(dnnc::tensor<bool> &a, int k = 0) ;
extern dnnc::tensor<double> \
		 flatten(dnnc::tensor<double> &a, int axis = 1) ;
extern dnnc::tensor<float> \
		 flatten(dnnc::tensor<float> &a, int axis = 1) ;
extern dnnc::tensor<int> \
		 flatten(dnnc::tensor<int> &a, int axis = 1) ;
extern dnnc::tensor<bool> \
		 flatten(dnnc::tensor<bool> &a, int axis = 1) ;
extern dnnc::tensor<double> \
		 floor(dnnc::tensor<double> &a) ;
extern dnnc::tensor<float> \
		 floor(dnnc::tensor<float> &a) ;
extern dnnc::tensor<double> \
		 gemm(dnnc::tensor<double> &a, dnnc::tensor<double> &b, dnnc::tensor<double> &c, float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0) ;
extern dnnc::tensor<float> \
		 gemm(dnnc::tensor<float> &a, dnnc::tensor<float> &b, dnnc::tensor<float> &c, float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0) ;
extern dnnc::tensor<float> \
		 global_average_pool(dnnc::tensor<float> &a) ;
extern dnnc::tensor<double> \
		 global_average_pool(dnnc::tensor<double> &a) ;
extern dnnc::tensor<float> \
		 global_lp_pool(dnnc::tensor<float> &a, int p = 2) ;
extern dnnc::tensor<double> \
		 global_lp_pool(dnnc::tensor<double> &a, int p = 2) ;
extern dnnc::tensor<float> \
		 global_max_pool(dnnc::tensor<float> &a) ;
extern dnnc::tensor<double> \
		 global_max_pool(dnnc::tensor<double> &a) ;
extern dnnc::tensor<bool> \
		 greater(dnnc::tensor<int> &a, dnnc::tensor<int> &b) ;
extern dnnc::tensor<bool> \
		 greater(dnnc::tensor<float> &a, dnnc::tensor<float> &b) ;
extern dnnc::tensor<bool> \
		 greater(dnnc::tensor<double> &a, dnnc::tensor<double> &b) ;
extern dnnc::tensor<float> \
		 hardmax(dnnc::tensor<float> &a, int axis = 0) ;
extern dnnc::tensor<double> \
		 hardmax(dnnc::tensor<double> &a, int axis = 0) ;
extern dnnc::tensor<float> \
		 hardsigmoid(dnnc::tensor<float> &a, float alpha = 0.2,float beta = 0.5) ;
extern dnnc::tensor<double> \
		 hardsigmoid(dnnc::tensor<double> &a, float alpha = 0.2,float beta = 0.5) ;
extern dnnc::tensor<bool> \
		 identity(dnnc::tensor<bool> &a) ;
extern dnnc::tensor<int> \
		 identity(dnnc::tensor<int> &a) ;
extern dnnc::tensor<float> \
		 identity(dnnc::tensor<float> &a) ;
extern dnnc::tensor<double> \
		 identity(dnnc::tensor<double> &a) ;
extern dnnc::tensor<float> \
		 instancenormalization(dnnc::tensor<float> &a, dnnc::tensor<float> &scale,dnnc::tensor<float> &B, float epsilon = 1e-5) ;
extern dnnc::tensor<double> \
		 instancenormalization(dnnc::tensor<double> &a, dnnc::tensor<double> &scale,dnnc::tensor<double> &B, float epsilon = 1e-5) ;
extern dnnc::tensor<bool> \
		 isinf(dnnc::tensor<float> &a, int detect_positive = 1,int detect_negative = 1) ;
extern dnnc::tensor<bool> \
		 isinf(dnnc::tensor<double> &a, int detect_positive = 1,int detect_negative = 1) ;
extern dnnc::tensor<bool> \
		 isnan(dnnc::tensor<float> &a) ;
extern dnnc::tensor<bool> \
		 isnan(dnnc::tensor<double> &a) ;
extern dnnc::tensor<float> \
		 lrn(dnnc::tensor<float> &a, int size, float alpha = 0.0001,float beta = 0.75, float bias = 1.0) ;
extern dnnc::tensor<double> \
		 lrn(dnnc::tensor<double> &a, int size, float alpha = 0.0001,float beta = 0.75, float bias = 1.0) ;
extern dnnc::tensor<float> \
		 leakyrelu(dnnc::tensor<float> &a, float alpha = 0.01) ;
extern dnnc::tensor<double> \
		 leakyrelu(dnnc::tensor<double> &a, float alpha = 0.01) ;
extern dnnc::tensor<bool> \
		 less(dnnc::tensor<int> &a, dnnc::tensor<int> &b) ;
extern dnnc::tensor<bool> \
		 less(dnnc::tensor<float> &a, dnnc::tensor<float> &b) ;
extern dnnc::tensor<bool> \
		 less(dnnc::tensor<double> &a, dnnc::tensor<double> &b) ;
extern dnnc::tensor<float> \
		 log(dnnc::tensor<float> &a) ;
extern dnnc::tensor<double> \
		 log(dnnc::tensor<double> &a) ;
extern dnnc::tensor<float> \
		 lpnormalization(dnnc::tensor<float> &a, int p = 2, int axis = -1) ;
extern dnnc::tensor<double> \
		 lpnormalization(dnnc::tensor<double> &a, int p = 2, int axis = -1) ;
extern dnnc::tensor<int> \
		 matmulinteger(dnnc::tensor<int> &a, dnnc::tensor<int> &b) ;
extern dnnc::tensor<float> \
		 min(std::vector<dnnc::tensor<float>> floats) ;
extern dnnc::tensor<double> \
		 min(std::vector<dnnc::tensor<double>> doubles) ;
extern dnnc::tensor<float> \
		 mean(std::vector<dnnc::tensor<float>> floats) ;
extern dnnc::tensor<double> \
		 mean(std::vector<dnnc::tensor<double>> doubles) ;
extern dnnc::tensor<float> \
		 max(std::vector<dnnc::tensor<float>> floats) ;
extern dnnc::tensor<double> \
		 max(std::vector<dnnc::tensor<double>> doubles) ;
extern dnnc::tensor<float> \
		 thresholded_relu(dnnc::tensor<float> &a) ;
extern dnnc::tensor<float> \
		 transpose(dnnc::tensor<float> &a) ;
