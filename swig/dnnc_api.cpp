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

#include <Python.h>

#include "core/tensor.h"
#include "operators/Add.h"
#include "operators/Abs.h"
#include "operators/And.h"
#include "operators/BitShift.h"
#include "operators/DequantizeLinear.h"
#include "operators/Div.h"
#include "operators/Dropout.h"
#include "operators/Elu.h"
#include "operators/Equal.h"
#include "operators/Erf.h"
#include "operators/Exp.h"
#include "operators/Expand.h"
#include "operators/EyeLike.h"
#include "operators/Flatten.h"
#include "operators/Floor.h"
#include "operators/FloorDiv.h"
#include "operators/Gemm.h"
#include "operators/GlobalAveragePool.h"
#include "operators/GlobalLpPool.h"
#include "operators/GlobalMaxPool.h"
#include "operators/Greater.h"
#include "operators/GreaterEqual.h"
#include "operators/HardSigmoid.h"
#include "operators/Hardmax.h"
#include "operators/Identity.h"
#include "operators/InstanceNormalization.h"
#include "operators/IsInf.h"
#include "operators/IsNaN.h"
#include "operators/LRN.h"
#include "operators/LeakyRelu.h"
#include "operators/Less.h"
#include "operators/LessEqual.h"
#include "operators/Log.h"
#include "operators/LogSoftmax.h"
#include "operators/LpNormalization.h"
#include "operators/MatMul.h"
#include "operators/MatMulInteger.h"
#include "operators/Max.h"
#include "operators/Mean.h"
#include "operators/Min.h"
#include "operators/Mod.h"
#include "operators/Mul.h"
#include "operators/Neg.h"
#include "operators/Not.h"
#include "operators/NotEqual.h"
#include "operators/Or.h"
#include "operators/Pow.h"
#include "operators/Reciprocal.h"
#include "operators/Remainder.h"
#include "operators/Sub.h"
#include "operators/Tan.h"
#include "operators/Tanh.h"
#include "operators/ThresholdedRelu.h"
#include "operators/Transpose.h"
#include "operators/TrueDiv.h"
#include "operators/Xor.h"
#include "operators/Sigmoid.h"
#include "operators/Sign.h"
#include "operators/Sin.h"
#include "operators/Sinh.h"
#include "operators/Softplus.h"
#include "operators/Softmax.h"
#include "operators/Softsign.h"
#include "operators/Sqrt.h"
#include "operators/Clip.h"
#include "operators/Ceil.h"
#include "operators/Cos.h"
#include "operators/Cosh.h"
#include "operators/Slice.h"
#include "operators/PRelu.h"

extern std::vector<float> listTupleToVector_Float(PyObject *);
extern std::vector<size_t> listTupleToVector_SizeT(PyObject *);

namespace dnnc {

	tensor<double> add(tensor<double> &a, tensor<double> &b) {
		Add<double, double> op;
		return op.compute(a, b);
	}
	
	tensor<double> add(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		Add<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> add(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		Add<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> add(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		Add<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> add(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		Add<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> add(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Add<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> add(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Add<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> add(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Add<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> add(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Add<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<float> add(tensor<float> &a, tensor<float> &b) {
		Add<float, float> op;
		return op.compute(a, b);
	}
	
	tensor<float> add(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		Add<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> add(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		Add<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> add(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		Add<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> add(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Add<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> add(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Add<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> add(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Add<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<size_t> add(tensor<size_t> &a, tensor<size_t> &b) {
		Add<size_t, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<size_t> add(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Add<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> add(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Add<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> add(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Add<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<size_t> add(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Add<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<int> add(tensor<int> &a, tensor<int> &b) {
		Add<int, int> op;
		return op.compute(a, b);
	}
	
	tensor<int> add(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		Add<int, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> add(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		Add<int, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> add(tensor<bool> &a, tensor<bool> &b) {
		Add<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<double> add(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<double> add(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<double> add(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<double> add(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<double> add(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<double> add(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<double> add(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<double> add(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<double> add(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<float> add(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<float> add(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<float> add(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<float> add(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<float> add(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<float> add(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<float> add(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<size_t> add(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<size_t> add(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<size_t> add(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<size_t> add(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<size_t> add(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<int> add(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<int> add(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<int> add(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<bool> add(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return add(a, tensor_b);
	}
	
	tensor<double> add(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<double> add(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<double> add(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<double> add(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<double> add(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<double> add(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<double> add(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<double> add(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<double> add(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<float> add(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<float> add(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<float> add(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<float> add(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<float> add(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<float> add(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<float> add(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<size_t> add(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<size_t> add(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<size_t> add(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<size_t> add(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<size_t> add(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<int> add(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<int> add(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<int> add(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	tensor<bool> add(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return add(tensor_a, b);
	}
	
	double add(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	double add(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	double add(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	double add(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	double add(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	double add(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	double add(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	double add(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	double add(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	float add(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	float add(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	float add(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	float add(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	float add(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	float add(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	float add(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	size_t add(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	size_t add(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	size_t add(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	size_t add(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	size_t add(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	int add(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	int add(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	int add(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	bool add(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return add(tensor_a, tensor_b)[0];
	}
	
	tensor<double> sub(tensor<double> &a, tensor<double> &b) {
		Sub<double, double> op;
		return op.compute(a, b);
	}
	
	tensor<double> sub(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		Sub<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> sub(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		Sub<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> sub(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		Sub<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> sub(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		Sub<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> sub(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Sub<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> sub(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Sub<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> sub(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Sub<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> sub(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Sub<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<float> sub(tensor<float> &a, tensor<float> &b) {
		Sub<float, float> op;
		return op.compute(a, b);
	}
	
	tensor<float> sub(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		Sub<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> sub(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		Sub<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> sub(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		Sub<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> sub(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Sub<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> sub(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Sub<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> sub(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Sub<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<size_t> sub(tensor<size_t> &a, tensor<size_t> &b) {
		Sub<size_t, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<size_t> sub(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Sub<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> sub(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Sub<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> sub(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Sub<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<size_t> sub(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Sub<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<int> sub(tensor<int> &a, tensor<int> &b) {
		Sub<int, int> op;
		return op.compute(a, b);
	}
	
	tensor<int> sub(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		Sub<int, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> sub(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		Sub<int, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> sub(tensor<bool> &a, tensor<bool> &b) {
		Sub<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<double> sub(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<double> sub(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<double> sub(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<double> sub(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<double> sub(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<double> sub(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<double> sub(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<double> sub(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<double> sub(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<float> sub(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<float> sub(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<float> sub(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<float> sub(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<float> sub(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<float> sub(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<float> sub(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<size_t> sub(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<size_t> sub(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<size_t> sub(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<size_t> sub(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<size_t> sub(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<int> sub(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<int> sub(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<int> sub(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<bool> sub(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return sub(a, tensor_b);
	}
	
	tensor<double> sub(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<double> sub(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<double> sub(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<double> sub(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<double> sub(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<double> sub(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<double> sub(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<double> sub(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<double> sub(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<float> sub(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<float> sub(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<float> sub(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<float> sub(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<float> sub(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<float> sub(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<float> sub(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<size_t> sub(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<size_t> sub(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<size_t> sub(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<size_t> sub(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<size_t> sub(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<int> sub(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<int> sub(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<int> sub(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	tensor<bool> sub(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return sub(tensor_a, b);
	}
	
	double sub(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	double sub(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	double sub(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	double sub(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	double sub(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	double sub(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	double sub(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	double sub(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	double sub(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	float sub(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	float sub(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	float sub(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	float sub(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	float sub(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	float sub(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	float sub(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	size_t sub(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	size_t sub(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	size_t sub(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	size_t sub(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	size_t sub(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	int sub(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	int sub(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	int sub(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	bool sub(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return sub(tensor_a, tensor_b)[0];
	}
	
	tensor<double> mul(tensor<double> &a, tensor<double> &b) {
		Mul<double, double> op;
		return op.compute(a, b);
	}
	
	tensor<double> mul(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		Mul<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> mul(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		Mul<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> mul(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		Mul<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> mul(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		Mul<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> mul(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Mul<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> mul(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Mul<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> mul(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Mul<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> mul(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Mul<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<float> mul(tensor<float> &a, tensor<float> &b) {
		Mul<float, float> op;
		return op.compute(a, b);
	}
	
	tensor<float> mul(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		Mul<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> mul(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		Mul<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> mul(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		Mul<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> mul(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Mul<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> mul(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Mul<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> mul(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Mul<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<size_t> mul(tensor<size_t> &a, tensor<size_t> &b) {
		Mul<size_t, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<size_t> mul(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Mul<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> mul(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Mul<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> mul(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Mul<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<size_t> mul(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Mul<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<int> mul(tensor<int> &a, tensor<int> &b) {
		Mul<int, int> op;
		return op.compute(a, b);
	}
	
	tensor<int> mul(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		Mul<int, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> mul(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		Mul<int, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> mul(tensor<bool> &a, tensor<bool> &b) {
		Mul<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<double> mul(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<double> mul(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<double> mul(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<double> mul(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<double> mul(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<double> mul(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<double> mul(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<double> mul(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<double> mul(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<float> mul(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<float> mul(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<float> mul(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<float> mul(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<float> mul(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<float> mul(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<float> mul(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<size_t> mul(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<size_t> mul(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<size_t> mul(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<size_t> mul(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<size_t> mul(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<int> mul(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<int> mul(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<int> mul(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<bool> mul(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return mul(a, tensor_b);
	}
	
	tensor<double> mul(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<double> mul(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<double> mul(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<double> mul(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<double> mul(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<double> mul(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<double> mul(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<double> mul(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<double> mul(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<float> mul(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<float> mul(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<float> mul(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<float> mul(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<float> mul(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<float> mul(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<float> mul(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<size_t> mul(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<size_t> mul(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<size_t> mul(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<size_t> mul(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<size_t> mul(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<int> mul(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<int> mul(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<int> mul(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	tensor<bool> mul(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return mul(tensor_a, b);
	}
	
	double mul(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	double mul(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	double mul(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	double mul(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	double mul(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	double mul(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	double mul(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	double mul(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	double mul(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	float mul(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	float mul(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	float mul(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	float mul(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	float mul(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	float mul(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	float mul(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	size_t mul(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	size_t mul(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	size_t mul(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	size_t mul(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	size_t mul(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	int mul(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	int mul(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	int mul(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	bool mul(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return mul(tensor_a, tensor_b)[0];
	}
	
	tensor<int> floor_div(tensor<double> &a, tensor<double> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<double> &a, tensor<float> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<double> &a, tensor<size_t> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<double> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<int> floor_div(tensor<double> &a, tensor<bool> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<float> &a, tensor<double> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, tensor<double> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<int> &a, tensor<double> &b) {
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, tensor<double> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<float> &a, tensor<float> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<float> &a, tensor<size_t> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<float> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<int> floor_div(tensor<float> &a, tensor<bool> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, tensor<float> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<int> &a, tensor<float> &b) {
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, tensor<float> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, tensor<size_t> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, tensor<bool> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<int> &a, tensor<size_t> &b) {
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, tensor<size_t> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<int> &a, tensor<int> &b) {
		FloorDiv<int, int> op;
		return op.compute(a, b);
	}
	
	tensor<int> floor_div(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, tensor<bool> &b) {
		tensor<int> int_a = a.asType<int>();
		tensor<int> int_b = b.asType<int>();
		FloorDiv<int, int> op;
		return op.compute(int_a, int_b);
	}
	
	tensor<int> floor_div(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return floor_div(a, tensor_b);
	}
	
	tensor<int> floor_div(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	tensor<int> floor_div(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return floor_div(tensor_a, b);
	}
	
	int floor_div(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	int floor_div(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return floor_div(tensor_a, tensor_b)[0];
	}
	
	tensor<double> true_div(tensor<double> &a, tensor<double> &b) {
		TrueDiv<double, double> op;
		return op.compute(a, b);
	}
	
	tensor<double> true_div(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> true_div(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> true_div(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> true_div(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> true_div(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> true_div(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> true_div(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> true_div(tensor<float> &a, tensor<float> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<float> &a, tensor<size_t> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<float> &a, tensor<int> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<float> &a, tensor<bool> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, tensor<float> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<int> &a, tensor<float> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<bool> &a, tensor<float> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, tensor<size_t> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, tensor<int> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, tensor<bool> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<int> &a, tensor<size_t> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<bool> &a, tensor<size_t> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<int> &a, tensor<int> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<int> &a, tensor<bool> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<bool> &a, tensor<int> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<bool> &a, tensor<bool> &b) {
		tensor<double> double_a = a.asType<double>();
		tensor<double> double_b = b.asType<double>();
		TrueDiv<double, double> op;
		return op.compute(double_a, double_b);
	}
	
	tensor<double> true_div(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return true_div(a, tensor_b);
	}
	
	tensor<double> true_div(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	tensor<double> true_div(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return true_div(tensor_a, b);
	}
	
	double true_div(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	double true_div(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return true_div(tensor_a, tensor_b)[0];
	}
	
	tensor<double> remainder(tensor<double> &a, tensor<double> &b) {
		Remainder<double, double> op;
		return op.compute(a, b);
	}
	
	tensor<double> remainder(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		Remainder<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> remainder(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		Remainder<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> remainder(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		Remainder<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> remainder(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		Remainder<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> remainder(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Remainder<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> remainder(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Remainder<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> remainder(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Remainder<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> remainder(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Remainder<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<float> remainder(tensor<float> &a, tensor<float> &b) {
		Remainder<float, float> op;
		return op.compute(a, b);
	}
	
	tensor<float> remainder(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		Remainder<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> remainder(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		Remainder<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> remainder(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		Remainder<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> remainder(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Remainder<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> remainder(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Remainder<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> remainder(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Remainder<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<size_t> remainder(tensor<size_t> &a, tensor<size_t> &b) {
		Remainder<size_t, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<size_t> remainder(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Remainder<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> remainder(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Remainder<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> remainder(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Remainder<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<size_t> remainder(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Remainder<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<int> remainder(tensor<int> &a, tensor<int> &b) {
		Remainder<int, int> op;
		return op.compute(a, b);
	}
	
	tensor<int> remainder(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		Remainder<int, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> remainder(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		Remainder<int, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> remainder(tensor<bool> &a, tensor<bool> &b) {
		Remainder<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<double> remainder(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<double> remainder(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<double> remainder(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<double> remainder(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<double> remainder(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<double> remainder(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<double> remainder(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<double> remainder(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<double> remainder(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<float> remainder(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<float> remainder(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<float> remainder(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<float> remainder(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<float> remainder(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<float> remainder(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<float> remainder(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<size_t> remainder(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<size_t> remainder(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<size_t> remainder(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<size_t> remainder(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<size_t> remainder(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<int> remainder(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<int> remainder(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<int> remainder(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<bool> remainder(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return remainder(a, tensor_b);
	}
	
	tensor<double> remainder(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<double> remainder(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<double> remainder(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<double> remainder(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<double> remainder(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<double> remainder(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<double> remainder(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<double> remainder(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<double> remainder(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<float> remainder(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<float> remainder(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<float> remainder(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<float> remainder(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<float> remainder(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<float> remainder(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<float> remainder(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<size_t> remainder(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<size_t> remainder(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<size_t> remainder(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<size_t> remainder(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<size_t> remainder(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<int> remainder(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<int> remainder(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<int> remainder(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	tensor<bool> remainder(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return remainder(tensor_a, b);
	}
	
	double remainder(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	double remainder(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	double remainder(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	double remainder(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	double remainder(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	double remainder(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	double remainder(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	double remainder(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	double remainder(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	float remainder(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	float remainder(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	float remainder(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	float remainder(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	float remainder(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	float remainder(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	float remainder(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	size_t remainder(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	size_t remainder(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	size_t remainder(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	size_t remainder(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	size_t remainder(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	int remainder(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	int remainder(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	int remainder(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	bool remainder(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return remainder(tensor_a, tensor_b)[0];
	}
	
	tensor<double> power(tensor<double> &a, tensor<double> &b) {
		Pow<double, double> op;
		return op.compute(a, b);
	}
	
	tensor<double> power(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		Pow<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> power(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		Pow<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> power(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		Pow<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> power(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		Pow<double, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> power(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Pow<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> power(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Pow<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> power(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Pow<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> power(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Pow<double, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<float> power(tensor<float> &a, tensor<float> &b) {
		Pow<float, float> op;
		return op.compute(a, b);
	}
	
	tensor<float> power(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		Pow<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> power(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		Pow<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> power(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		Pow<float, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> power(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Pow<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> power(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Pow<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> power(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Pow<float, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<size_t> power(tensor<size_t> &a, tensor<size_t> &b) {
		Pow<size_t, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<size_t> power(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Pow<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> power(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Pow<size_t, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<size_t> power(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Pow<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<size_t> power(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Pow<size_t, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<int> power(tensor<int> &a, tensor<int> &b) {
		Pow<int, int> op;
		return op.compute(a, b);
	}
	
	tensor<int> power(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		Pow<int, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> power(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		Pow<int, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> power(tensor<bool> &a, tensor<bool> &b) {
		Pow<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<double> power(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<double> power(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<double> power(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<double> power(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<double> power(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<double> power(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<double> power(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<double> power(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<double> power(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<float> power(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<float> power(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<float> power(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<float> power(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<float> power(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<float> power(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<float> power(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<size_t> power(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<size_t> power(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<size_t> power(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<size_t> power(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<size_t> power(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<int> power(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<int> power(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<int> power(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<bool> power(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return power(a, tensor_b);
	}
	
	tensor<double> power(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<double> power(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<double> power(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<double> power(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<double> power(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<double> power(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<double> power(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<double> power(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<double> power(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<float> power(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<float> power(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<float> power(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<float> power(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<float> power(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<float> power(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<float> power(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<size_t> power(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<size_t> power(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<size_t> power(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<size_t> power(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<size_t> power(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<int> power(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<int> power(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<int> power(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	tensor<bool> power(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return power(tensor_a, b);
	}
	
	double power(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	double power(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	double power(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	double power(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	double power(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	double power(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	double power(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	double power(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	double power(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	float power(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	float power(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	float power(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	float power(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	float power(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	float power(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	float power(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	size_t power(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	size_t power(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	size_t power(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	size_t power(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	size_t power(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	int power(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	int power(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	int power(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	bool power(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return power(tensor_a, tensor_b)[0];
	}
	
	tensor<bool> logical_and(tensor<double> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<double> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<double> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<double> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<double> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, tensor<double> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, tensor<float> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, tensor<size_t> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		And<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, tensor<int> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		And<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, tensor<bool> &b) {
		And<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_and(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_and(a, tensor_b);
	}
	
	tensor<bool> logical_and(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	tensor<bool> logical_and(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_and(tensor_a, b);
	}
	
	bool logical_and(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	bool logical_and(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_and(tensor_a, tensor_b)[0];
	}
	
	tensor<bool> logical_or(tensor<double> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<double> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<double> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<double> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<double> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, tensor<double> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, tensor<float> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, tensor<size_t> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		Or<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, tensor<int> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		Or<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, tensor<bool> &b) {
		Or<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_or(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_or(a, tensor_b);
	}
	
	tensor<bool> logical_or(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	tensor<bool> logical_or(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_or(tensor_a, b);
	}
	
	bool logical_or(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	bool logical_or(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_or(tensor_a, tensor_b)[0];
	}
	
	tensor<bool> logical_xor(tensor<double> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<double> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<double> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<double> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<double> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, tensor<double> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, tensor<double> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, tensor<float> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, tensor<float> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, tensor<size_t> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, tensor<size_t> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, tensor<int> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, tensor<bool> &b) {
		tensor<bool> bool_a = a.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(bool_a, b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, tensor<int> &b) {
		tensor<bool> bool_b = b.asType<bool>();
		Xor<bool, bool> op;
		return op.compute(a, bool_b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, tensor<bool> &b) {
		Xor<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_xor(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return logical_xor(a, tensor_b);
	}
	
	tensor<bool> logical_xor(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	tensor<bool> logical_xor(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return logical_xor(tensor_a, b);
	}
	
	bool logical_xor(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	bool logical_xor(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return logical_xor(tensor_a, tensor_b)[0];
	}
	
	tensor<bool> equal(tensor<double> &a, tensor<double> &b) {
		Equal<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> equal(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		Equal<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> equal(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		Equal<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> equal(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		Equal<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> equal(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		Equal<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> equal(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Equal<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Equal<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> equal(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Equal<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> equal(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Equal<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> equal(tensor<float> &a, tensor<float> &b) {
		Equal<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> equal(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		Equal<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> equal(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		Equal<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> equal(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		Equal<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Equal<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> equal(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Equal<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> equal(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Equal<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, tensor<size_t> &b) {
		Equal<bool, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Equal<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Equal<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> equal(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Equal<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> equal(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Equal<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> equal(tensor<int> &a, tensor<int> &b) {
		Equal<bool, int> op;
		return op.compute(a, b);
	}
	
	tensor<bool> equal(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		Equal<bool, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<bool> equal(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		Equal<bool, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> equal(tensor<bool> &a, tensor<bool> &b) {
		Equal<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> equal(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return equal(a, tensor_b);
	}
	
	tensor<bool> equal(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	tensor<bool> equal(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return equal(tensor_a, b);
	}
	
	bool equal(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	bool equal(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return equal(tensor_a, tensor_b)[0];
	}
	
	tensor<bool> less(tensor<double> &a, tensor<double> &b) {
		Less<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		Less<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> less(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		Less<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> less(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		Less<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> less(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		Less<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> less(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Less<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> less(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Less<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> less(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Less<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> less(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Less<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> less(tensor<float> &a, tensor<float> &b) {
		Less<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		Less<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> less(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		Less<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> less(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		Less<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> less(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Less<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> less(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Less<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> less(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Less<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> less(tensor<size_t> &a, tensor<size_t> &b) {
		Less<bool, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Less<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> less(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Less<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> less(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Less<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> less(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Less<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> less(tensor<int> &a, tensor<int> &b) {
		Less<bool, int> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		Less<bool, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<bool> less(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		Less<bool, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> less(tensor<bool> &a, tensor<bool> &b) {
		Less<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less(a, tensor_b);
	}
	
	tensor<bool> less(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	tensor<bool> less(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less(tensor_a, b);
	}
	
	bool less(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	bool less(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less(tensor_a, tensor_b)[0];
	}
	
	tensor<bool> greater(tensor<double> &a, tensor<double> &b) {
		Greater<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		Greater<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> greater(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		Greater<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> greater(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		Greater<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> greater(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		Greater<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> greater(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Greater<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Greater<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> greater(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Greater<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> greater(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Greater<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> greater(tensor<float> &a, tensor<float> &b) {
		Greater<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		Greater<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> greater(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		Greater<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> greater(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		Greater<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Greater<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> greater(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Greater<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> greater(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Greater<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, tensor<size_t> &b) {
		Greater<bool, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Greater<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		Greater<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> greater(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Greater<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> greater(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		Greater<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> greater(tensor<int> &a, tensor<int> &b) {
		Greater<bool, int> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		Greater<bool, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<bool> greater(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		Greater<bool, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> greater(tensor<bool> &a, tensor<bool> &b) {
		Greater<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater(a, tensor_b);
	}
	
	tensor<bool> greater(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	tensor<bool> greater(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater(tensor_a, b);
	}
	
	bool greater(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	bool greater(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater(tensor_a, tensor_b)[0];
	}
	
	tensor<bool> not_equal(tensor<double> &a, tensor<double> &b) {
		NotEqual<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> not_equal(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		NotEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> not_equal(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		NotEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> not_equal(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		NotEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> not_equal(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		NotEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		NotEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		NotEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		NotEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		NotEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, tensor<float> &b) {
		NotEqual<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		NotEqual<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		NotEqual<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		NotEqual<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		NotEqual<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		NotEqual<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		NotEqual<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, tensor<size_t> &b) {
		NotEqual<bool, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		NotEqual<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		NotEqual<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		NotEqual<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		NotEqual<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, tensor<int> &b) {
		NotEqual<bool, int> op;
		return op.compute(a, b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		NotEqual<bool, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		NotEqual<bool, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, tensor<bool> &b) {
		NotEqual<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> not_equal(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return not_equal(a, tensor_b);
	}
	
	tensor<bool> not_equal(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	tensor<bool> not_equal(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return not_equal(tensor_a, b);
	}
	
	bool not_equal(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	bool not_equal(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return not_equal(tensor_a, tensor_b)[0];
	}
	
	tensor<bool> less_equal(tensor<double> &a, tensor<double> &b) {
		LessEqual<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		LessEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		LessEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		LessEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		LessEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		LessEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		LessEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		LessEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		LessEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, tensor<float> &b) {
		LessEqual<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		LessEqual<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		LessEqual<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		LessEqual<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		LessEqual<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		LessEqual<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		LessEqual<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, tensor<size_t> &b) {
		LessEqual<bool, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		LessEqual<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		LessEqual<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		LessEqual<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		LessEqual<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, tensor<int> &b) {
		LessEqual<bool, int> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		LessEqual<bool, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		LessEqual<bool, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, tensor<bool> &b) {
		LessEqual<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return less_equal(a, tensor_b);
	}
	
	tensor<bool> less_equal(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	tensor<bool> less_equal(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return less_equal(tensor_a, b);
	}
	
	bool less_equal(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	bool less_equal(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return less_equal(tensor_a, tensor_b)[0];
	}
	
	tensor<bool> greater_equal(tensor<double> &a, tensor<double> &b) {
		GreaterEqual<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		GreaterEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, tensor<size_t> &b) {
		tensor<double> double_b = b.asType<double>();
		GreaterEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		GreaterEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		GreaterEqual<bool, double> op;
		return op.compute(a, double_b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		GreaterEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		GreaterEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		GreaterEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		GreaterEqual<bool, double> op;
		return op.compute(double_a, b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, tensor<float> &b) {
		GreaterEqual<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, tensor<size_t> &b) {
		tensor<float> float_b = b.asType<float>();
		GreaterEqual<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		GreaterEqual<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		GreaterEqual<bool, float> op;
		return op.compute(a, float_b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		GreaterEqual<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		GreaterEqual<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		GreaterEqual<bool, float> op;
		return op.compute(float_a, b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, tensor<size_t> &b) {
		GreaterEqual<bool, size_t> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, tensor<int> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		GreaterEqual<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, tensor<bool> &b) {
		tensor<size_t> size_t_b = b.asType<size_t>();
		GreaterEqual<bool, size_t> op;
		return op.compute(a, size_t_b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		GreaterEqual<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, tensor<size_t> &b) {
		tensor<size_t> size_t_a = a.asType<size_t>();
		GreaterEqual<bool, size_t> op;
		return op.compute(size_t_a, b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, tensor<int> &b) {
		GreaterEqual<bool, int> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		GreaterEqual<bool, int> op;
		return op.compute(a, int_b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		GreaterEqual<bool, int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, tensor<bool> &b) {
		GreaterEqual<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, double b) {
		tensor<double> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, float b) {
		tensor<float> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<size_t> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, size_t b) {
		tensor<size_t> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, int b) {
		tensor<int> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(tensor<bool> &a, bool b) {
		tensor<bool> tensor_b(1);
		tensor_b.load(&b);
		return greater_equal(a, tensor_b);
	}
	
	tensor<bool> greater_equal(double a, tensor<double> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(double a, tensor<float> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(double a, tensor<size_t> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(double a, tensor<int> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(double a, tensor<bool> &b) {
		tensor<double> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(float a, tensor<double> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(size_t a, tensor<double> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(int a, tensor<double> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(bool a, tensor<double> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(float a, tensor<float> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(float a, tensor<size_t> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(float a, tensor<int> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(float a, tensor<bool> &b) {
		tensor<float> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(size_t a, tensor<float> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(int a, tensor<float> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(bool a, tensor<float> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(size_t a, tensor<size_t> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(size_t a, tensor<int> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(size_t a, tensor<bool> &b) {
		tensor<size_t> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(int a, tensor<size_t> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(bool a, tensor<size_t> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(int a, tensor<int> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(int a, tensor<bool> &b) {
		tensor<int> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(bool a, tensor<int> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	tensor<bool> greater_equal(bool a, tensor<bool> &b) {
		tensor<bool> tensor_a(1);
		tensor_a.load(&a);
		return greater_equal(tensor_a, b);
	}
	
	bool greater_equal(double a, double b) {
		tensor<double> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(double a, float b) {
		tensor<double> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(double a, size_t b) {
		tensor<double> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(double a, int b) {
		tensor<double> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(double a, bool b) {
		tensor<double> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(float a, double b) {
		tensor<float> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(size_t a, double b) {
		tensor<size_t> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(int a, double b) {
		tensor<int> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(bool a, double b) {
		tensor<bool> tensor_a(1);
		tensor<double> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(float a, float b) {
		tensor<float> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(float a, size_t b) {
		tensor<float> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(float a, int b) {
		tensor<float> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(float a, bool b) {
		tensor<float> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(size_t a, float b) {
		tensor<size_t> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(int a, float b) {
		tensor<int> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(bool a, float b) {
		tensor<bool> tensor_a(1);
		tensor<float> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(size_t a, size_t b) {
		tensor<size_t> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(size_t a, int b) {
		tensor<size_t> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(size_t a, bool b) {
		tensor<size_t> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(int a, size_t b) {
		tensor<int> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(bool a, size_t b) {
		tensor<bool> tensor_a(1);
		tensor<size_t> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(int a, int b) {
		tensor<int> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(int a, bool b) {
		tensor<int> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(bool a, int b) {
		tensor<bool> tensor_a(1);
		tensor<int> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	bool greater_equal(bool a, bool b) {
		tensor<bool> tensor_a(1);
		tensor<bool> tensor_b(1);
		tensor_a.load(&a);
		tensor_b.load(&b);
		return greater_equal(tensor_a, tensor_b)[0];
	}
	
	tensor<float> array(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
		return tensor<float>(x, y, z, w, "", dnnc::INIT_RANDOM);
	}
	
	tensor<float> empty(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
		return tensor<float>(x, y, z, w, "", dnnc::INIT_NONE);
	}
	
	tensor<float> zeros(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
		return tensor<float>(x, y, z, w, "", dnnc::INIT_ZERO);
	}
	
	tensor<float> ones(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
		return tensor<float>(x, y, z, w, "", dnnc::INIT_ONE);
	}
	
	tensor<float> random(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
		return tensor<float>(x, y, z, w, "", dnnc::INIT_RANDOM);
	}
	
	tensor<float> array(PyObject *objects) {
		if (!PyList_Check(objects))
			throw std::logic_error("array expects list to create array.\n");
		Py_ssize_t sz = PyList_Size(objects);
		if (!sz)
			throw std::logic_error("array expects list of non-zero size.\n");
		PyObject *a_list = PyList_GetItem(objects, 0);
		if (PyList_Check(a_list)) {
			// 2D: dc.array(([1,2],[10,20]]);
			// Py_ssize_t sz = PyList_Size(a_list);
			std::vector<size_t> vDims;
			std::vector<float> vContents;
			for (Py_ssize_t i = 0; i < sz; i++) {
				std::vector<float> aVec =
						listTupleToVector_Float(PyList_GetItem(objects, i));
				vDims.push_back(aVec.size());
				vContents.insert(vContents.end(), aVec.begin(), aVec.end());
			}
			// make sure VDims are consistent.
			for (size_t i = 1; i < vDims.size(); i++) {
				if (vDims[0] != vDims[i])
					throw std::logic_error(
							"2D array expects list of lists with same size.");
			}
			// fill it as row major data.
			tensor<float> result(vDims.size(), vDims[0]);
			result.load(vContents);
			return result;
		} else {
			// 1D: dc.array([2,3,4])
			std::vector<float> aVec = listTupleToVector_Float(objects);
			tensor<float> result(aVec.size());
			result.load(aVec);
			return result;
		}
		return tensor<float>();
	}
	
	tensor<float> arange(size_t stop, size_t start = 0, size_t step = 1) {
		// swap if the range is invalid.
		if (stop < start) {
			size_t tmp = start;
			start = stop;
			stop = tmp;
		}
		if (stop == start)
			throw std::logic_error("arange expects stop arg more than start.\n");
		std::vector<float> vContents;
		for (size_t i = start; i < stop; i = i + step)
			vContents.push_back(static_cast<float>(i));
		tensor<float> result(vContents.size());
		result.load(vContents);
		return result;
	}
	
	tensor<double> reshape(tensor<double> &tens_obj, PyObject *newShape) {
		tensor<double> result ;
		if (PyLong_Check(newShape)) {
			std::vector<size_t> nShape;
			auto newShapeMember = PyLong_AsLong(newShape);
			nShape.push_back(static_cast<size_t>(newShapeMember));
			if (nShape[0] <= 0)
				throw std::logic_error("reshape integer must have positive value.\n");
			result = tens_obj.reshape(nShape);
		    return result;
		}else if (PyTuple_Check(newShape)) {
			auto vShape = listTupleToVector_SizeT(newShape);
			for (size_t i = 0; i < vShape.size(); i++)
				if (vShape[i] <= 0)
					throw std::logic_error("reshape tupel must have positive elements.\n");
			result = tens_obj.reshape(vShape);
		    return result;
		}else {
			throw std::logic_error("reshape type must be int or tuple.\n");
	    }
		return tens_obj;
	}
	
	tensor<float> reshape(tensor<float> &tens_obj, PyObject *newShape) {
		tensor<float> result ;
		if (PyLong_Check(newShape)) {
			std::vector<size_t> nShape;
			auto newShapeMember = PyLong_AsLong(newShape);
			nShape.push_back(static_cast<size_t>(newShapeMember));
			if (nShape[0] <= 0)
				throw std::logic_error("reshape integer must have positive value.\n");
			result = tens_obj.reshape(nShape);
		    return result;
		}else if (PyTuple_Check(newShape)) {
			auto vShape = listTupleToVector_SizeT(newShape);
			for (size_t i = 0; i < vShape.size(); i++)
				if (vShape[i] <= 0)
					throw std::logic_error("reshape tupel must have positive elements.\n");
			result = tens_obj.reshape(vShape);
		    return result;
		}else {
			throw std::logic_error("reshape type must be int or tuple.\n");
	    }
		return tens_obj;
	}
	
	tensor<int> reshape(tensor<int> &tens_obj, PyObject *newShape) {
		tensor<int> result ;
		if (PyLong_Check(newShape)) {
			std::vector<size_t> nShape;
			auto newShapeMember = PyLong_AsLong(newShape);
			nShape.push_back(static_cast<size_t>(newShapeMember));
			if (nShape[0] <= 0)
				throw std::logic_error("reshape integer must have positive value.\n");
			result = tens_obj.reshape(nShape);
		    return result;
		}else if (PyTuple_Check(newShape)) {
			auto vShape = listTupleToVector_SizeT(newShape);
			for (size_t i = 0; i < vShape.size(); i++)
				if (vShape[i] <= 0)
					throw std::logic_error("reshape tupel must have positive elements.\n");
			result = tens_obj.reshape(vShape);
		    return result;
		}else {
			throw std::logic_error("reshape type must be int or tuple.\n");
	    }
		return tens_obj;
	}
	
	tensor<bool> reshape(tensor<bool> &tens_obj, PyObject *newShape) {
		tensor<bool> result ;
		if (PyLong_Check(newShape)) {
			std::vector<size_t> nShape;
			auto newShapeMember = PyLong_AsLong(newShape);
			nShape.push_back(static_cast<size_t>(newShapeMember));
			if (nShape[0] <= 0)
				throw std::logic_error("reshape integer must have positive value.\n");
			result = tens_obj.reshape(nShape);
		    return result;
		}else if (PyTuple_Check(newShape)) {
			auto vShape = listTupleToVector_SizeT(newShape);
			for (size_t i = 0; i < vShape.size(); i++)
				if (vShape[i] <= 0)
					throw std::logic_error("reshape tupel must have positive elements.\n");
			result = tens_obj.reshape(vShape);
		    return result;
		}else {
			throw std::logic_error("reshape type must be int or tuple.\n");
	    }
		return tens_obj;
	}
	
	tensor<double> matmul(tensor<double> &a, tensor<double> &b) {
		MatMul<double> op;
		return op.compute(a, b);
	}
	
	tensor<float> matmul(tensor<float> &a, tensor<float> &b) {
		MatMul<float> op;
		return op.compute(a, b);
	}
	
	tensor<int> matmul(tensor<int> &a, tensor<int> &b) {
		MatMul<int> op;
		return op.compute(a, b);
	}
	
	tensor<double> abs(tensor<double> &a) {
		Abs<double> op;
		return op.compute(a);
	}
	
	tensor<float> abs(tensor<float> &a) {
		Abs<float> op;
		return op.compute(a);
	}
	
	tensor<int> abs(tensor<int> &a) {
		Abs<int> op;
		return op.compute(a);
	}
	
	tensor<int> bitshift(tensor<int> &a, tensor<int> &b, std::string direction = "") {
		BitShift<int> op("localOpName", direction);
		return op.compute(a, b);
	}
	
	tensor<float> dequantize_linear(tensor<int> &a, tensor<float> &b, tensor<int> &c) {
		DequantizeLinear<float, int> op;
		return op.compute(a, b, c);
	}
	
	tensor<double> dropout(tensor<double> &a, float ratio = 0.5) {
		Dropout<double> op("localOpName", ratio);
		return op.compute(a);
	}
	
	tensor<float> dropout(tensor<float> &a, float ratio = 0.5) {
		Dropout<float> op("localOpName", ratio);
		return op.compute(a);
	}
	
	tensor<double> div(tensor<double> &a, tensor<double> &b) {
		Div<double> op;
		return op.compute(a, b);
	}
	
	tensor<float> div(tensor<float> &a, tensor<float> &b) {
		Div<float> op;
		return op.compute(a, b);
	}
	
	tensor<int> div(tensor<int> &a, tensor<int> &b) {
		Div<int> op;
		return op.compute(a, b);
	}
	
	tensor<double> elu(tensor<double> &a, float alpha = 1.0) {
		Elu<double> op("localOpName", alpha);
		return op.compute(a);
	}
	
	tensor<float> elu(tensor<float> &a, float alpha = 1.0) {
		Elu<float> op("localOpName", alpha);
		return op.compute(a);
	}
	
	tensor<double> erf(tensor<double> &a) {
		Erf<double> op;
		return op.compute(a);
	}
	
	tensor<float> erf(tensor<float> &a) {
		Erf<float> op;
		return op.compute(a);
	}
	
	tensor<double> exp(tensor<double> &a) {
		Exp<double> op;
		return op.compute(a);
	}
	
	tensor<float> exp(tensor<float> &a) {
		Exp<float> op;
		return op.compute(a);
	}
	
	tensor<double> expand(tensor<double> &a, tensor<int> &b) {
		Expand<double,int> op;
		return op.compute(a, b);
	}
	
	tensor<float> expand(tensor<float> &a, tensor<int> &b) {
		Expand<float,int> op;
		return op.compute(a, b);
	}
	
	tensor<double> eye_like(tensor<double> &a, int k = 0) {
		EyeLike<double> op("localOpName", k);
		return op.compute(a);
	}
	
	tensor<float> eye_like(tensor<float> &a, int k = 0) {
		EyeLike<float> op("localOpName", k);
		return op.compute(a);
	}
	
	tensor<int> eye_like(tensor<int> &a, int k = 0) {
		EyeLike<int> op("localOpName", k);
		return op.compute(a);
	}
	
	tensor<bool> eye_like(tensor<bool> &a, int k = 0) {
		EyeLike<bool> op("localOpName", k);
		return op.compute(a);
	}
	
	tensor<double> flatten(tensor<double> &a, int axis = 1) {
		Flatten<double> op("localOpName", axis);
		return op.compute(a);
	}
	
	tensor<float> flatten(tensor<float> &a, int axis = 1) {
		Flatten<float> op("localOpName", axis);
		return op.compute(a);
	}
	
	tensor<int> flatten(tensor<int> &a, int axis = 1) {
		Flatten<int> op("localOpName", axis);
		return op.compute(a);
	}
	
	tensor<bool> flatten(tensor<bool> &a, int axis = 1) {
		Flatten<bool> op("localOpName", axis);
		return op.compute(a);
	}
	
	tensor<double> floor(tensor<double> &a) {
		Floor<double> op;
		return op.compute(a);
	}
	
	tensor<float> floor(tensor<float> &a) {
		Floor<float> op;
		return op.compute(a);
	}
	
	tensor<double> gemm(tensor<double> &a, tensor<double> &b, tensor<double> &c, float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0) {
		Gemm<double> op("localOpName", alpha, beta, transA, transB);
		return op.compute(a, b, c);
	}
	
	tensor<float> gemm(tensor<float> &a, tensor<float> &b, tensor<float> &c, float alpha = 1.0, float beta = 1.0, int transA = 0, int transB = 0) {
		Gemm<float> op("localOpName", alpha, beta, transA, transB);
		return op.compute(a, b, c);
	}
	
	tensor<float> global_average_pool(tensor<float> &a) {
	  GlobalAveragePool<float> op;
	  return op.compute(a);
	}
	
	tensor<double> global_average_pool(tensor<double> &a) {
	  GlobalAveragePool<double> op;
	  return op.compute(a);
	}
	
	tensor<float> global_lp_pool(tensor<float> &a, int p = 2) {
	  GlobalLpPool<float> op("localOpName", p);
	  return op.compute(a);
	}
	
	tensor<double> global_lp_pool(tensor<double> &a, int p = 2) {
	  GlobalLpPool<double> op("localOpName", p);
	  return op.compute(a);
	}
	
	tensor<float> global_max_pool(tensor<float> &a) {
	  GlobalMaxPool<float> op;
	  return op.compute(a);
	}
	
	tensor<double> global_max_pool(tensor<double> &a) {
	  GlobalMaxPool<double> op;
	  return op.compute(a);
	}
	
	tensor<float> hardmax(tensor<float> &a, int axis = 0) {
	  Hardmax<float> op("localOpName", axis);
	  return op.compute(a);
	}
	
	tensor<double> hardmax(tensor<double> &a, int axis = 0) {
	  Hardmax<double> op("localOpName", axis);
	  return op.compute(a);
	}
	
	tensor<float> hardsigmoid(tensor<float> &a, float alpha = 0.2,float beta = 0.5) {
	  HardSigmoid<float> op("localOpName", alpha, beta);
	  return op.compute(a);
	}
	
	tensor<double> hardsigmoid(tensor<double> &a, float alpha = 0.2,float beta = 0.5) {
	  HardSigmoid<double> op("localOpName", alpha, beta);
	  return op.compute(a);
	}
	
	tensor<bool> identity(tensor<bool> &a) {
	  Identity<bool> op;
	  return op.compute(a);
	}
	
	tensor<int> identity(tensor<int> &a) {
	  Identity<int> op;
	  return op.compute(a);
	}
	
	tensor<float> identity(tensor<float> &a) {
	  Identity<float> op;
	  return op.compute(a);
	}
	
	tensor<double> identity(tensor<double> &a) {
	  Identity<double> op;
	  return op.compute(a);
	}
	
	tensor<float> instancenormalization(tensor<float> &a, tensor<float> &scale,tensor<float> &B, float epsilon = 1e-5) {
	  InstanceNormalization<float> op("localOpName", epsilon);
	  return op.compute(a, scale, B);
	}
	
	tensor<double> instancenormalization(tensor<double> &a, tensor<double> &scale,tensor<double> &B, float epsilon = 1e-5) {
	  InstanceNormalization<double> op("localOpName", epsilon);
	  return op.compute(a, scale, B);
	}
	
	tensor<bool> isinf(tensor<float> &a, int detect_positive = 1,int detect_negative = 1) {
	  IsInf<bool, float> op("localOpName", detect_positive, detect_negative);
	  return op.compute(a);
	}
	
	tensor<bool> isinf(tensor<double> &a, int detect_positive = 1,int detect_negative = 1) {
	  IsInf<bool, double> op("localOpName", detect_positive, detect_negative);
	  return op.compute(a);
	}
	
	tensor<bool> isnan(tensor<float> &a) {
	  IsNaN<bool, float> op;
	  return op.compute(a);
	}
	
	tensor<bool> isnan(tensor<double> &a) {
	  IsNaN<bool, double> op;
	  return op.compute(a);
	}
	
	tensor<float> lrn(tensor<float> &a, int size, float alpha = 0.0001,float beta = 0.75, float bias = 1.0) {
	  LRN<float> op(size, "localOpName", alpha, beta, bias);
	  return op.compute(a);
	}
	
	tensor<double> lrn(tensor<double> &a, int size, float alpha = 0.0001,float beta = 0.75, float bias = 1.0) {
	  LRN<double> op(size, "localOpName", alpha, beta, bias);
	  return op.compute(a);
	}
	
	tensor<float> leakyrelu(tensor<float> &a, float alpha = 0.01) {
	  LeakyRelu<float> op("localOpName", alpha);
	  return op.compute(a);
	}
	
	tensor<double> leakyrelu(tensor<double> &a, float alpha = 0.01) {
	  LeakyRelu<double> op("localOpName", alpha);
	  return op.compute(a);
	}
	
	tensor<float> reciprocal(tensor<float> &a) {
	  Reciprocal<float> op("localOpName");
	  return op.compute(a);
	}
	
	tensor<double> reciprocal(tensor<double> &a) {
	  Reciprocal<double> op("localOpName");
	  return op.compute(a);
	}
	
	tensor<float> log(tensor<float> &a) {
	  Log<float> op;
	  return op.compute(a);
	}
	
	tensor<double> log(tensor<double> &a) {
	  Log<double> op;
	  return op.compute(a);
	}
	
	tensor<float> logsoftmax(tensor<float> &a, int axis = 1) {
		LogSoftmax<float> op("localOpName", axis);
		return op.compute(a);
	}
	
	tensor<double> logsoftmax(tensor<double> &a, int axis = 1) {
		LogSoftmax<double> op("localOpName", axis);
		return op.compute(a);
	}
	
	tensor<float> lpnormalization(tensor<float> &a, int p = 2, int axis = -1) {
	  LpNormalization<float> op("localOpName", p, axis);
	  return op.compute(a);
	}
	
	tensor<double> lpnormalization(tensor<double> &a, int p = 2, int axis = -1) {
	  LpNormalization<double> op("localOpName", p, axis);
	  return op.compute(a);
	}
	
	tensor<int> matmulinteger(tensor<int> &a, tensor<int> &b, tensor<int> &c, tensor<int> &d) {
		MatMulInteger<int> op;
		return op.compute(a, b, c, d);
	}
	
	tensor<float> min(std::vector<tensor<float>> a) {
	  Min<float> op;
	  return op.compute(a);
	}
	
	tensor<double> min(std::vector<tensor<double>> a) {
	  Min<double> op;
	  return op.compute(a);
	}
	
	tensor<float> mean(std::vector<tensor<float>> a) {
	  Mean<float> op;
	  return op.compute(a);
	}
	
	tensor<double> mean(std::vector<tensor<double>> a) {
	  Mean<double> op;
	  return op.compute(a);
	}
	
	tensor<float> max(std::vector<tensor<float>> a) {
	  Max<float> op;
	  return op.compute(a);
	}
	
	tensor<double> max(std::vector<tensor<double>> a) {
	  Max<double> op;
	  return op.compute(a);
	}
	
	tensor<double> mod(tensor<double> &a, tensor<double> &b, int fmod_flag = 0) {
		Mod<double> op("localOpName", fmod_flag);
		return op.compute(a, b);
	}
	
	tensor<float> mod(tensor<float> &a, tensor<float> &b, int fmod_flag = 0) {
		Mod<float> op("localOpName", fmod_flag);
		return op.compute(a, b);
	}
	
	tensor<int> mod(tensor<int> &a, tensor<int> &b, int fmod_flag = 0) {
		Mod<int> op("localOpName", fmod_flag);
		return op.compute(a, b);
	}
	
	tensor<float> thresholded_relu(tensor<float> &a) {
	  ThresholdedRelu<float> op;
	  return op.compute(a);
	}
	
	tensor<double> thresholded_relu(tensor<double> &a) {
	  ThresholdedRelu<double> op;
	  return op.compute(a);
	}
	
	tensor<double> neg(tensor<double> &a) {
		Neg<double> op;
		return op.compute(a);
	}
	
	tensor<float> neg(tensor<float> &a) {
		Neg<float> op;
		return op.compute(a);
	}
	
	tensor<int> neg(tensor<int> &a) {
		Neg<int> op;
		return op.compute(a);
	}
	
	tensor<bool> logical_not(tensor<bool> &a) {
	  Not<bool, bool> op;
	  return op.compute(a);
	}
	
	tensor<bool> logical_not(tensor<int> &a) {
	  Not<bool, int> op;
	  return op.compute(a);
	}
	
	tensor<bool> logical_not(tensor<float> &a) {
	  Not<bool, float> op;
	  return op.compute(a);
	}
	
	tensor<bool> logical_not(tensor<double> &a) {
	  Not<bool, double> op;
	  return op.compute(a);
	}
	
	tensor<float> tan(tensor<float> &a) {
	  Tan<float> op;
	  return op.compute(a);
	}
	
	tensor<double> tan(tensor<double> &a) {
	  Tan<double> op;
	  return op.compute(a);
	}
	
	tensor<float> tanh(tensor<float> &a) {
	  Tanh<float> op;
	  return op.compute(a);
	}
	
	tensor<double> tanh(tensor<double> &a) {
	  Tanh<double> op;
	  return op.compute(a);
	}
	
	tensor<double> transpose(tensor<double> &a) {
		Transpose<double> op;
		return op.compute(a);
	}
	
	tensor<float> transpose(tensor<float> &a) {
		Transpose<float> op;
		return op.compute(a);
	}
	
	tensor<int> transpose(tensor<int> &a) {
		Transpose<int> op;
		return op.compute(a);
	}
	
	tensor<bool> transpose(tensor<bool> &a) {
		Transpose<bool> op;
		return op.compute(a);
	}
	
	tensor<float> sigmoid(tensor<float> &a) {
		Sigmoid<float> op;
		return op.compute(a);
	}
	
	tensor<float> sign(tensor<float> &a) {
		Sign<float> op;
		return op.compute(a);
	}
	
	tensor<double> sign(tensor<double> &a) {
		Sign<double> op;
		return op.compute(a);
	}
	
	tensor<float> sin(tensor<float> &a) {
		Sin<float> op;
		return op.compute(a);
	}
	
	tensor<double> sin(tensor<double> &a) {
		Sin<double> op;
		return op.compute(a);
	}
	
	tensor<float> sinh(tensor<float> &a) {
		Sinh<float> op;
		return op.compute(a);
	}
	
	tensor<double> sinh(tensor<double> &a) {
		Sinh<double> op;
		return op.compute(a);
	}
	
	tensor<float> softplus(tensor<float> &a) {
		Softplus<float> op;
		return op.compute(a);
	}
	
	tensor<double> softplus(tensor<double> &a) {
		Softplus<double> op;
		return op.compute(a);
	}
	
	tensor<float> softmax(tensor<float> &a, int axis = 1) {
		Softmax<float> op("localOpName", axis);
		return op.compute(a);
	}
	
	tensor<double> softmax(tensor<double> &a, int axis = 1) {
		Softmax<double> op("localOpName", axis);
		return op.compute(a);
	}
	
	tensor<float> softsign(tensor<float> &a) {
		Softsign<float> op;
		return op.compute(a);
	}
	
	tensor<double> softsign(tensor<double> &a) {
		Softsign<double> op;
		return op.compute(a);
	}
	
	tensor<float> sqrt(tensor<float> &a) {
		Sqrt<float> op;
		return op.compute(a);
	}
	
	tensor<double> sqrt(tensor<double> &a) {
		Sqrt<double> op;
		return op.compute(a);
	}
	
	tensor<float> ceil(tensor<float> &a){
	  Ceil<float> op;
	  return op.compute(a);
	}
	
	tensor<double> ceil(tensor<double> &a){
	  Ceil<double> op;
	  return op.compute(a);
	}
	
	tensor<float> clip(tensor<float> &a, float min, float max){
	  Clip<float> op;
	  return op.compute(a, min, max);
	}
	
	tensor<double> clip(tensor<double> &a, double min, double max){
	  Clip<double> op;
	  return op.compute(a, min, max);
	}
	
	tensor<float> cos(tensor<float> &a){
	  Cos<float> op;
	  return op.compute(a);
	}
	
	tensor<double> cos(tensor<double> &a){
	  Cos<double> op;
	  return op.compute(a);
	}
	
	tensor<float> cosh(tensor<float> &a){
	  Cosh<float> op;
	  return op.compute(a);
	}
	
	tensor<double> cosh(tensor<double> &a){
	  Cosh<double> op;
	  return op.compute(a);
	}
	
	tensor<double> slice(tensor<double> &a, 
		             tensor<size_t> &start, 
			     tensor<size_t> &end, 
			     tensor<int> axes = NULL_TENSOR<int>, 
			     tensor<size_t> steps = NULL_TENSOR<size_t>) {
		Slice<double> op;
		return op.compute(a, start, end, axes, steps);
	}
	
	tensor<float> slice(tensor<float> &a, 
		             tensor<size_t> &start, 
			     tensor<size_t> &end, 
			     tensor<int> axes = NULL_TENSOR<int>, 
			     tensor<size_t> steps = NULL_TENSOR<size_t>) {
		Slice<float> op;
		return op.compute(a, start, end, axes, steps);
	}
	
	tensor<int> slice(tensor<int> &a, 
		             tensor<size_t> &start, 
			     tensor<size_t> &end, 
			     tensor<int> axes = NULL_TENSOR<int>, 
			     tensor<size_t> steps = NULL_TENSOR<size_t>) {
		Slice<int> op;
		return op.compute(a, start, end, axes, steps);
	}
	
	tensor<bool> slice(tensor<bool> &a, 
		             tensor<size_t> &start, 
			     tensor<size_t> &end, 
			     tensor<int> axes = NULL_TENSOR<int>, 
			     tensor<size_t> steps = NULL_TENSOR<size_t>) {
		Slice<bool> op;
		return op.compute(a, start, end, axes, steps);
	}
	
	tensor<float> prelu(tensor<float> &x, tensor<float> &slope) {
	  PRelu<float> op("localOpName");
	  return op.compute(x, slope);
	}
	
	
	tensor<double> prelu(tensor<double> &x, tensor<double> &slope) {
	  PRelu<double> op("localOpName");
	  return op.compute(x, slope);
	}
	
	
	tensor<int> prelu(tensor<int> &x, tensor<int> &slope) {
	  PRelu<int> op("localOpName");
	  return op.compute(x, slope);
	}
	
	
	
}
