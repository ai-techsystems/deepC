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
	tensor<double> sub(tensor<double> &a, tensor<double> &b) {
		Sub<double> op;
		return op.compute(a, b);
	}
	
	tensor<double> sub(tensor<double> &a, tensor<float> &b) {
		tensor<double> double_b = b.asType<double>();
		Sub<double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> sub(tensor<double> &a, tensor<int> &b) {
		tensor<double> double_b = b.asType<double>();
		Sub<double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> sub(tensor<double> &a, tensor<bool> &b) {
		tensor<double> double_b = b.asType<double>();
		Sub<double> op;
		return op.compute(a, double_b);
	}
	
	tensor<double> sub(tensor<float> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Sub<double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> sub(tensor<int> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Sub<double> op;
		return op.compute(double_a, b);
	}
	
	tensor<double> sub(tensor<bool> &a, tensor<double> &b) {
		tensor<double> double_a = a.asType<double>();
		Sub<double> op;
		return op.compute(double_a, b);
	}
	
	tensor<float> sub(tensor<float> &a, tensor<float> &b) {
		Sub<float> op;
		return op.compute(a, b);
	}
	
	tensor<float> sub(tensor<float> &a, tensor<int> &b) {
		tensor<float> float_b = b.asType<float>();
		Sub<float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> sub(tensor<float> &a, tensor<bool> &b) {
		tensor<float> float_b = b.asType<float>();
		Sub<float> op;
		return op.compute(a, float_b);
	}
	
	tensor<float> sub(tensor<int> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Sub<float> op;
		return op.compute(float_a, b);
	}
	
	tensor<float> sub(tensor<bool> &a, tensor<float> &b) {
		tensor<float> float_a = a.asType<float>();
		Sub<float> op;
		return op.compute(float_a, b);
	}
	
	tensor<int> sub(tensor<int> &a, tensor<int> &b) {
		Sub<int> op;
		return op.compute(a, b);
	}
	
	tensor<int> sub(tensor<int> &a, tensor<bool> &b) {
		tensor<int> int_b = b.asType<int>();
		Sub<int> op;
		return op.compute(a, int_b);
	}
	
	tensor<int> sub(tensor<bool> &a, tensor<int> &b) {
		tensor<int> int_a = a.asType<int>();
		Sub<int> op;
		return op.compute(int_a, b);
	}
	
	tensor<bool> sub(tensor<bool> &a, tensor<bool> &b) {
		Sub<bool> op;
		return op.compute(a, b);
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
	
	tensor<bool> logical_and(tensor<double> &a, tensor<double> &b) {
		And<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_and(tensor<float> &a, tensor<float> &b) {
		And<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_and(tensor<bool> &a, tensor<bool> &b) {
		And<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_and(tensor<int> &a, tensor<int> &b) {
		And<bool, int> op;
		return op.compute(a, b);
	}
	
	tensor<float> dequantize_linear(tensor<int> &a, tensor<float> &b, tensor<int> &c) {
		DequantizeLinear<float, int> op;
		return op.compute(a, b, c);
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
	
	tensor<double> dropout(tensor<double> &a, float ratio = 0.5) {
		Dropout<double> op("localOpName", ratio);
		return op.compute(a);
	}
	
	tensor<float> dropout(tensor<float> &a, float ratio = 0.5) {
		Dropout<float> op("localOpName", ratio);
		return op.compute(a);
	}
	
	tensor<int> floor_div(tensor<double> &a, tensor<double> &b) {
		FloorDiv<int, double> op;
		return op.compute(a, b);
	}
	
	tensor<int> floor_div(tensor<float> &a, tensor<float> &b) {
		FloorDiv<int, float> op;
		return op.compute(a, b);
	}
	
	tensor<int> floor_div(tensor<int> &a, tensor<int> &b) {
		FloorDiv<int, int> op;
		return op.compute(a, b);
	}
	
	tensor<float> true_div(tensor<double> &a, tensor<double> &b) {
		TrueDiv<float, double> op;
		return op.compute(a, b);
	}
	
	tensor<float> true_div(tensor<float> &a, tensor<float> &b) {
		TrueDiv<float, float> op;
		return op.compute(a, b);
	}
	
	tensor<float> true_div(tensor<int> &a, tensor<int> &b) {
		TrueDiv<float, int> op;
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
	
	tensor<bool> equal(tensor<double> &a, tensor<double> &b) {
		Equal<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> equal(tensor<float> &a, tensor<float> &b) {
		Equal<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> equal(tensor<bool> &a, tensor<bool> &b) {
		Equal<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> equal(tensor<int> &a, tensor<int> &b) {
		Equal<bool, int> op;
		return op.compute(a, b);
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
	
	tensor<bool> greater(tensor<int> &a, tensor<int> &b) {
	  Greater<bool, int> op;
	  return op.compute(a, b);
	}
	
	tensor<bool> greater(tensor<float> &a, tensor<float> &b) {
	  Greater<bool, float> op;
	  return op.compute(a, b);
	}
	
	tensor<bool> greater(tensor<double> &a, tensor<double> &b) {
	  Greater<bool, double> op;
	  return op.compute(a, b);
	}
	
	tensor<bool> greater_equal(tensor<int> &a, tensor<int> &b) {
	  GreaterEqual<bool, int> op;
	  return op.compute(a, b);
	}
	
	tensor<bool> greater_equal(tensor<float> &a, tensor<float> &b) {
	  GreaterEqual<bool, float> op;
	  return op.compute(a, b);
	}
	
	tensor<bool> greater_equal(tensor<double> &a, tensor<double> &b) {
	  GreaterEqual<bool, double> op;
	  return op.compute(a, b);
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
	
	tensor<bool> less(tensor<int> &a, tensor<int> &b) {
	  Less<bool, int> op;
	  return op.compute(a, b);
	}
	
	tensor<bool> less(tensor<float> &a, tensor<float> &b) {
	  Less<bool, float> op;
	  return op.compute(a, b);
	}
	
	tensor<bool> less(tensor<double> &a, tensor<double> &b) {
	  Less<bool, double> op;
	  return op.compute(a, b);
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
	
	tensor<bool> less_equal(tensor<int> &a, tensor<int> &b) {
	  LessEqual<bool, int> op;
	  return op.compute(a, b);
	}
	
	tensor<bool> less_equal(tensor<float> &a, tensor<float> &b) {
	  LessEqual<bool, float> op;
	  return op.compute(a, b);
	}
	
	tensor<bool> less_equal(tensor<double> &a, tensor<double> &b) {
	  LessEqual<bool, double> op;
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
	
	tensor<int> mul(tensor<int> &a, tensor<int> &b) {
	  Mul<int> op;
	  return op.compute(a, b);
	}
	
	tensor<float> mul(tensor<float> &a, tensor<float> &b) {
	  Mul<float> op;
	  return op.compute(a, b);
	}
	
	tensor<double> mul(tensor<double> &a, tensor<double> &b) {
	  Mul<double> op;
	  return op.compute(a, b);
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
	
	tensor<bool> not_equal(tensor<double> &a, tensor<double> &b) {
		NotEqual<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> not_equal(tensor<float> &a, tensor<float> &b) {
		NotEqual<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> not_equal(tensor<bool> &a, tensor<bool> &b) {
		NotEqual<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> not_equal(tensor<int> &a, tensor<int> &b) {
		NotEqual<bool, int> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_not(tensor<double> &a) {
		Not<bool, double> op;
		return op.compute(a);
	}
	
	tensor<bool> logical_not(tensor<float> &a) {
		Not<bool, float> op;
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
	
	tensor<bool> logical_or(tensor<double> &a, tensor<double> &b) {
		Or<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_or(tensor<float> &a, tensor<float> &b) {
		Or<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_or(tensor<bool> &a, tensor<bool> &b) {
		Or<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_or(tensor<int> &a, tensor<int> &b) {
		Or<bool, int> op;
		return op.compute(a, b);
	}
	
	tensor<double> pow(tensor<double> &a, tensor<double> &b) {
		Pow<double> op;
		return op.compute(a, b);
	}
	
	tensor<float> pow(tensor<float> &a, tensor<float> &b) {
		Pow<float> op;
		return op.compute(a, b);
	}
	
	tensor<int> pow(tensor<int> &a, tensor<int> &b) {
		Pow<int> op;
		return op.compute(a, b);
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
	
	tensor<bool> logical_xor(tensor<double> &a, tensor<double> &b) {
		Xor<bool, double> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_xor(tensor<float> &a, tensor<float> &b) {
		Xor<bool, float> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_xor(tensor<bool> &a, tensor<bool> &b) {
		Xor<bool, bool> op;
		return op.compute(a, b);
	}
	
	tensor<bool> logical_xor(tensor<int> &a, tensor<int> &b) {
		Xor<bool, int> op;
		return op.compute(a, b);
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
	
	
	
}
