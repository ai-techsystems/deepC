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
	extern tensor<double> \
		add(tensor<double> &a, tensor<double> &b) ;
	extern tensor<double> \
		add(tensor<double> &a, tensor<float> &b) ;
	extern tensor<double> \
		add(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		add(tensor<double> &a, tensor<int> &b) ;
	extern tensor<double> \
		add(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<double> \
		add(tensor<float> &a, tensor<double> &b) ;
	extern tensor<double> \
		add(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<double> \
		add(tensor<int> &a, tensor<double> &b) ;
	extern tensor<double> \
		add(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<float> \
		add(tensor<float> &a, tensor<float> &b) ;
	extern tensor<float> \
		add(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<float> \
		add(tensor<float> &a, tensor<int> &b) ;
	extern tensor<float> \
		add(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<float> \
		add(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<float> \
		add(tensor<int> &a, tensor<float> &b) ;
	extern tensor<float> \
		add(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<size_t> \
		add(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		add(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<size_t> \
		add(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<size_t> \
		add(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		add(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		add(tensor<int> &a, tensor<int> &b) ;
	extern tensor<int> \
		add(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<int> \
		add(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		add(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<double> \
		add(tensor<double> &a, double b) ;
	extern tensor<double> \
		add(tensor<double> &a, float b) ;
	extern tensor<double> \
		add(tensor<double> &a, size_t b) ;
	extern tensor<double> \
		add(tensor<double> &a, int b) ;
	extern tensor<double> \
		add(tensor<double> &a, bool b) ;
	extern tensor<double> \
		add(tensor<float> &a, double b) ;
	extern tensor<double> \
		add(tensor<size_t> &a, double b) ;
	extern tensor<double> \
		add(tensor<int> &a, double b) ;
	extern tensor<double> \
		add(tensor<bool> &a, double b) ;
	extern tensor<float> \
		add(tensor<float> &a, float b) ;
	extern tensor<float> \
		add(tensor<float> &a, size_t b) ;
	extern tensor<float> \
		add(tensor<float> &a, int b) ;
	extern tensor<float> \
		add(tensor<float> &a, bool b) ;
	extern tensor<float> \
		add(tensor<size_t> &a, float b) ;
	extern tensor<float> \
		add(tensor<int> &a, float b) ;
	extern tensor<float> \
		add(tensor<bool> &a, float b) ;
	extern tensor<size_t> \
		add(tensor<size_t> &a, size_t b) ;
	extern tensor<size_t> \
		add(tensor<size_t> &a, int b) ;
	extern tensor<size_t> \
		add(tensor<size_t> &a, bool b) ;
	extern tensor<size_t> \
		add(tensor<int> &a, size_t b) ;
	extern tensor<size_t> \
		add(tensor<bool> &a, size_t b) ;
	extern tensor<int> \
		add(tensor<int> &a, int b) ;
	extern tensor<int> \
		add(tensor<int> &a, bool b) ;
	extern tensor<int> \
		add(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		add(tensor<bool> &a, bool b) ;
	extern tensor<double> \
		add(double a, tensor<double> &b) ;
	extern tensor<double> \
		add(double a, tensor<float> &b) ;
	extern tensor<double> \
		add(double a, tensor<size_t> &b) ;
	extern tensor<double> \
		add(double a, tensor<int> &b) ;
	extern tensor<double> \
		add(double a, tensor<bool> &b) ;
	extern tensor<double> \
		add(float a, tensor<double> &b) ;
	extern tensor<double> \
		add(size_t a, tensor<double> &b) ;
	extern tensor<double> \
		add(int a, tensor<double> &b) ;
	extern tensor<double> \
		add(bool a, tensor<double> &b) ;
	extern tensor<float> \
		add(float a, tensor<float> &b) ;
	extern tensor<float> \
		add(float a, tensor<size_t> &b) ;
	extern tensor<float> \
		add(float a, tensor<int> &b) ;
	extern tensor<float> \
		add(float a, tensor<bool> &b) ;
	extern tensor<float> \
		add(size_t a, tensor<float> &b) ;
	extern tensor<float> \
		add(int a, tensor<float> &b) ;
	extern tensor<float> \
		add(bool a, tensor<float> &b) ;
	extern tensor<size_t> \
		add(size_t a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		add(size_t a, tensor<int> &b) ;
	extern tensor<size_t> \
		add(size_t a, tensor<bool> &b) ;
	extern tensor<size_t> \
		add(int a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		add(bool a, tensor<size_t> &b) ;
	extern tensor<int> \
		add(int a, tensor<int> &b) ;
	extern tensor<int> \
		add(int a, tensor<bool> &b) ;
	extern tensor<int> \
		add(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		add(bool a, tensor<bool> &b) ;
	extern double \
		add(double a, double b) ;
	extern double \
		add(double a, float b) ;
	extern double \
		add(double a, size_t b) ;
	extern double \
		add(double a, int b) ;
	extern double \
		add(double a, bool b) ;
	extern double \
		add(float a, double b) ;
	extern double \
		add(size_t a, double b) ;
	extern double \
		add(int a, double b) ;
	extern double \
		add(bool a, double b) ;
	extern float \
		add(float a, float b) ;
	extern float \
		add(float a, size_t b) ;
	extern float \
		add(float a, int b) ;
	extern float \
		add(float a, bool b) ;
	extern float \
		add(size_t a, float b) ;
	extern float \
		add(int a, float b) ;
	extern float \
		add(bool a, float b) ;
	extern size_t \
		add(size_t a, size_t b) ;
	extern size_t \
		add(size_t a, int b) ;
	extern size_t \
		add(size_t a, bool b) ;
	extern size_t \
		add(int a, size_t b) ;
	extern size_t \
		add(bool a, size_t b) ;
	extern int \
		add(int a, int b) ;
	extern int \
		add(int a, bool b) ;
	extern int \
		add(bool a, int b) ;
	extern bool \
		add(bool a, bool b) ;
	extern tensor<double> \
		sub(tensor<double> &a, tensor<double> &b) ;
	extern tensor<double> \
		sub(tensor<double> &a, tensor<float> &b) ;
	extern tensor<double> \
		sub(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		sub(tensor<double> &a, tensor<int> &b) ;
	extern tensor<double> \
		sub(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<double> \
		sub(tensor<float> &a, tensor<double> &b) ;
	extern tensor<double> \
		sub(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<double> \
		sub(tensor<int> &a, tensor<double> &b) ;
	extern tensor<double> \
		sub(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<float> \
		sub(tensor<float> &a, tensor<float> &b) ;
	extern tensor<float> \
		sub(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<float> \
		sub(tensor<float> &a, tensor<int> &b) ;
	extern tensor<float> \
		sub(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<float> \
		sub(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<float> \
		sub(tensor<int> &a, tensor<float> &b) ;
	extern tensor<float> \
		sub(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<size_t> \
		sub(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		sub(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<size_t> \
		sub(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<size_t> \
		sub(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		sub(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		sub(tensor<int> &a, tensor<int> &b) ;
	extern tensor<int> \
		sub(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<int> \
		sub(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		sub(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<double> \
		sub(tensor<double> &a, double b) ;
	extern tensor<double> \
		sub(tensor<double> &a, float b) ;
	extern tensor<double> \
		sub(tensor<double> &a, size_t b) ;
	extern tensor<double> \
		sub(tensor<double> &a, int b) ;
	extern tensor<double> \
		sub(tensor<double> &a, bool b) ;
	extern tensor<double> \
		sub(tensor<float> &a, double b) ;
	extern tensor<double> \
		sub(tensor<size_t> &a, double b) ;
	extern tensor<double> \
		sub(tensor<int> &a, double b) ;
	extern tensor<double> \
		sub(tensor<bool> &a, double b) ;
	extern tensor<float> \
		sub(tensor<float> &a, float b) ;
	extern tensor<float> \
		sub(tensor<float> &a, size_t b) ;
	extern tensor<float> \
		sub(tensor<float> &a, int b) ;
	extern tensor<float> \
		sub(tensor<float> &a, bool b) ;
	extern tensor<float> \
		sub(tensor<size_t> &a, float b) ;
	extern tensor<float> \
		sub(tensor<int> &a, float b) ;
	extern tensor<float> \
		sub(tensor<bool> &a, float b) ;
	extern tensor<size_t> \
		sub(tensor<size_t> &a, size_t b) ;
	extern tensor<size_t> \
		sub(tensor<size_t> &a, int b) ;
	extern tensor<size_t> \
		sub(tensor<size_t> &a, bool b) ;
	extern tensor<size_t> \
		sub(tensor<int> &a, size_t b) ;
	extern tensor<size_t> \
		sub(tensor<bool> &a, size_t b) ;
	extern tensor<int> \
		sub(tensor<int> &a, int b) ;
	extern tensor<int> \
		sub(tensor<int> &a, bool b) ;
	extern tensor<int> \
		sub(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		sub(tensor<bool> &a, bool b) ;
	extern tensor<double> \
		sub(double a, tensor<double> &b) ;
	extern tensor<double> \
		sub(double a, tensor<float> &b) ;
	extern tensor<double> \
		sub(double a, tensor<size_t> &b) ;
	extern tensor<double> \
		sub(double a, tensor<int> &b) ;
	extern tensor<double> \
		sub(double a, tensor<bool> &b) ;
	extern tensor<double> \
		sub(float a, tensor<double> &b) ;
	extern tensor<double> \
		sub(size_t a, tensor<double> &b) ;
	extern tensor<double> \
		sub(int a, tensor<double> &b) ;
	extern tensor<double> \
		sub(bool a, tensor<double> &b) ;
	extern tensor<float> \
		sub(float a, tensor<float> &b) ;
	extern tensor<float> \
		sub(float a, tensor<size_t> &b) ;
	extern tensor<float> \
		sub(float a, tensor<int> &b) ;
	extern tensor<float> \
		sub(float a, tensor<bool> &b) ;
	extern tensor<float> \
		sub(size_t a, tensor<float> &b) ;
	extern tensor<float> \
		sub(int a, tensor<float> &b) ;
	extern tensor<float> \
		sub(bool a, tensor<float> &b) ;
	extern tensor<size_t> \
		sub(size_t a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		sub(size_t a, tensor<int> &b) ;
	extern tensor<size_t> \
		sub(size_t a, tensor<bool> &b) ;
	extern tensor<size_t> \
		sub(int a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		sub(bool a, tensor<size_t> &b) ;
	extern tensor<int> \
		sub(int a, tensor<int> &b) ;
	extern tensor<int> \
		sub(int a, tensor<bool> &b) ;
	extern tensor<int> \
		sub(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		sub(bool a, tensor<bool> &b) ;
	extern double \
		sub(double a, double b) ;
	extern double \
		sub(double a, float b) ;
	extern double \
		sub(double a, size_t b) ;
	extern double \
		sub(double a, int b) ;
	extern double \
		sub(double a, bool b) ;
	extern double \
		sub(float a, double b) ;
	extern double \
		sub(size_t a, double b) ;
	extern double \
		sub(int a, double b) ;
	extern double \
		sub(bool a, double b) ;
	extern float \
		sub(float a, float b) ;
	extern float \
		sub(float a, size_t b) ;
	extern float \
		sub(float a, int b) ;
	extern float \
		sub(float a, bool b) ;
	extern float \
		sub(size_t a, float b) ;
	extern float \
		sub(int a, float b) ;
	extern float \
		sub(bool a, float b) ;
	extern size_t \
		sub(size_t a, size_t b) ;
	extern size_t \
		sub(size_t a, int b) ;
	extern size_t \
		sub(size_t a, bool b) ;
	extern size_t \
		sub(int a, size_t b) ;
	extern size_t \
		sub(bool a, size_t b) ;
	extern int \
		sub(int a, int b) ;
	extern int \
		sub(int a, bool b) ;
	extern int \
		sub(bool a, int b) ;
	extern bool \
		sub(bool a, bool b) ;
	extern tensor<double> \
		mul(tensor<double> &a, tensor<double> &b) ;
	extern tensor<double> \
		mul(tensor<double> &a, tensor<float> &b) ;
	extern tensor<double> \
		mul(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		mul(tensor<double> &a, tensor<int> &b) ;
	extern tensor<double> \
		mul(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<double> \
		mul(tensor<float> &a, tensor<double> &b) ;
	extern tensor<double> \
		mul(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<double> \
		mul(tensor<int> &a, tensor<double> &b) ;
	extern tensor<double> \
		mul(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<float> \
		mul(tensor<float> &a, tensor<float> &b) ;
	extern tensor<float> \
		mul(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<float> \
		mul(tensor<float> &a, tensor<int> &b) ;
	extern tensor<float> \
		mul(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<float> \
		mul(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<float> \
		mul(tensor<int> &a, tensor<float> &b) ;
	extern tensor<float> \
		mul(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<size_t> \
		mul(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		mul(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<size_t> \
		mul(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<size_t> \
		mul(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		mul(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		mul(tensor<int> &a, tensor<int> &b) ;
	extern tensor<int> \
		mul(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<int> \
		mul(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		mul(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<double> \
		mul(tensor<double> &a, double b) ;
	extern tensor<double> \
		mul(tensor<double> &a, float b) ;
	extern tensor<double> \
		mul(tensor<double> &a, size_t b) ;
	extern tensor<double> \
		mul(tensor<double> &a, int b) ;
	extern tensor<double> \
		mul(tensor<double> &a, bool b) ;
	extern tensor<double> \
		mul(tensor<float> &a, double b) ;
	extern tensor<double> \
		mul(tensor<size_t> &a, double b) ;
	extern tensor<double> \
		mul(tensor<int> &a, double b) ;
	extern tensor<double> \
		mul(tensor<bool> &a, double b) ;
	extern tensor<float> \
		mul(tensor<float> &a, float b) ;
	extern tensor<float> \
		mul(tensor<float> &a, size_t b) ;
	extern tensor<float> \
		mul(tensor<float> &a, int b) ;
	extern tensor<float> \
		mul(tensor<float> &a, bool b) ;
	extern tensor<float> \
		mul(tensor<size_t> &a, float b) ;
	extern tensor<float> \
		mul(tensor<int> &a, float b) ;
	extern tensor<float> \
		mul(tensor<bool> &a, float b) ;
	extern tensor<size_t> \
		mul(tensor<size_t> &a, size_t b) ;
	extern tensor<size_t> \
		mul(tensor<size_t> &a, int b) ;
	extern tensor<size_t> \
		mul(tensor<size_t> &a, bool b) ;
	extern tensor<size_t> \
		mul(tensor<int> &a, size_t b) ;
	extern tensor<size_t> \
		mul(tensor<bool> &a, size_t b) ;
	extern tensor<int> \
		mul(tensor<int> &a, int b) ;
	extern tensor<int> \
		mul(tensor<int> &a, bool b) ;
	extern tensor<int> \
		mul(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		mul(tensor<bool> &a, bool b) ;
	extern tensor<double> \
		mul(double a, tensor<double> &b) ;
	extern tensor<double> \
		mul(double a, tensor<float> &b) ;
	extern tensor<double> \
		mul(double a, tensor<size_t> &b) ;
	extern tensor<double> \
		mul(double a, tensor<int> &b) ;
	extern tensor<double> \
		mul(double a, tensor<bool> &b) ;
	extern tensor<double> \
		mul(float a, tensor<double> &b) ;
	extern tensor<double> \
		mul(size_t a, tensor<double> &b) ;
	extern tensor<double> \
		mul(int a, tensor<double> &b) ;
	extern tensor<double> \
		mul(bool a, tensor<double> &b) ;
	extern tensor<float> \
		mul(float a, tensor<float> &b) ;
	extern tensor<float> \
		mul(float a, tensor<size_t> &b) ;
	extern tensor<float> \
		mul(float a, tensor<int> &b) ;
	extern tensor<float> \
		mul(float a, tensor<bool> &b) ;
	extern tensor<float> \
		mul(size_t a, tensor<float> &b) ;
	extern tensor<float> \
		mul(int a, tensor<float> &b) ;
	extern tensor<float> \
		mul(bool a, tensor<float> &b) ;
	extern tensor<size_t> \
		mul(size_t a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		mul(size_t a, tensor<int> &b) ;
	extern tensor<size_t> \
		mul(size_t a, tensor<bool> &b) ;
	extern tensor<size_t> \
		mul(int a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		mul(bool a, tensor<size_t> &b) ;
	extern tensor<int> \
		mul(int a, tensor<int> &b) ;
	extern tensor<int> \
		mul(int a, tensor<bool> &b) ;
	extern tensor<int> \
		mul(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		mul(bool a, tensor<bool> &b) ;
	extern double \
		mul(double a, double b) ;
	extern double \
		mul(double a, float b) ;
	extern double \
		mul(double a, size_t b) ;
	extern double \
		mul(double a, int b) ;
	extern double \
		mul(double a, bool b) ;
	extern double \
		mul(float a, double b) ;
	extern double \
		mul(size_t a, double b) ;
	extern double \
		mul(int a, double b) ;
	extern double \
		mul(bool a, double b) ;
	extern float \
		mul(float a, float b) ;
	extern float \
		mul(float a, size_t b) ;
	extern float \
		mul(float a, int b) ;
	extern float \
		mul(float a, bool b) ;
	extern float \
		mul(size_t a, float b) ;
	extern float \
		mul(int a, float b) ;
	extern float \
		mul(bool a, float b) ;
	extern size_t \
		mul(size_t a, size_t b) ;
	extern size_t \
		mul(size_t a, int b) ;
	extern size_t \
		mul(size_t a, bool b) ;
	extern size_t \
		mul(int a, size_t b) ;
	extern size_t \
		mul(bool a, size_t b) ;
	extern int \
		mul(int a, int b) ;
	extern int \
		mul(int a, bool b) ;
	extern int \
		mul(bool a, int b) ;
	extern bool \
		mul(bool a, bool b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, double b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, float b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, size_t b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, int b) ;
	extern tensor<int> \
		floor_div(tensor<double> &a, bool b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, double b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, double b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, double b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, double b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, float b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, size_t b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, int b) ;
	extern tensor<int> \
		floor_div(tensor<float> &a, bool b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, float b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, float b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, float b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, size_t b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, int b) ;
	extern tensor<int> \
		floor_div(tensor<size_t> &a, bool b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, size_t b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, size_t b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, int b) ;
	extern tensor<int> \
		floor_div(tensor<int> &a, bool b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, int b) ;
	extern tensor<int> \
		floor_div(tensor<bool> &a, bool b) ;
	extern tensor<int> \
		floor_div(double a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(double a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(double a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(double a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(double a, tensor<bool> &b) ;
	extern tensor<int> \
		floor_div(float a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(size_t a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(int a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(bool a, tensor<double> &b) ;
	extern tensor<int> \
		floor_div(float a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(float a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(float a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(float a, tensor<bool> &b) ;
	extern tensor<int> \
		floor_div(size_t a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(int a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(bool a, tensor<float> &b) ;
	extern tensor<int> \
		floor_div(size_t a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(size_t a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(size_t a, tensor<bool> &b) ;
	extern tensor<int> \
		floor_div(int a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(bool a, tensor<size_t> &b) ;
	extern tensor<int> \
		floor_div(int a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(int a, tensor<bool> &b) ;
	extern tensor<int> \
		floor_div(bool a, tensor<int> &b) ;
	extern tensor<int> \
		floor_div(bool a, tensor<bool> &b) ;
	extern int \
		floor_div(double a, double b) ;
	extern int \
		floor_div(double a, float b) ;
	extern int \
		floor_div(double a, size_t b) ;
	extern int \
		floor_div(double a, int b) ;
	extern int \
		floor_div(double a, bool b) ;
	extern int \
		floor_div(float a, double b) ;
	extern int \
		floor_div(size_t a, double b) ;
	extern int \
		floor_div(int a, double b) ;
	extern int \
		floor_div(bool a, double b) ;
	extern int \
		floor_div(float a, float b) ;
	extern int \
		floor_div(float a, size_t b) ;
	extern int \
		floor_div(float a, int b) ;
	extern int \
		floor_div(float a, bool b) ;
	extern int \
		floor_div(size_t a, float b) ;
	extern int \
		floor_div(int a, float b) ;
	extern int \
		floor_div(bool a, float b) ;
	extern int \
		floor_div(size_t a, size_t b) ;
	extern int \
		floor_div(size_t a, int b) ;
	extern int \
		floor_div(size_t a, bool b) ;
	extern int \
		floor_div(int a, size_t b) ;
	extern int \
		floor_div(bool a, size_t b) ;
	extern int \
		floor_div(int a, int b) ;
	extern int \
		floor_div(int a, bool b) ;
	extern int \
		floor_div(bool a, int b) ;
	extern int \
		floor_div(bool a, bool b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, double b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, float b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, size_t b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, int b) ;
	extern tensor<double> \
		true_div(tensor<double> &a, bool b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, double b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, double b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, double b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, double b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, float b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, size_t b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, int b) ;
	extern tensor<double> \
		true_div(tensor<float> &a, bool b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, float b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, float b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, float b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, size_t b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, int b) ;
	extern tensor<double> \
		true_div(tensor<size_t> &a, bool b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, size_t b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, size_t b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, int b) ;
	extern tensor<double> \
		true_div(tensor<int> &a, bool b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, int b) ;
	extern tensor<double> \
		true_div(tensor<bool> &a, bool b) ;
	extern tensor<double> \
		true_div(double a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(double a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(double a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(double a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(double a, tensor<bool> &b) ;
	extern tensor<double> \
		true_div(float a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(size_t a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(int a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(bool a, tensor<double> &b) ;
	extern tensor<double> \
		true_div(float a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(float a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(float a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(float a, tensor<bool> &b) ;
	extern tensor<double> \
		true_div(size_t a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(int a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(bool a, tensor<float> &b) ;
	extern tensor<double> \
		true_div(size_t a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(size_t a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(size_t a, tensor<bool> &b) ;
	extern tensor<double> \
		true_div(int a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(bool a, tensor<size_t> &b) ;
	extern tensor<double> \
		true_div(int a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(int a, tensor<bool> &b) ;
	extern tensor<double> \
		true_div(bool a, tensor<int> &b) ;
	extern tensor<double> \
		true_div(bool a, tensor<bool> &b) ;
	extern double \
		true_div(double a, double b) ;
	extern double \
		true_div(double a, float b) ;
	extern double \
		true_div(double a, size_t b) ;
	extern double \
		true_div(double a, int b) ;
	extern double \
		true_div(double a, bool b) ;
	extern double \
		true_div(float a, double b) ;
	extern double \
		true_div(size_t a, double b) ;
	extern double \
		true_div(int a, double b) ;
	extern double \
		true_div(bool a, double b) ;
	extern double \
		true_div(float a, float b) ;
	extern double \
		true_div(float a, size_t b) ;
	extern double \
		true_div(float a, int b) ;
	extern double \
		true_div(float a, bool b) ;
	extern double \
		true_div(size_t a, float b) ;
	extern double \
		true_div(int a, float b) ;
	extern double \
		true_div(bool a, float b) ;
	extern double \
		true_div(size_t a, size_t b) ;
	extern double \
		true_div(size_t a, int b) ;
	extern double \
		true_div(size_t a, bool b) ;
	extern double \
		true_div(int a, size_t b) ;
	extern double \
		true_div(bool a, size_t b) ;
	extern double \
		true_div(int a, int b) ;
	extern double \
		true_div(int a, bool b) ;
	extern double \
		true_div(bool a, int b) ;
	extern double \
		true_div(bool a, bool b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, tensor<double> &b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, tensor<float> &b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, tensor<int> &b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<double> \
		remainder(tensor<float> &a, tensor<double> &b) ;
	extern tensor<double> \
		remainder(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<double> \
		remainder(tensor<int> &a, tensor<double> &b) ;
	extern tensor<double> \
		remainder(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<float> \
		remainder(tensor<float> &a, tensor<float> &b) ;
	extern tensor<float> \
		remainder(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<float> \
		remainder(tensor<float> &a, tensor<int> &b) ;
	extern tensor<float> \
		remainder(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<float> \
		remainder(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<float> \
		remainder(tensor<int> &a, tensor<float> &b) ;
	extern tensor<float> \
		remainder(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<size_t> \
		remainder(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		remainder(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<size_t> \
		remainder(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<size_t> \
		remainder(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		remainder(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		remainder(tensor<int> &a, tensor<int> &b) ;
	extern tensor<int> \
		remainder(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<int> \
		remainder(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		remainder(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, double b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, float b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, size_t b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, int b) ;
	extern tensor<double> \
		remainder(tensor<double> &a, bool b) ;
	extern tensor<double> \
		remainder(tensor<float> &a, double b) ;
	extern tensor<double> \
		remainder(tensor<size_t> &a, double b) ;
	extern tensor<double> \
		remainder(tensor<int> &a, double b) ;
	extern tensor<double> \
		remainder(tensor<bool> &a, double b) ;
	extern tensor<float> \
		remainder(tensor<float> &a, float b) ;
	extern tensor<float> \
		remainder(tensor<float> &a, size_t b) ;
	extern tensor<float> \
		remainder(tensor<float> &a, int b) ;
	extern tensor<float> \
		remainder(tensor<float> &a, bool b) ;
	extern tensor<float> \
		remainder(tensor<size_t> &a, float b) ;
	extern tensor<float> \
		remainder(tensor<int> &a, float b) ;
	extern tensor<float> \
		remainder(tensor<bool> &a, float b) ;
	extern tensor<size_t> \
		remainder(tensor<size_t> &a, size_t b) ;
	extern tensor<size_t> \
		remainder(tensor<size_t> &a, int b) ;
	extern tensor<size_t> \
		remainder(tensor<size_t> &a, bool b) ;
	extern tensor<size_t> \
		remainder(tensor<int> &a, size_t b) ;
	extern tensor<size_t> \
		remainder(tensor<bool> &a, size_t b) ;
	extern tensor<int> \
		remainder(tensor<int> &a, int b) ;
	extern tensor<int> \
		remainder(tensor<int> &a, bool b) ;
	extern tensor<int> \
		remainder(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		remainder(tensor<bool> &a, bool b) ;
	extern tensor<double> \
		remainder(double a, tensor<double> &b) ;
	extern tensor<double> \
		remainder(double a, tensor<float> &b) ;
	extern tensor<double> \
		remainder(double a, tensor<size_t> &b) ;
	extern tensor<double> \
		remainder(double a, tensor<int> &b) ;
	extern tensor<double> \
		remainder(double a, tensor<bool> &b) ;
	extern tensor<double> \
		remainder(float a, tensor<double> &b) ;
	extern tensor<double> \
		remainder(size_t a, tensor<double> &b) ;
	extern tensor<double> \
		remainder(int a, tensor<double> &b) ;
	extern tensor<double> \
		remainder(bool a, tensor<double> &b) ;
	extern tensor<float> \
		remainder(float a, tensor<float> &b) ;
	extern tensor<float> \
		remainder(float a, tensor<size_t> &b) ;
	extern tensor<float> \
		remainder(float a, tensor<int> &b) ;
	extern tensor<float> \
		remainder(float a, tensor<bool> &b) ;
	extern tensor<float> \
		remainder(size_t a, tensor<float> &b) ;
	extern tensor<float> \
		remainder(int a, tensor<float> &b) ;
	extern tensor<float> \
		remainder(bool a, tensor<float> &b) ;
	extern tensor<size_t> \
		remainder(size_t a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		remainder(size_t a, tensor<int> &b) ;
	extern tensor<size_t> \
		remainder(size_t a, tensor<bool> &b) ;
	extern tensor<size_t> \
		remainder(int a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		remainder(bool a, tensor<size_t> &b) ;
	extern tensor<int> \
		remainder(int a, tensor<int> &b) ;
	extern tensor<int> \
		remainder(int a, tensor<bool> &b) ;
	extern tensor<int> \
		remainder(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		remainder(bool a, tensor<bool> &b) ;
	extern double \
		remainder(double a, double b) ;
	extern double \
		remainder(double a, float b) ;
	extern double \
		remainder(double a, size_t b) ;
	extern double \
		remainder(double a, int b) ;
	extern double \
		remainder(double a, bool b) ;
	extern double \
		remainder(float a, double b) ;
	extern double \
		remainder(size_t a, double b) ;
	extern double \
		remainder(int a, double b) ;
	extern double \
		remainder(bool a, double b) ;
	extern float \
		remainder(float a, float b) ;
	extern float \
		remainder(float a, size_t b) ;
	extern float \
		remainder(float a, int b) ;
	extern float \
		remainder(float a, bool b) ;
	extern float \
		remainder(size_t a, float b) ;
	extern float \
		remainder(int a, float b) ;
	extern float \
		remainder(bool a, float b) ;
	extern size_t \
		remainder(size_t a, size_t b) ;
	extern size_t \
		remainder(size_t a, int b) ;
	extern size_t \
		remainder(size_t a, bool b) ;
	extern size_t \
		remainder(int a, size_t b) ;
	extern size_t \
		remainder(bool a, size_t b) ;
	extern int \
		remainder(int a, int b) ;
	extern int \
		remainder(int a, bool b) ;
	extern int \
		remainder(bool a, int b) ;
	extern bool \
		remainder(bool a, bool b) ;
	extern tensor<double> \
		power(tensor<double> &a, tensor<double> &b) ;
	extern tensor<double> \
		power(tensor<double> &a, tensor<float> &b) ;
	extern tensor<double> \
		power(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<double> \
		power(tensor<double> &a, tensor<int> &b) ;
	extern tensor<double> \
		power(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<double> \
		power(tensor<float> &a, tensor<double> &b) ;
	extern tensor<double> \
		power(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<double> \
		power(tensor<int> &a, tensor<double> &b) ;
	extern tensor<double> \
		power(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<float> \
		power(tensor<float> &a, tensor<float> &b) ;
	extern tensor<float> \
		power(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<float> \
		power(tensor<float> &a, tensor<int> &b) ;
	extern tensor<float> \
		power(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<float> \
		power(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<float> \
		power(tensor<int> &a, tensor<float> &b) ;
	extern tensor<float> \
		power(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<size_t> \
		power(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		power(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<size_t> \
		power(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<size_t> \
		power(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		power(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<int> \
		power(tensor<int> &a, tensor<int> &b) ;
	extern tensor<int> \
		power(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<int> \
		power(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		power(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<double> \
		power(tensor<double> &a, double b) ;
	extern tensor<double> \
		power(tensor<double> &a, float b) ;
	extern tensor<double> \
		power(tensor<double> &a, size_t b) ;
	extern tensor<double> \
		power(tensor<double> &a, int b) ;
	extern tensor<double> \
		power(tensor<double> &a, bool b) ;
	extern tensor<double> \
		power(tensor<float> &a, double b) ;
	extern tensor<double> \
		power(tensor<size_t> &a, double b) ;
	extern tensor<double> \
		power(tensor<int> &a, double b) ;
	extern tensor<double> \
		power(tensor<bool> &a, double b) ;
	extern tensor<float> \
		power(tensor<float> &a, float b) ;
	extern tensor<float> \
		power(tensor<float> &a, size_t b) ;
	extern tensor<float> \
		power(tensor<float> &a, int b) ;
	extern tensor<float> \
		power(tensor<float> &a, bool b) ;
	extern tensor<float> \
		power(tensor<size_t> &a, float b) ;
	extern tensor<float> \
		power(tensor<int> &a, float b) ;
	extern tensor<float> \
		power(tensor<bool> &a, float b) ;
	extern tensor<size_t> \
		power(tensor<size_t> &a, size_t b) ;
	extern tensor<size_t> \
		power(tensor<size_t> &a, int b) ;
	extern tensor<size_t> \
		power(tensor<size_t> &a, bool b) ;
	extern tensor<size_t> \
		power(tensor<int> &a, size_t b) ;
	extern tensor<size_t> \
		power(tensor<bool> &a, size_t b) ;
	extern tensor<int> \
		power(tensor<int> &a, int b) ;
	extern tensor<int> \
		power(tensor<int> &a, bool b) ;
	extern tensor<int> \
		power(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		power(tensor<bool> &a, bool b) ;
	extern tensor<double> \
		power(double a, tensor<double> &b) ;
	extern tensor<double> \
		power(double a, tensor<float> &b) ;
	extern tensor<double> \
		power(double a, tensor<size_t> &b) ;
	extern tensor<double> \
		power(double a, tensor<int> &b) ;
	extern tensor<double> \
		power(double a, tensor<bool> &b) ;
	extern tensor<double> \
		power(float a, tensor<double> &b) ;
	extern tensor<double> \
		power(size_t a, tensor<double> &b) ;
	extern tensor<double> \
		power(int a, tensor<double> &b) ;
	extern tensor<double> \
		power(bool a, tensor<double> &b) ;
	extern tensor<float> \
		power(float a, tensor<float> &b) ;
	extern tensor<float> \
		power(float a, tensor<size_t> &b) ;
	extern tensor<float> \
		power(float a, tensor<int> &b) ;
	extern tensor<float> \
		power(float a, tensor<bool> &b) ;
	extern tensor<float> \
		power(size_t a, tensor<float> &b) ;
	extern tensor<float> \
		power(int a, tensor<float> &b) ;
	extern tensor<float> \
		power(bool a, tensor<float> &b) ;
	extern tensor<size_t> \
		power(size_t a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		power(size_t a, tensor<int> &b) ;
	extern tensor<size_t> \
		power(size_t a, tensor<bool> &b) ;
	extern tensor<size_t> \
		power(int a, tensor<size_t> &b) ;
	extern tensor<size_t> \
		power(bool a, tensor<size_t> &b) ;
	extern tensor<int> \
		power(int a, tensor<int> &b) ;
	extern tensor<int> \
		power(int a, tensor<bool> &b) ;
	extern tensor<int> \
		power(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		power(bool a, tensor<bool> &b) ;
	extern double \
		power(double a, double b) ;
	extern double \
		power(double a, float b) ;
	extern double \
		power(double a, size_t b) ;
	extern double \
		power(double a, int b) ;
	extern double \
		power(double a, bool b) ;
	extern double \
		power(float a, double b) ;
	extern double \
		power(size_t a, double b) ;
	extern double \
		power(int a, double b) ;
	extern double \
		power(bool a, double b) ;
	extern float \
		power(float a, float b) ;
	extern float \
		power(float a, size_t b) ;
	extern float \
		power(float a, int b) ;
	extern float \
		power(float a, bool b) ;
	extern float \
		power(size_t a, float b) ;
	extern float \
		power(int a, float b) ;
	extern float \
		power(bool a, float b) ;
	extern size_t \
		power(size_t a, size_t b) ;
	extern size_t \
		power(size_t a, int b) ;
	extern size_t \
		power(size_t a, bool b) ;
	extern size_t \
		power(int a, size_t b) ;
	extern size_t \
		power(bool a, size_t b) ;
	extern int \
		power(int a, int b) ;
	extern int \
		power(int a, bool b) ;
	extern int \
		power(bool a, int b) ;
	extern bool \
		power(bool a, bool b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, double b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, float b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, size_t b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, int b) ;
	extern tensor<bool> \
		logical_and(tensor<double> &a, bool b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, double b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, double b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, double b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, double b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, float b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, size_t b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, int b) ;
	extern tensor<bool> \
		logical_and(tensor<float> &a, bool b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, float b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, float b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, float b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, size_t b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, int b) ;
	extern tensor<bool> \
		logical_and(tensor<size_t> &a, bool b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, size_t b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, size_t b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, int b) ;
	extern tensor<bool> \
		logical_and(tensor<int> &a, bool b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		logical_and(tensor<bool> &a, bool b) ;
	extern tensor<bool> \
		logical_and(double a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(double a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(double a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(double a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(double a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_and(float a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(size_t a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(int a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(bool a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_and(float a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(float a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(float a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(float a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_and(size_t a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(int a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(bool a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_and(size_t a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(size_t a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(size_t a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_and(int a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(bool a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_and(int a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(int a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_and(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_and(bool a, tensor<bool> &b) ;
	extern bool \
		logical_and(double a, double b) ;
	extern bool \
		logical_and(double a, float b) ;
	extern bool \
		logical_and(double a, size_t b) ;
	extern bool \
		logical_and(double a, int b) ;
	extern bool \
		logical_and(double a, bool b) ;
	extern bool \
		logical_and(float a, double b) ;
	extern bool \
		logical_and(size_t a, double b) ;
	extern bool \
		logical_and(int a, double b) ;
	extern bool \
		logical_and(bool a, double b) ;
	extern bool \
		logical_and(float a, float b) ;
	extern bool \
		logical_and(float a, size_t b) ;
	extern bool \
		logical_and(float a, int b) ;
	extern bool \
		logical_and(float a, bool b) ;
	extern bool \
		logical_and(size_t a, float b) ;
	extern bool \
		logical_and(int a, float b) ;
	extern bool \
		logical_and(bool a, float b) ;
	extern bool \
		logical_and(size_t a, size_t b) ;
	extern bool \
		logical_and(size_t a, int b) ;
	extern bool \
		logical_and(size_t a, bool b) ;
	extern bool \
		logical_and(int a, size_t b) ;
	extern bool \
		logical_and(bool a, size_t b) ;
	extern bool \
		logical_and(int a, int b) ;
	extern bool \
		logical_and(int a, bool b) ;
	extern bool \
		logical_and(bool a, int b) ;
	extern bool \
		logical_and(bool a, bool b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, double b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, float b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, size_t b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, int b) ;
	extern tensor<bool> \
		logical_or(tensor<double> &a, bool b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, double b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, double b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, double b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, double b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, float b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, size_t b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, int b) ;
	extern tensor<bool> \
		logical_or(tensor<float> &a, bool b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, float b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, float b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, float b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, size_t b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, int b) ;
	extern tensor<bool> \
		logical_or(tensor<size_t> &a, bool b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, size_t b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, size_t b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, int b) ;
	extern tensor<bool> \
		logical_or(tensor<int> &a, bool b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		logical_or(tensor<bool> &a, bool b) ;
	extern tensor<bool> \
		logical_or(double a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(double a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(double a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(double a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(double a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_or(float a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(size_t a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(int a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(bool a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_or(float a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(float a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(float a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(float a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_or(size_t a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(int a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(bool a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_or(size_t a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(size_t a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(size_t a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_or(int a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(bool a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_or(int a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(int a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_or(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_or(bool a, tensor<bool> &b) ;
	extern bool \
		logical_or(double a, double b) ;
	extern bool \
		logical_or(double a, float b) ;
	extern bool \
		logical_or(double a, size_t b) ;
	extern bool \
		logical_or(double a, int b) ;
	extern bool \
		logical_or(double a, bool b) ;
	extern bool \
		logical_or(float a, double b) ;
	extern bool \
		logical_or(size_t a, double b) ;
	extern bool \
		logical_or(int a, double b) ;
	extern bool \
		logical_or(bool a, double b) ;
	extern bool \
		logical_or(float a, float b) ;
	extern bool \
		logical_or(float a, size_t b) ;
	extern bool \
		logical_or(float a, int b) ;
	extern bool \
		logical_or(float a, bool b) ;
	extern bool \
		logical_or(size_t a, float b) ;
	extern bool \
		logical_or(int a, float b) ;
	extern bool \
		logical_or(bool a, float b) ;
	extern bool \
		logical_or(size_t a, size_t b) ;
	extern bool \
		logical_or(size_t a, int b) ;
	extern bool \
		logical_or(size_t a, bool b) ;
	extern bool \
		logical_or(int a, size_t b) ;
	extern bool \
		logical_or(bool a, size_t b) ;
	extern bool \
		logical_or(int a, int b) ;
	extern bool \
		logical_or(int a, bool b) ;
	extern bool \
		logical_or(bool a, int b) ;
	extern bool \
		logical_or(bool a, bool b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, double b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, float b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, size_t b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, int b) ;
	extern tensor<bool> \
		logical_xor(tensor<double> &a, bool b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, double b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, double b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, double b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, double b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, float b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, size_t b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, int b) ;
	extern tensor<bool> \
		logical_xor(tensor<float> &a, bool b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, float b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, float b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, float b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, size_t b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, int b) ;
	extern tensor<bool> \
		logical_xor(tensor<size_t> &a, bool b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, size_t b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, size_t b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, int b) ;
	extern tensor<bool> \
		logical_xor(tensor<int> &a, bool b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		logical_xor(tensor<bool> &a, bool b) ;
	extern tensor<bool> \
		logical_xor(double a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(double a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(double a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(double a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(double a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_xor(float a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(size_t a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(int a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(bool a, tensor<double> &b) ;
	extern tensor<bool> \
		logical_xor(float a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(float a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(float a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(float a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_xor(size_t a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(int a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(bool a, tensor<float> &b) ;
	extern tensor<bool> \
		logical_xor(size_t a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(size_t a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(size_t a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_xor(int a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(bool a, tensor<size_t> &b) ;
	extern tensor<bool> \
		logical_xor(int a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(int a, tensor<bool> &b) ;
	extern tensor<bool> \
		logical_xor(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		logical_xor(bool a, tensor<bool> &b) ;
	extern bool \
		logical_xor(double a, double b) ;
	extern bool \
		logical_xor(double a, float b) ;
	extern bool \
		logical_xor(double a, size_t b) ;
	extern bool \
		logical_xor(double a, int b) ;
	extern bool \
		logical_xor(double a, bool b) ;
	extern bool \
		logical_xor(float a, double b) ;
	extern bool \
		logical_xor(size_t a, double b) ;
	extern bool \
		logical_xor(int a, double b) ;
	extern bool \
		logical_xor(bool a, double b) ;
	extern bool \
		logical_xor(float a, float b) ;
	extern bool \
		logical_xor(float a, size_t b) ;
	extern bool \
		logical_xor(float a, int b) ;
	extern bool \
		logical_xor(float a, bool b) ;
	extern bool \
		logical_xor(size_t a, float b) ;
	extern bool \
		logical_xor(int a, float b) ;
	extern bool \
		logical_xor(bool a, float b) ;
	extern bool \
		logical_xor(size_t a, size_t b) ;
	extern bool \
		logical_xor(size_t a, int b) ;
	extern bool \
		logical_xor(size_t a, bool b) ;
	extern bool \
		logical_xor(int a, size_t b) ;
	extern bool \
		logical_xor(bool a, size_t b) ;
	extern bool \
		logical_xor(int a, int b) ;
	extern bool \
		logical_xor(int a, bool b) ;
	extern bool \
		logical_xor(bool a, int b) ;
	extern bool \
		logical_xor(bool a, bool b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, double b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, float b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, size_t b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, int b) ;
	extern tensor<bool> \
		equal(tensor<double> &a, bool b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, double b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, double b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, double b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, double b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, float b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, size_t b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, int b) ;
	extern tensor<bool> \
		equal(tensor<float> &a, bool b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, float b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, float b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, float b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, size_t b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, int b) ;
	extern tensor<bool> \
		equal(tensor<size_t> &a, bool b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, size_t b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, size_t b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, int b) ;
	extern tensor<bool> \
		equal(tensor<int> &a, bool b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		equal(tensor<bool> &a, bool b) ;
	extern tensor<bool> \
		equal(double a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(double a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(double a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(double a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(double a, tensor<bool> &b) ;
	extern tensor<bool> \
		equal(float a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(size_t a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(int a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(bool a, tensor<double> &b) ;
	extern tensor<bool> \
		equal(float a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(float a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(float a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(float a, tensor<bool> &b) ;
	extern tensor<bool> \
		equal(size_t a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(int a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(bool a, tensor<float> &b) ;
	extern tensor<bool> \
		equal(size_t a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(size_t a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(size_t a, tensor<bool> &b) ;
	extern tensor<bool> \
		equal(int a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(bool a, tensor<size_t> &b) ;
	extern tensor<bool> \
		equal(int a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(int a, tensor<bool> &b) ;
	extern tensor<bool> \
		equal(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		equal(bool a, tensor<bool> &b) ;
	extern bool \
		equal(double a, double b) ;
	extern bool \
		equal(double a, float b) ;
	extern bool \
		equal(double a, size_t b) ;
	extern bool \
		equal(double a, int b) ;
	extern bool \
		equal(double a, bool b) ;
	extern bool \
		equal(float a, double b) ;
	extern bool \
		equal(size_t a, double b) ;
	extern bool \
		equal(int a, double b) ;
	extern bool \
		equal(bool a, double b) ;
	extern bool \
		equal(float a, float b) ;
	extern bool \
		equal(float a, size_t b) ;
	extern bool \
		equal(float a, int b) ;
	extern bool \
		equal(float a, bool b) ;
	extern bool \
		equal(size_t a, float b) ;
	extern bool \
		equal(int a, float b) ;
	extern bool \
		equal(bool a, float b) ;
	extern bool \
		equal(size_t a, size_t b) ;
	extern bool \
		equal(size_t a, int b) ;
	extern bool \
		equal(size_t a, bool b) ;
	extern bool \
		equal(int a, size_t b) ;
	extern bool \
		equal(bool a, size_t b) ;
	extern bool \
		equal(int a, int b) ;
	extern bool \
		equal(int a, bool b) ;
	extern bool \
		equal(bool a, int b) ;
	extern bool \
		equal(bool a, bool b) ;
	extern tensor<bool> \
		less(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less(tensor<double> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(tensor<double> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less(tensor<float> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less(tensor<int> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(tensor<float> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less(tensor<int> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less(tensor<double> &a, double b) ;
	extern tensor<bool> \
		less(tensor<double> &a, float b) ;
	extern tensor<bool> \
		less(tensor<double> &a, size_t b) ;
	extern tensor<bool> \
		less(tensor<double> &a, int b) ;
	extern tensor<bool> \
		less(tensor<double> &a, bool b) ;
	extern tensor<bool> \
		less(tensor<float> &a, double b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, double b) ;
	extern tensor<bool> \
		less(tensor<int> &a, double b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, double b) ;
	extern tensor<bool> \
		less(tensor<float> &a, float b) ;
	extern tensor<bool> \
		less(tensor<float> &a, size_t b) ;
	extern tensor<bool> \
		less(tensor<float> &a, int b) ;
	extern tensor<bool> \
		less(tensor<float> &a, bool b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, float b) ;
	extern tensor<bool> \
		less(tensor<int> &a, float b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, float b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, size_t b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, int b) ;
	extern tensor<bool> \
		less(tensor<size_t> &a, bool b) ;
	extern tensor<bool> \
		less(tensor<int> &a, size_t b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, size_t b) ;
	extern tensor<bool> \
		less(tensor<int> &a, int b) ;
	extern tensor<bool> \
		less(tensor<int> &a, bool b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		less(tensor<bool> &a, bool b) ;
	extern tensor<bool> \
		less(double a, tensor<double> &b) ;
	extern tensor<bool> \
		less(double a, tensor<float> &b) ;
	extern tensor<bool> \
		less(double a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(double a, tensor<int> &b) ;
	extern tensor<bool> \
		less(double a, tensor<bool> &b) ;
	extern tensor<bool> \
		less(float a, tensor<double> &b) ;
	extern tensor<bool> \
		less(size_t a, tensor<double> &b) ;
	extern tensor<bool> \
		less(int a, tensor<double> &b) ;
	extern tensor<bool> \
		less(bool a, tensor<double> &b) ;
	extern tensor<bool> \
		less(float a, tensor<float> &b) ;
	extern tensor<bool> \
		less(float a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(float a, tensor<int> &b) ;
	extern tensor<bool> \
		less(float a, tensor<bool> &b) ;
	extern tensor<bool> \
		less(size_t a, tensor<float> &b) ;
	extern tensor<bool> \
		less(int a, tensor<float> &b) ;
	extern tensor<bool> \
		less(bool a, tensor<float> &b) ;
	extern tensor<bool> \
		less(size_t a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(size_t a, tensor<int> &b) ;
	extern tensor<bool> \
		less(size_t a, tensor<bool> &b) ;
	extern tensor<bool> \
		less(int a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(bool a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less(int a, tensor<int> &b) ;
	extern tensor<bool> \
		less(int a, tensor<bool> &b) ;
	extern tensor<bool> \
		less(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		less(bool a, tensor<bool> &b) ;
	extern bool \
		less(double a, double b) ;
	extern bool \
		less(double a, float b) ;
	extern bool \
		less(double a, size_t b) ;
	extern bool \
		less(double a, int b) ;
	extern bool \
		less(double a, bool b) ;
	extern bool \
		less(float a, double b) ;
	extern bool \
		less(size_t a, double b) ;
	extern bool \
		less(int a, double b) ;
	extern bool \
		less(bool a, double b) ;
	extern bool \
		less(float a, float b) ;
	extern bool \
		less(float a, size_t b) ;
	extern bool \
		less(float a, int b) ;
	extern bool \
		less(float a, bool b) ;
	extern bool \
		less(size_t a, float b) ;
	extern bool \
		less(int a, float b) ;
	extern bool \
		less(bool a, float b) ;
	extern bool \
		less(size_t a, size_t b) ;
	extern bool \
		less(size_t a, int b) ;
	extern bool \
		less(size_t a, bool b) ;
	extern bool \
		less(int a, size_t b) ;
	extern bool \
		less(bool a, size_t b) ;
	extern bool \
		less(int a, int b) ;
	extern bool \
		less(int a, bool b) ;
	extern bool \
		less(bool a, int b) ;
	extern bool \
		less(bool a, bool b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, double b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, float b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, size_t b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, int b) ;
	extern tensor<bool> \
		greater(tensor<double> &a, bool b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, double b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, double b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, double b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, double b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, float b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, size_t b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, int b) ;
	extern tensor<bool> \
		greater(tensor<float> &a, bool b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, float b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, float b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, float b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, size_t b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, int b) ;
	extern tensor<bool> \
		greater(tensor<size_t> &a, bool b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, size_t b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, size_t b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, int b) ;
	extern tensor<bool> \
		greater(tensor<int> &a, bool b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		greater(tensor<bool> &a, bool b) ;
	extern tensor<bool> \
		greater(double a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(double a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(double a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(double a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(double a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater(float a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(size_t a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(int a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(bool a, tensor<double> &b) ;
	extern tensor<bool> \
		greater(float a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(float a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(float a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(float a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater(size_t a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(int a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(bool a, tensor<float> &b) ;
	extern tensor<bool> \
		greater(size_t a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(size_t a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(size_t a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater(int a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(bool a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater(int a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(int a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		greater(bool a, tensor<bool> &b) ;
	extern bool \
		greater(double a, double b) ;
	extern bool \
		greater(double a, float b) ;
	extern bool \
		greater(double a, size_t b) ;
	extern bool \
		greater(double a, int b) ;
	extern bool \
		greater(double a, bool b) ;
	extern bool \
		greater(float a, double b) ;
	extern bool \
		greater(size_t a, double b) ;
	extern bool \
		greater(int a, double b) ;
	extern bool \
		greater(bool a, double b) ;
	extern bool \
		greater(float a, float b) ;
	extern bool \
		greater(float a, size_t b) ;
	extern bool \
		greater(float a, int b) ;
	extern bool \
		greater(float a, bool b) ;
	extern bool \
		greater(size_t a, float b) ;
	extern bool \
		greater(int a, float b) ;
	extern bool \
		greater(bool a, float b) ;
	extern bool \
		greater(size_t a, size_t b) ;
	extern bool \
		greater(size_t a, int b) ;
	extern bool \
		greater(size_t a, bool b) ;
	extern bool \
		greater(int a, size_t b) ;
	extern bool \
		greater(bool a, size_t b) ;
	extern bool \
		greater(int a, int b) ;
	extern bool \
		greater(int a, bool b) ;
	extern bool \
		greater(bool a, int b) ;
	extern bool \
		greater(bool a, bool b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, double b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, float b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, size_t b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, int b) ;
	extern tensor<bool> \
		not_equal(tensor<double> &a, bool b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, double b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, double b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, double b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, double b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, float b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, size_t b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, int b) ;
	extern tensor<bool> \
		not_equal(tensor<float> &a, bool b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, float b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, float b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, float b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, size_t b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, int b) ;
	extern tensor<bool> \
		not_equal(tensor<size_t> &a, bool b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, size_t b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, size_t b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, int b) ;
	extern tensor<bool> \
		not_equal(tensor<int> &a, bool b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		not_equal(tensor<bool> &a, bool b) ;
	extern tensor<bool> \
		not_equal(double a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(double a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(double a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(double a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(double a, tensor<bool> &b) ;
	extern tensor<bool> \
		not_equal(float a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(size_t a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(int a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(bool a, tensor<double> &b) ;
	extern tensor<bool> \
		not_equal(float a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(float a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(float a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(float a, tensor<bool> &b) ;
	extern tensor<bool> \
		not_equal(size_t a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(int a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(bool a, tensor<float> &b) ;
	extern tensor<bool> \
		not_equal(size_t a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(size_t a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(size_t a, tensor<bool> &b) ;
	extern tensor<bool> \
		not_equal(int a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(bool a, tensor<size_t> &b) ;
	extern tensor<bool> \
		not_equal(int a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(int a, tensor<bool> &b) ;
	extern tensor<bool> \
		not_equal(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		not_equal(bool a, tensor<bool> &b) ;
	extern bool \
		not_equal(double a, double b) ;
	extern bool \
		not_equal(double a, float b) ;
	extern bool \
		not_equal(double a, size_t b) ;
	extern bool \
		not_equal(double a, int b) ;
	extern bool \
		not_equal(double a, bool b) ;
	extern bool \
		not_equal(float a, double b) ;
	extern bool \
		not_equal(size_t a, double b) ;
	extern bool \
		not_equal(int a, double b) ;
	extern bool \
		not_equal(bool a, double b) ;
	extern bool \
		not_equal(float a, float b) ;
	extern bool \
		not_equal(float a, size_t b) ;
	extern bool \
		not_equal(float a, int b) ;
	extern bool \
		not_equal(float a, bool b) ;
	extern bool \
		not_equal(size_t a, float b) ;
	extern bool \
		not_equal(int a, float b) ;
	extern bool \
		not_equal(bool a, float b) ;
	extern bool \
		not_equal(size_t a, size_t b) ;
	extern bool \
		not_equal(size_t a, int b) ;
	extern bool \
		not_equal(size_t a, bool b) ;
	extern bool \
		not_equal(int a, size_t b) ;
	extern bool \
		not_equal(bool a, size_t b) ;
	extern bool \
		not_equal(int a, int b) ;
	extern bool \
		not_equal(int a, bool b) ;
	extern bool \
		not_equal(bool a, int b) ;
	extern bool \
		not_equal(bool a, bool b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, double b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, float b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, size_t b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, int b) ;
	extern tensor<bool> \
		less_equal(tensor<double> &a, bool b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, double b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, double b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, double b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, double b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, float b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, size_t b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, int b) ;
	extern tensor<bool> \
		less_equal(tensor<float> &a, bool b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, float b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, float b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, float b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, size_t b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, int b) ;
	extern tensor<bool> \
		less_equal(tensor<size_t> &a, bool b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, size_t b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, size_t b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, int b) ;
	extern tensor<bool> \
		less_equal(tensor<int> &a, bool b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		less_equal(tensor<bool> &a, bool b) ;
	extern tensor<bool> \
		less_equal(double a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(double a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(double a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(double a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(double a, tensor<bool> &b) ;
	extern tensor<bool> \
		less_equal(float a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(size_t a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(int a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(bool a, tensor<double> &b) ;
	extern tensor<bool> \
		less_equal(float a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(float a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(float a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(float a, tensor<bool> &b) ;
	extern tensor<bool> \
		less_equal(size_t a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(int a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(bool a, tensor<float> &b) ;
	extern tensor<bool> \
		less_equal(size_t a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(size_t a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(size_t a, tensor<bool> &b) ;
	extern tensor<bool> \
		less_equal(int a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(bool a, tensor<size_t> &b) ;
	extern tensor<bool> \
		less_equal(int a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(int a, tensor<bool> &b) ;
	extern tensor<bool> \
		less_equal(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		less_equal(bool a, tensor<bool> &b) ;
	extern bool \
		less_equal(double a, double b) ;
	extern bool \
		less_equal(double a, float b) ;
	extern bool \
		less_equal(double a, size_t b) ;
	extern bool \
		less_equal(double a, int b) ;
	extern bool \
		less_equal(double a, bool b) ;
	extern bool \
		less_equal(float a, double b) ;
	extern bool \
		less_equal(size_t a, double b) ;
	extern bool \
		less_equal(int a, double b) ;
	extern bool \
		less_equal(bool a, double b) ;
	extern bool \
		less_equal(float a, float b) ;
	extern bool \
		less_equal(float a, size_t b) ;
	extern bool \
		less_equal(float a, int b) ;
	extern bool \
		less_equal(float a, bool b) ;
	extern bool \
		less_equal(size_t a, float b) ;
	extern bool \
		less_equal(int a, float b) ;
	extern bool \
		less_equal(bool a, float b) ;
	extern bool \
		less_equal(size_t a, size_t b) ;
	extern bool \
		less_equal(size_t a, int b) ;
	extern bool \
		less_equal(size_t a, bool b) ;
	extern bool \
		less_equal(int a, size_t b) ;
	extern bool \
		less_equal(bool a, size_t b) ;
	extern bool \
		less_equal(int a, int b) ;
	extern bool \
		less_equal(int a, bool b) ;
	extern bool \
		less_equal(bool a, int b) ;
	extern bool \
		less_equal(bool a, bool b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, double b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, float b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, size_t b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, int b) ;
	extern tensor<bool> \
		greater_equal(tensor<double> &a, bool b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, double b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, double b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, double b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, double b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, float b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, size_t b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, int b) ;
	extern tensor<bool> \
		greater_equal(tensor<float> &a, bool b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, float b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, float b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, float b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, size_t b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, int b) ;
	extern tensor<bool> \
		greater_equal(tensor<size_t> &a, bool b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, size_t b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, size_t b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, int b) ;
	extern tensor<bool> \
		greater_equal(tensor<int> &a, bool b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, int b) ;
	extern tensor<bool> \
		greater_equal(tensor<bool> &a, bool b) ;
	extern tensor<bool> \
		greater_equal(double a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(double a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(double a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(double a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(double a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater_equal(float a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(size_t a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(int a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(bool a, tensor<double> &b) ;
	extern tensor<bool> \
		greater_equal(float a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(float a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(float a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(float a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater_equal(size_t a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(int a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(bool a, tensor<float> &b) ;
	extern tensor<bool> \
		greater_equal(size_t a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(size_t a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(size_t a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater_equal(int a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(bool a, tensor<size_t> &b) ;
	extern tensor<bool> \
		greater_equal(int a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(int a, tensor<bool> &b) ;
	extern tensor<bool> \
		greater_equal(bool a, tensor<int> &b) ;
	extern tensor<bool> \
		greater_equal(bool a, tensor<bool> &b) ;
	extern bool \
		greater_equal(double a, double b) ;
	extern bool \
		greater_equal(double a, float b) ;
	extern bool \
		greater_equal(double a, size_t b) ;
	extern bool \
		greater_equal(double a, int b) ;
	extern bool \
		greater_equal(double a, bool b) ;
	extern bool \
		greater_equal(float a, double b) ;
	extern bool \
		greater_equal(size_t a, double b) ;
	extern bool \
		greater_equal(int a, double b) ;
	extern bool \
		greater_equal(bool a, double b) ;
	extern bool \
		greater_equal(float a, float b) ;
	extern bool \
		greater_equal(float a, size_t b) ;
	extern bool \
		greater_equal(float a, int b) ;
	extern bool \
		greater_equal(float a, bool b) ;
	extern bool \
		greater_equal(size_t a, float b) ;
	extern bool \
		greater_equal(int a, float b) ;
	extern bool \
		greater_equal(bool a, float b) ;
	extern bool \
		greater_equal(size_t a, size_t b) ;
	extern bool \
		greater_equal(size_t a, int b) ;
	extern bool \
		greater_equal(size_t a, bool b) ;
	extern bool \
		greater_equal(int a, size_t b) ;
	extern bool \
		greater_equal(bool a, size_t b) ;
	extern bool \
		greater_equal(int a, int b) ;
	extern bool \
		greater_equal(int a, bool b) ;
	extern bool \
		greater_equal(bool a, int b) ;
	extern bool \
		greater_equal(bool a, bool b) ;
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
		abs(tensor<double> &a) ;
	extern tensor<float> \
		abs(tensor<float> &a) ;
	extern tensor<int> \
		abs(tensor<int> &a) ;
	extern tensor<int> \
		bitshift(tensor<int> &a, tensor<int> &b, std::string direction = "") ;
	extern tensor<float> \
		dequantize_linear(tensor<int> &a, tensor<float> &b, tensor<int> &c) ;
	extern tensor<double> \
		dropout(tensor<double> &a, float ratio = 0.5) ;
	extern tensor<float> \
		dropout(tensor<float> &a, float ratio = 0.5) ;
	extern tensor<double> \
		div(tensor<double> &a, tensor<double> &b) ;
	extern tensor<float> \
		div(tensor<float> &a, tensor<float> &b) ;
	extern tensor<int> \
		div(tensor<int> &a, tensor<int> &b) ;
	extern tensor<double> \
		elu(tensor<double> &a, float alpha = 1.0) ;
	extern tensor<float> \
		elu(tensor<float> &a, float alpha = 1.0) ;
	extern tensor<double> \
		erf(tensor<double> &a) ;
	extern tensor<float> \
		erf(tensor<float> &a) ;
	extern tensor<double> \
		exp(tensor<double> &a) ;
	extern tensor<float> \
		exp(tensor<float> &a) ;
	extern tensor<double> \
		expand(tensor<double> &a, tensor<int> &b) ;
	extern tensor<float> \
		expand(tensor<float> &a, tensor<int> &b) ;
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
	extern tensor<float> \
		reciprocal(tensor<float> &a) ;
	extern tensor<double> \
		reciprocal(tensor<double> &a) ;
	extern tensor<float> \
		log(tensor<float> &a) ;
	extern tensor<double> \
		log(tensor<double> &a) ;
	extern tensor<float> \
		logsoftmax(tensor<float> &a, int axis = 1) ;
	extern tensor<double> \
		logsoftmax(tensor<double> &a, int axis = 1) ;
	extern tensor<float> \
		lpnormalization(tensor<float> &a, int p = 2, int axis = -1) ;
	extern tensor<double> \
		lpnormalization(tensor<double> &a, int p = 2, int axis = -1) ;
	extern tensor<int> \
		matmulinteger(tensor<int> &a, tensor<int> &b, tensor<int> &c, tensor<int> &d) ;
	extern tensor<float> \
		min(std::vector<tensor<float>> a) ;
	extern tensor<double> \
		min(std::vector<tensor<double>> a) ;
	extern tensor<float> \
		mean(std::vector<tensor<float>> a) ;
	extern tensor<double> \
		mean(std::vector<tensor<double>> a) ;
	extern tensor<float> \
		max(std::vector<tensor<float>> a) ;
	extern tensor<double> \
		max(std::vector<tensor<double>> a) ;
	extern tensor<double> \
		mod(tensor<double> &a, tensor<double> &b, int fmod_flag = 0) ;
	extern tensor<float> \
		mod(tensor<float> &a, tensor<float> &b, int fmod_flag = 0) ;
	extern tensor<int> \
		mod(tensor<int> &a, tensor<int> &b, int fmod_flag = 0) ;
	extern tensor<float> \
		thresholded_relu(tensor<float> &a) ;
	extern tensor<double> \
		thresholded_relu(tensor<double> &a) ;
	extern tensor<double> \
		neg(tensor<double> &a) ;
	extern tensor<float> \
		neg(tensor<float> &a) ;
	extern tensor<int> \
		neg(tensor<int> &a) ;
	extern tensor<bool> \
		logical_not(tensor<bool> &a) ;
	extern tensor<bool> \
		logical_not(tensor<int> &a) ;
	extern tensor<bool> \
		logical_not(tensor<float> &a) ;
	extern tensor<bool> \
		logical_not(tensor<double> &a) ;
	extern tensor<float> \
		tan(tensor<float> &a) ;
	extern tensor<double> \
		tan(tensor<double> &a) ;
	extern tensor<float> \
		tanh(tensor<float> &a) ;
	extern tensor<double> \
		tanh(tensor<double> &a) ;
	extern tensor<double> \
		transpose(tensor<double> &a) ;
	extern tensor<float> \
		transpose(tensor<float> &a) ;
	extern tensor<int> \
		transpose(tensor<int> &a) ;
	extern tensor<bool> \
		transpose(tensor<bool> &a) ;
	extern tensor<float> \
		sigmoid(tensor<float> &a) ;
	extern tensor<float> \
		sign(tensor<float> &a) ;
	extern tensor<double> \
		sign(tensor<double> &a) ;
	extern tensor<float> \
		sin(tensor<float> &a) ;
	extern tensor<double> \
		sin(tensor<double> &a) ;
	extern tensor<float> \
		sinh(tensor<float> &a) ;
	extern tensor<double> \
		sinh(tensor<double> &a) ;
	extern tensor<float> \
		softplus(tensor<float> &a) ;
	extern tensor<double> \
		softplus(tensor<double> &a) ;
	extern tensor<float> \
		softmax(tensor<float> &a, int axis = 1) ;
	extern tensor<double> \
		softmax(tensor<double> &a, int axis = 1) ;
	extern tensor<float> \
		softsign(tensor<float> &a) ;
	extern tensor<double> \
		softsign(tensor<double> &a) ;
	extern tensor<float> \
		sqrt(tensor<float> &a) ;
	extern tensor<double> \
		sqrt(tensor<double> &a) ;
	extern tensor<float> \
		ceil(tensor<float> &a);
	extern tensor<double> \
		ceil(tensor<double> &a);
	extern tensor<float> \
		clip(tensor<float> &a, float min, float max);
	extern tensor<double> \
		clip(tensor<double> &a, double min, double max);
	extern tensor<float> \
		cos(tensor<float> &a);
	extern tensor<double> \
		cos(tensor<double> &a);
	extern tensor<float> \
		cosh(tensor<float> &a);
	extern tensor<double> \
		cosh(tensor<double> &a);
	extern tensor<double> \
		slice(tensor<double> &a, 
	             tensor<size_t> &start, 
		     tensor<size_t> &end, 
		     tensor<int> axes = NULL_TENSOR<int>, 
		     tensor<size_t> steps = NULL_TENSOR<size_t>) ;
	extern tensor<float> \
		slice(tensor<float> &a, 
	             tensor<size_t> &start, 
		     tensor<size_t> &end, 
		     tensor<int> axes = NULL_TENSOR<int>, 
		     tensor<size_t> steps = NULL_TENSOR<size_t>) ;
	extern tensor<int> \
		slice(tensor<int> &a, 
	             tensor<size_t> &start, 
		     tensor<size_t> &end, 
		     tensor<int> axes = NULL_TENSOR<int>, 
		     tensor<size_t> steps = NULL_TENSOR<size_t>) ;
	extern tensor<bool> \
		slice(tensor<bool> &a, 
	             tensor<size_t> &start, 
		     tensor<size_t> &end, 
		     tensor<int> axes = NULL_TENSOR<int>, 
		     tensor<size_t> steps = NULL_TENSOR<size_t>) ;
	extern tensor<float> \
		prelu(tensor<float> &x, tensor<float> &slope) ;
	extern tensor<double> \
		prelu(tensor<double> &x, tensor<double> &slope) ;
	extern tensor<int> \
		prelu(tensor<int> &x, tensor<int> &slope) ;
}
