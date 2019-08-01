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

#pragma once

#include <assert.h>
#include <stdlib.h>    // malloc, free
#include <memory>     // shared_ptr
#include <vector>
#include <string>

namespace dnnc {
	typedef size_t INDEX;
	typedef size_t DIMENSION;

	// Tensor with arbitrary dimension.
	template <typename T>
	class tensor {
	private:
		// NO default constructor, because makes no sense for tensors.
		tensor() {} 
	protected:
		std::vector<DIMENSION> _shape;
		T* _mem_layout; // TODO: converrt it into object of layout class to accomodate tiling.

		T* getMemory()
		{
			_mem_layout = static_cast<T*>(malloc(size() * sizeof(T)));
			assert(_mem_layout);
			return _mem_layout;
		}
	public:
		// tensor constructor with arbitrary dimension
		tensor(std::vector<DIMENSION> dimn)
		{
			_shape = dimn;
			_mem_layout = getMemory(size());
		}
		tensor(DIMENSION x, DIMENSION y = 0, DIMENSION z = 0, DIMENSION w = 0)
		{
			_shape.push_back(x);
			if (y)
				_shape.push_back(y);
			if (z)
				_shape.push_back(z);
			if (w)
				_shape.push_back(w);
			_mem_layout = getMemory();
		}
		~tensor()
		{
			if (_mem_layout)
				free(_mem_layout);
		}

		// public methods
		const DIMENSION size() const
		{
			DIMENSION sz = 1;
			for (size_t i = 0; i < _shape.size(); i++)
				sz = sz * _shape[i];
			return sz;
		}
		const std::vector<DIMENSION> shape() const
		{
			return _shape;
		}
		// flat index, unsafe method
		T& operator[](const INDEX& index) const
		{
			assert(index < size());
			return _mem_layout[index];
		}

		T& operator() (std::vector<INDEX>& indices) const
		{
			INDEX index = 0;
			for (size_t i = 0; i < indices.size(); i++)
			{
				DIMENSION dsz = 1;
				for (size_t j = i + 1; j < _shape.size(); j++)
					dsz *= _shape[j];
				index += indices[i] * dsz;
			}
			return _mem_layout[index];
		}
		T& operator()(const INDEX x = 0, const INDEX y = 0,
			const INDEX z = 0, const INDEX w = 0) const
		{
			std::vector<INDEX> indices;
			indices.push_back(x);
			if (_shape.size() > 1)
				indices.push_back(y);
			if (_shape.size() > 2)
				indices.push_back(z);
			if (_shape.size() > 3)
				indices.push_back(w);

			return this->operator()(indices);
		}
		bool empty()
		{
			return size() == 0;
		}
		std::string toProto() // return proto string
		{
      std::string tensor_proto = "";
			return tensor_proto;
		}
		void toEigen()
		{}

	};
}
