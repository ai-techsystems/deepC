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

#ifndef SWIGPYTHON
#include <typeinfo>
#include <stdlib.h>    // malloc, free
#include <vector>
#include <string>
#include <stdexcept>
#endif
#include "datatypes.h"

namespace dnnc {
	typedef size_t INDEX;
	typedef size_t DIMENSION;

	// Tensor with arbitrary dimension.
	template <typename T>
	class tensor {

	private:

	protected:
		std::vector<DIMENSION> _shape;
		T* _mem_layout; // TODO: convert it into object of layout class to accomodate tiling and reference counting.

		T* getMemory(size_t sz)
		{
			_mem_layout = sz ? 
#ifndef SWIGPYTHON
			    static_cast<T*> ( malloc(sizeof(T)*sz) ) :
#else
			    new T [sz] :
#endif
                0x0;
			if (sz && !_mem_layout)
				throw std::bad_alloc();
			return _mem_layout;
		}
        void init() {
            if ( _shape.size() == 0 )
#ifndef SWIGPYTHON
                throw std::invalid_argument("tensor with no shape.");
#endif
			_mem_layout = getMemory(size());
        }
	public:
		// tensor constructor with arbitrary dimension
		tensor(std::vector<DIMENSION> dimn) : _mem_layout(0x0)
		{
			_shape = dimn;
            init();
		}
		tensor( DIMENSION x = 0, DIMENSION y = 0, 
				DIMENSION z = 0, DIMENSION w = 0) : _mem_layout(0x0)
		{
			if ( x ) {
				_shape.push_back(x);
				if (y)
					_shape.push_back(y);
				if (z)
					_shape.push_back(z);
				if (w)
					_shape.push_back(w);
			}
            init();
		}
		~tensor()
		{
#ifndef SWIGPYTHON
            if ( _mem_layout )
                free(_mem_layout);
#else
			delete [] _mem_layout;
#endif
		}

		// public methods
		const DIMENSION size() const
		{
			DIMENSION sz = _shape.size() ? 1 : 0;
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
			if (index >= size())
				throw std::out_of_range ("illegal tensor index.");

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
			return this->operator[](index);
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
		std::string dtype()
		{
			return dtype_str[typeid(T).name()[0]-'a'];
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
