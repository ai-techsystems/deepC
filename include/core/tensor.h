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

#include "core/macros.h"
#include "core/datatypes.h"

#ifndef SWIGPYTHON
#include <typeinfo>
#include <stdlib.h>	// malloc, free
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#endif
#include <random>

namespace dnnc {
	typedef size_t INDEX;
	typedef size_t DIMENSION;

	template <typename T> class baseOperator ;

	// Tensor with arbitrary rank.
	template <typename T>
	class tensor {
		friend class baseOperator<T> ;

	protected:
		//////////// protected members /////////////////
		std::vector<DIMENSION> _shape;
		T* _mem_layout; // TODO: convert it into object of layout class to accomodate tiling and reference counting.

		//////////// protected methods /////////////////
		T* getMemory(size_t sz)
		{
			_mem_layout = sz ?  static_cast<T*> ( malloc(sizeof(T)*sz) ) : 0x0;
			if (sz && !_mem_layout)
				throw std::bad_alloc();
			return _mem_layout;
		}
		void init() {
			size_t msize = size() ; // memory size
			if ( rank() == 0 )
#ifndef SWIGPYTHON
				throw std::invalid_argument("tensor with no shape.");
#endif
			_mem_layout = getMemory(msize);

			// initilize with uniform distribution.
			std::default_random_engine generator ;
			std::uniform_real_distribution<double> distribution(0, 255);
			for (size_t i=0; i<msize; i++) 
				_mem_layout[i] = static_cast<T>(distribution(generator));
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
		tensor (const tensor& other)
		{
			_shape = other._shape ;
			_mem_layout = getMemory(size());
			for (size_t i=0; i<size(); i++) 
				_mem_layout[i] = other._mem_layout[i] ;
		}
		tensor& operator=(const tensor& other)
		{ 
		 	// Gracefully handle self assignment
			if (this == &other) 
				return *this;

			_shape = other._shape ;
			_mem_layout = getMemory(size());
			for (size_t i=0; i<size(); i++) 
				_mem_layout[i] = other._mem_layout[i] ;

			return *this ;
		}
		~tensor()
		{
			if ( _mem_layout )
				free(_mem_layout);
		}

		// WARNING: Make sure data being loaded has same size as tensor.
		void load(T* data) {
			if ( !data )
				return ;
			for(size_t i=0; i<size(); i++)
				_mem_layout[i] = data[i];
		}
		friend std::ostream& operator<<(std::ostream& os, const tensor<T>& t)
		{
			for (size_t i=0; i<t.size(); i++)
				os << t._mem_layout[i] << ' ' ;
			return os;
		}
		std::string to_string()
		{
			std::string str ;
			for (size_t i=0; i<size(); i++)
				str += std::to_string(_mem_layout[i]) + (i==size()-1?"":" ");
			return str;
		}

		// public methods

		const DIMENSION size() const
		{
			DIMENSION sz = rank() ? 1 : 0;
			for (size_t i = 0; i < rank(); i++)
				sz = sz * _shape[i];
			return sz;
		}
		const DIMENSION rank() const
		{
			return _shape.size();
		}
		const std::vector<DIMENSION> shape() const
		{
			return _shape;
		}
		void reshape(const std::vector<size_t>& new_shape)
		{
			//ensure new_shape is same size as original shape
			if (new_shape.size() != _shape.size())
				throw "new size does not match existing tensor size";
			//ensure vector is of proper data type
			//ensure vector has length between 1 and 4, inclusive

			_shape = new_shape;
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
				for (size_t j = i + 1; j < rank(); j++)
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
			if (rank() > 1)
				indices.push_back(y);
			if (rank() > 2)
				indices.push_back(z);
			if (rank() > 3)
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
		std::string to_proto() // return proto string
		{
			std::string tensor_proto = "";
			return tensor_proto;
		}
	};
}

