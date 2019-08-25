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
// dnnc.i
//
#define MODE DEBUG

%exception
{
 try
 {
   $action
 }
 catch (const std::runtime_error& e) {
   SWIG_exception(SWIG_RuntimeError, e.what());
 } 
 catch (const std::invalid_argument& e) {
   SWIG_exception(SWIG_ValueError, e.what());
 }
 catch (const std::out_of_range& e) {
   SWIG_exception(SWIG_IndexError, e.what());
 }
 catch (const std::logic_error& e) {
   SWIG_exception(SWIG_IndexError, e.what());
 }
 catch (const std::bad_alloc& e) {
   PyErr_NoMemory();
   SWIG_exception(SWIG_MemoryError, e.what());
 }
 catch (...) { 
   SWIG_exception(SWIG_RuntimeError, "unknown exception");
 }
}
%ignore *::operator=;
%ignore *::operator[];

%module dnnc
%include "typemaps.i"
%include <std_string.i>
%include <std_vector.i>
%include <std_shared_ptr.i>
%include <exception.i>
%include <core/tensor.h>

%inline %{
typedef long unsigned int size_t;
%}
namespace std {
  %template(ivec) vector<size_t>;
  %template(fvec) vector<float>;
}
%{
#include <core/tensor.h>
extern dnnc::tensor<float>  \
        array(size_t x,     size_t y = 0,  \
              size_t z = 0, size_t w = 0) ;
extern dnnc::tensor<float>  \
        matmul(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        add(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        thresholded_relu(dnnc::tensor<float>& input);
extern dnnc::tensor<float> array(PyObject*);
extern dnnc::tensor<float> arange(size_t stop, size_t start=0, size_t step=1);
extern dnnc::tensor<float> empty(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> zeros(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> ones(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> random(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> reshape(dnnc::tensor<float>&, PyObject*) ;
%}
%extend dnnc::tensor<T> {
  T __getitem__(size_t i) {
    return (*$self)[i];
  }
}
%template(iTensor) dnnc::tensor<int>;
%template(fTensor) dnnc::tensor<float>;
%template(dTensor) dnnc::tensor<double>;

extern dnnc::tensor<float>
        array(size_t x,     size_t y = 0, 
              size_t z = 0, size_t w = 0) ;
extern dnnc::tensor<float>  \
        matmul(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        add(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        thresholded_relu(dnnc::tensor<float>& input);
extern dnnc::tensor<float> array(PyObject* objects);
extern dnnc::tensor<float> arange(size_t stop, size_t start=0, size_t step=1);
extern dnnc::tensor<float> empty(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> zeros(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> ones(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> random(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> reshape(dnnc::tensor<float>&, PyObject*) ;
