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
        dequantize_linear(dnnc::tensor<float>& a, dnnc::tensor<float>& b, dnnc::tensor<float>& c) ;
extern dnnc::tensor<float>  \
        div(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        elu(dnnc::tensor<float>& a, float alpha=1.0) ;
extern dnnc::tensor<bool>  \
        equal(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        erf(dnnc::tensor<float>& a) ;
extern dnnc::tensor<float>  \
        exp(dnnc::tensor<float>& a) ;
extern dnnc::tensor<float>  \
        eye_like(dnnc::tensor<float>& a, int k=0) ;
extern dnnc::tensor<float>  \
        flatten(dnnc::tensor<float>& a, int axis=1) ;
extern dnnc::tensor<float>  \
        floor(dnnc::tensor<float>& a) ;
extern dnnc::tensor<float>  \
        gemm(dnnc::tensor<float>& a, dnnc::tensor<float>& b, dnnc::tensor<float>& c, float alpha=1.0,float beta=1.0, int transA=0, int transB=0) ;
extern dnnc::tensor<float>  \
        thresholded_relu(dnnc::tensor<float>& input);
extern dnnc::tensor<bool>  \
        less(dnnc::tensor<float>& a, dnnc::tensor<float>& b);
extern dnnc::tensor<float>  \
        log(dnnc::tensor<float>& input);
extern dnnc::tensor<float>  \
        logsoftmax(dnnc::tensor<float>& input);
extern dnnc::tensor<float>  \
        lpnormalization(dnnc::tensor<float>& input);
extern dnnc::tensor<int>  \
        matmulinteger(dnnc::tensor<int>& a, dnnc::tensor<int>& b);
extern dnnc::tensor<float>  \
        transpose(dnnc::tensor<float> &a) ;
extern dnnc::tensor<float>  \
        global_average_pool(dnnc::tensor<float>& input);
extern dnnc::tensor<float>  \
        global_lp_pool(dnnc::tensor<float>& input,int p=2);
extern dnnc::tensor<float>  \
        global_max_pool(dnnc::tensor<float>& a);
extern dnnc::tensor<bool>  \
        greater(dnnc::tensor<float>& a,dnnc::tensor<float>& b);
extern dnnc::tensor<float>  \
        hardmax(dnnc::tensor<float>& a,int axis=0);
extern dnnc::tensor<float>  \
        hardsigmoid(dnnc::tensor<float>& a,float alpha=0.2,float beta=0.5);
extern dnnc::tensor<float>  \
        identity(dnnc::tensor<float>& a);
extern dnnc::tensor<bool>  \
        isinf(dnnc::tensor<float>& a,int detect_positive=1,int detect_negative=1);
extern dnnc::tensor<bool>  \
        isnan(dnnc::tensor<float>& a);
extern dnnc::tensor<float>  \
        leakyrelu(dnnc::tensor<float>& a,float alpha=0.01);
extern dnnc::tensor<float>  \
        instancenormalization(dnnc::tensor<float>& input,dnnc::tensor<float>& scale,dnnc::tensor<float>& B,float epsilon=1e-5);
extern dnnc::tensor<float>  \
        lrn(dnnc::tensor<float>& input,int size,float alpha = 0.0001,float beta = 0.75, float bias = 1.0);


extern dnnc::tensor<float> array(PyObject*);
extern dnnc::tensor<float> arange(size_t stop, size_t start=0, size_t step=1);
extern dnnc::tensor<float> empty(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> zeros(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> ones(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> random(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> reshape(dnnc::tensor<float>&, PyObject*) ;
%}
%feature("python:slot", "mp_subscript", functype="binaryfunc") dnnc::tensor::__getitem__;
%feature("python:slot", "mp_ass_subscript", functype="objobjargproc") dnnc::tensor::__setitem__;

%extend dnnc::tensor {
  const T& __getitem__(int i, int j=0, int k=0, int l=0) {
      const T& item = $self->operator()(i,j,k,l);
      return item;
  }
  void __setitem__(int i, const T& data) {
      $self->load(data, i);
      return ;
  }
}
%template(bTensor) dnnc::tensor<bool>;
%template(iTensor) dnnc::tensor<int>;
%template(fTensor) dnnc::tensor<float>;
%template(dTensor) dnnc::tensor<double>;
%pythoncode %{
    def astype(self, newType):
      if ( newType == "double" ) :
        return self.asTypeDouble();
      elif ( newType == "float" ) :
        return self.asTypeFloat();
      elif ( newType == "int" ) :
        return self.asTypeInt();
      elif ( newType == "bool" ) :
        return self.asTypeBool();
      
      return self

    bTensor.astype = astype;
    iTensor.astype = astype;
    fTensor.astype = astype;
    dTensor.astype = astype;
%}

extern dnnc::tensor<float>
        array(size_t x,     size_t y = 0,
              size_t z = 0, size_t w = 0) ;
extern dnnc::tensor<float>  \
        matmul(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        add(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        dequantize_linear(dnnc::tensor<float>& a, dnnc::tensor<float>& b, dnnc::tensor<float>& c) ;
extern dnnc::tensor<float>  \
        div(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        elu(dnnc::tensor<float>& a, float alpha=1.0) ;
extern dnnc::tensor<bool>  \
        equal(dnnc::tensor<float>& a, dnnc::tensor<float>& b) ;
extern dnnc::tensor<float>  \
        erf(dnnc::tensor<float>& a) ;
extern dnnc::tensor<float>  \
        exp(dnnc::tensor<float>& a) ;
extern dnnc::tensor<float>  \
        eye_like(dnnc::tensor<float>& a, int k=0) ;
extern dnnc::tensor<float>  \
        flatten(dnnc::tensor<float>& a, int axis=1) ;
extern dnnc::tensor<float>  \
        floor(dnnc::tensor<float>& a) ;
extern dnnc::tensor<float>  \
        gemm(dnnc::tensor<float>& a, dnnc::tensor<float>& b, dnnc::tensor<float>& c, float alpha=1.0,float beta=1.0, int transA=0, int transB=0) ;
extern dnnc::tensor<float>  \
        thresholded_relu(dnnc::tensor<float>& input);
extern dnnc::tensor<bool>  \
        less(dnnc::tensor<float>& a, dnnc::tensor<float>& b);
extern dnnc::tensor<float>  \
        log(dnnc::tensor<float>& input);
extern dnnc::tensor<float>  \
        logsoftmax(dnnc::tensor<float>& input);
extern dnnc::tensor<float>  \
        lpnormalization(dnnc::tensor<float>& input);
extern dnnc::tensor<int>  \
        matmulinteger(dnnc::tensor<int>& a, dnnc::tensor<int>& b);
extern dnnc::tensor<float>  \
        transpose(dnnc::tensor<float> &a) ;
extern dnnc::tensor<float>  \
        global_average_pool(dnnc::tensor<float>& input);
extern dnnc::tensor<float>  \
        global_lp_pool(dnnc::tensor<float>& input,int p=2);
extern dnnc::tensor<float>  \
        global_max_pool(dnnc::tensor<float>& a);
extern dnnc::tensor<bool>  \
        greater(dnnc::tensor<float>& a,dnnc::tensor<float>& b);
extern dnnc::tensor<float>  \
        hardmax(dnnc::tensor<float>& a,int axis=0);
extern dnnc::tensor<float>  \
        hardsigmoid(dnnc::tensor<float>& a,float alpha=0.2,float beta=0.5);
extern dnnc::tensor<float>  \
        identity(dnnc::tensor<float>& a);
extern dnnc::tensor<bool>  \
        isinf(dnnc::tensor<float>& a,int detect_positive=1,int detect_negative=1);
extern dnnc::tensor<bool>  \
        isnan(dnnc::tensor<float>& a);
extern dnnc::tensor<float>  \
        leakyrelu(dnnc::tensor<float>& a,float alpha=0.01);
extern dnnc::tensor<float>  \
        instancenormalization(dnnc::tensor<float>& input,dnnc::tensor<float>& scale,dnnc::tensor<float>& B,float epsilon=1e-5);
extern dnnc::tensor<float>  \
        lrn(dnnc::tensor<float>& input,int size,float alpha = 0.0001,float beta = 0.75, float bias = 1.0);

extern dnnc::tensor<float> array(PyObject* objects);
extern dnnc::tensor<float> arange(size_t stop, size_t start=0, size_t step=1);
extern dnnc::tensor<float> empty(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> zeros(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> ones(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> random(size_t x, size_t y = 0, size_t z = 0, size_t w = 0);
extern dnnc::tensor<float> reshape(dnnc::tensor<float>&, PyObject*) ;
