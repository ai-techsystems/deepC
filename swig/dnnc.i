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
%include <typemaps.i>
%include <std_string.i>
%include <std_vector.i>
%include <std_shared_ptr.i>
%include <exception.i>
%include "core/tensor.h"
%include "dnnc_swig_externs.h"

%inline %{
typedef long unsigned int size_t;
%}
namespace std {
  %template(bvec) vector<bool>;
  %template(ivec) vector<int>;
  %template(lvec) vector<size_t>;
  %template(fvec) vector<float>;
}
%{
#include <core/tensor.h>
#include "dnnc_swig_externs.h"
extern std::vector<size_t> listTupleToVector_SizeT(PyObject *);
%}
%feature("python:slot", "mp_subscript", functype="binaryfunc") dnnc::tensor::__getitem__;
%feature("python:slot", "mp_ass_subscript", functype="objobjargproc") dnnc::tensor::__setitem__;

%extend dnnc::tensor {
  const T& __getitem__(PyObject* indices) {
    std::vector<size_t> vIndices = listTupleToVector_SizeT(indices);
    const T& item = $self->operator()(vIndices);
    return item;
  }
  void __setitem__(PyObject* indices, const T& data) {
    std::vector<size_t> vIndices = listTupleToVector_SizeT(indices);
    $self->load(data, vIndices);
    return ;
  }
}
%template(bTensor) dnnc::tensor<bool>;
%template(iTensor) dnnc::tensor<int>;
%template(fTensor) dnnc::tensor<float>;
%template(dTensor) dnnc::tensor<double>;
namespace std {
  %template(itvec) vector<dnnc::tensor<int> >;
  %template(ftvec) vector<dnnc::tensor<float> >;
}
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
      else:
        raise ValueError("unsupported data type {} \n".format(newType))
      
      return self

    bTensor.astype = astype;
    iTensor.astype = astype;
    fTensor.astype = astype;
    dTensor.astype = astype;
%}
