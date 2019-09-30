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

%module dnnc
%include <typemaps.i>
%include <std_string.i>
%include <std_vector.i>
%include <std_shared_ptr.i>
%include <exception.i>

%inline %{
typedef long unsigned int size_t;
typedef std::string String;
%}
namespace std {
  %template(vectorBool) vector<bool>;
  %template(vectorStr) vector<String>;
  %template(vectorInt) vector<int>;
  %template(vectorSizeT) vector<size_t>;
  %template(vectorFloat) vector<float>;
}

%include "core/datatypes.h"
%include "core/macros.h"
%include "tensor.i"
%include "graph.i"

%{
%}

