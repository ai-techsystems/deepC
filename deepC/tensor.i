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
// tensor.i
//

%ignore *::operator=;
%ignore *::operator==;
%ignore *::operator[];
%include "python/pyopers.swg"
%include "core/placeHolder.h"
%include "core/tensor.h"
%include "dnnc_swig_externs.h"
%{
#include <string>
#include <Python.h>
#include <object.h> // where Py_LT, Pt_GT are defined.
#include "core/tensor.h"
#include "operators/Add.h"
#include "operators/Abs.h"
#include "operators/And.h"
#include "operators/BitShift.h"
#include "operators/Equal.h"
#include "operators/FloorDiv.h"
#include "operators/Greater.h"
#include "operators/GreaterEqual.h"
#include "operators/Identity.h"
#include "operators/Less.h"
#include "operators/LessEqual.h"
#include "operators/Mod.h"
#include "operators/Mul.h"
#include "operators/Neg.h"
#include "operators/Not.h"
#include "operators/NotEqual.h"
#include "operators/Or.h"
#include "operators/Pow.h"
#include "operators/Remainder.h"
#include "operators/Sub.h"
#include "operators/TrueDiv.h"
#include "operators/Xor.h"
#include "dnnc_swig_externs.h"
extern std::vector<size_t> listTupleToVector_SizeT(PyObject *);
%}

%feature("python:slot",    "mp_subscript", functype="binaryfunc")             dnnc::tensor::__getitem__;
%feature("python:slot",    "mp_ass_subscript", functype="objobjargproc")      dnnc::tensor::__setitem__;
//%feature("python:slot",  "sq_slice", functype="ssizessizeargfunc")          __getslice__;
//%feature("python:slot",  "sq_ass_slice", functype="ssizessizeobjargproc")   __setslice__;
%feature("python:slot",    "sq_length", functype="lenfunc")                   dnnc::tensor::length;
%feature("python:slot",    "nb_bool", functype="inquiry")                     dnnc::tensor::operator bool;
%feature("python:slot",    "tp_repr", functype="reprfunc")                    dnnc::tensor::__str__;
%feature("python:slot",    "tp_str", functype="reprfunc")                     dnnc::tensor::__repr__;

//%implicitconv dnnc::tensor;
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
  std::string __str__() {
    return $self->to_string();
  }
  std::string __repr__() {
    return $self->to_string();
  }
  /* unary operators */
  %pybinoperator(__neg__, dnnc::tensor::__neg__, unaryfunc, nb_negative);
  dnnc::tensor<T> __neg__() {
    dnnc::Neg<T> op("pythonOp");
    return op.compute(*$self);
  }
  %pybinoperator(__pos__, dnnc::tensor::__pos__, unaryfunc, nb_positive);
  dnnc::tensor<T> __pos__() {
    return *$self;
  }
  %pybinoperator(__abs__, dnnc::tensor::__abs__, unaryfunc, nb_absolute);
  dnnc::tensor<T> __abs__() {
    dnnc::Abs<T> op("pythonOp");
    return op.compute(*$self);
  }
  %pybinoperator(__invert__, dnnc::tensor::__invert__, unaryfunc, nb_invert);
  dnnc::tensor<bool> __invert__() {
    dnnc::Not<bool, T> op("pythonOp");
    return op.compute(*$self);
  }



    /*  Binary Left Shift  */
  %pyinplaceoper(__lshift__, dnnc::tensor::__lshift__, binaryfunc, nb_lshift);
  dnnc::tensor<int> __lshift__(dnnc::tensor<int>& other) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __lshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }


    /*  Assignment Left Shift  */
  %pyinplaceoper(__ilshift__, dnnc::tensor::__ilshift__, binaryfunc, nb_inplace_lshift);
  dnnc::tensor<int> __ilshift__(dnnc::tensor<int>& other) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __ilshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }



    /*  Binary Right Shift  */
  %pyinplaceoper(__rshift__, dnnc::tensor::__rshift__, binaryfunc, nb_rshift);
  dnnc::tensor<int> __rshift__(dnnc::tensor<int>& other) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    std::string direction = "RIGHT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __rshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "RIGHT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }


    /*  Assignment Right Shift  */
  %pyinplaceoper(__irshift__, dnnc::tensor::__irshift__, binaryfunc, nb_inplace_rshift);
  dnnc::tensor<int> __irshift__(dnnc::tensor<int>& other) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    std::string direction = "RIGHT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __irshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "RIGHT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }



// <\/>
%pythoncode %{


def __getitem__(self, index):
  """
     getitem method for tensor.
  """
  def get_item_helper_int(item, axis):
    flag = 0
    start = item
    if (start < 0):
      start += self.shape()[axis]
    stop = start+1
    step = 1
    if(start >= self.shape()[axis]):
      errorMsg = "index value " + str(start) + " along axis " + str(axis) + " is beyond the size " + str(self.shape()[axis]) + " of input tensor along that axis"
      raise ValueError(errorMsg)
      flag = 1
    return start, stop, step, flag

  def get_item_helper_slice(item, axis):
    flag = 0   # to check if all index items are supported or not
    start = 0
    stop = self.shape()[axis]
    step = 1
    if type(item.step) == int:
      step = item.step
      if step == 0:
        errorMsg = "slice step cannot be zero"
        raise TypeError(errorMsg)
        flag = 1
      elif step < 0:
        start = self.shape()[axis] -1
        stop = -1
    elif str(type(item.step)).split("'")[1] == "NoneType":
      pass
    else:
      errorMsg = "step of " + str(type(item.step)) + " not supported!"
      raise TypeError(errorMsg)
      flag = 1
    if type(item.start) == int:
      start = item.start
      if (start < 0):
        start += self.shape()[axis]
    elif str(type(item.start)).split("'")[1] == "NoneType":
      pass
    else:
      errorMsg = "start of " + str(type(item.start)) + " not supported!"
      raise TypeError(errorMsg)
      flag = 1
    if type(item.stop) == int:
      stop = item.stop
      if (stop < 0):
        stop += self.shape()[axis]
    elif str(type(item.stop)).split("'")[1] == "NoneType":
      pass
    else:
      errorMsg = "stop of " + str(type(item.stop)) + " not supported!"
      raise TypeError(errorMsg)
      flag = 1
    # if(start > self.shape()[axis]):
    #   errorMsg = "index value " + str(start) + " along axis " + str(axis) + " is beyond the size " + str(self.shape()[axis]) + " of input tensor along that axis"
    #   raise IndexError(errorMsg)
    #   flag = 1
    # if(stop > self.shape()[axis]):
    #   errorMsg = "index value " + str(stop) + " along axis " + str(axis) + " is beyond the size " + str(self.shape()[axis]) + " of input tensor along that axis"
    #   raise IndexError(errorMsg)
    #   flag = 1
    # if (step < 0) and not (start > stop):
    #   errorMsg = "stop index " + str(stop) + " along axis " + str(axis) + " is greater than start index " + str(start) + " while step is negative"
    #   raise IndexError(errorMsg)
    #   flag = 1
    # elif (step > 0) and not (start < stop):
    #   errorMsg = "stop index " + str(stop) + " along axis " + str(axis) + " is smaller than start index " + str(start) + " while step is positive"
    #   raise IndexError(errorMsg)
    #   flag = 1

    return start, stop, step, flag

  if str(type(index)).split("'")[1] == "int":
    if self.rank() < 1:
      return self.copy()
    axis = 0
    start, stop, step, flag = get_item_helper_int(index, axis)
    if flag:
      return
    start = array([start]).asTypeInt()
    stop = array([stop]).asTypeInt()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeInt()
    if (self.rank() == 1):
      return self.data()[index]
    return slice(self, start, stop, axis, step).reshape(self.shape()[1:])

  elif str(type(index)).split("'")[1] == "slice":
    if self.rank() > 1 or self.rank() < 1:
      return self.copy()
    axis = 0
    start, stop, step, flag = get_item_helper_slice(index, axis)
    if flag:
      return
    start = array([start]).asTypeInt()
    stop = array([stop]).asTypeInt()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeInt()
    return slice(self, start, stop, axis, step)

  elif str(type(index)).split("'")[1] == "ellipsis":
    return self.copy()

  elif str(type(index)).split("'")[1] == "tuple":
    if (len(index) > self.rank()):
      errorMsg = "Takes maximum " + str(self.rank()) + "arguments, " + str(len(index)) + " were given!"
      raise IndexError(errorMsg)
      return
    elif Ellipsis not in index and (len(index) < self.rank()):
      errorMsg = "Takes minimum " + str(self.rank()) + "arguments, " + str(len(index)) + " were given!"
      raise IndexError(errorMsg)
      return

    # checks if any float or bool or complex is not present
    if any(isinstance(x,(bool,float,complex)) for x in index):
      errorMsg = "Restrict to only integers as a slicing argument!"
      raise ValueError(errorMsg)
      return

    start_list = []
    stop_list = []
    step_list = []
    axis_list = []
    axis = -1   # -1 for starting axis as 0 in the next loops
    reshape_list = []   # reshape list to reshape
    replace_start = replace_stop = 0   # replace ellipsis with slice methods by index

    if Ellipsis in index:
      if (index.count(Ellipsis) > 1):
        errorMsg = str(index.count(Ellipsis)) + " 'Ellipsis' found, maximum 1 is supported!"
        raise IndexError(errorMsg)
        return
      elif (index.count(Ellipsis) == 1):
        non_ellipsis_count = 0
        for item in index:
          if str(type(item)).split("'")[1] == "int" or str(type(item)).split("'")[1] == "slice":
            non_ellipsis_count += 1
        # replace holds start and stop index which will be replaced by slice method in place of ellipsis
        replace_start = index.index(Ellipsis)
        replace_stop = replace_start + self.rank() - non_ellipsis_count
      else:
        errorMsg = "Error occured while handling ellipsis!"
        raise ValueError(errorMsg)
        return

    for item in index:
      axis += 1
      if str(type(item)).split("'")[1] == "ellipsis":
        while (axis >= replace_start and axis < replace_stop):
          start = 0
          stop = self.shape()[axis]
          step = 1
          start_list.append(start)
          stop_list.append(stop)
          step_list.append(step)
          axis_list.append(axis)
          reshape_list.append(1)  # This shape will be taken
          axis += 1
        axis -= 1   # recovering from last axis increment
      elif str(type(item)).split("'")[1] == "int":
        start, stop, step, flag = get_item_helper_int(item, axis)
        if flag:
          return
        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)
        axis_list.append(axis)
        reshape_list.append(0)  # This shape will not be taken
      elif str(type(item)).split("'")[1] == "slice":
        start, stop, step, flag = get_item_helper_slice(item, axis)
        if flag:
          return
        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)
        axis_list.append(axis)
        reshape_list.append(1)  # This shape will be taken
      else:
        errorMsg = "Doesn't support " + str(item) + " of " + str(type(item)) + " as a slicing argument!"
        raise TypeError(errorMsg)
        return

    while (axis < self.rank()-1):
      axis += 1
      start = 0
      stop = self.shape()[axis]
      step = 1
      start_list.append(start)
      stop_list.append(stop)
      step_list.append(step)
      axis_list.append(axis)
      reshape_list.append(1)  # This shape will be taken

    start_list = array(start_list).asTypeInt()
    stop_list = array(stop_list).asTypeInt()
    axis_list = array(axis_list).asTypeInt()
    step_list = array(step_list).asTypeInt()

    # print("test start list :  ", start_list)
    # print("test stop list :  ", stop_list)
    # print("test axis list :  ", axis_list)
    # print("test step list :  ", step_list)

    result = slice(self, start_list, stop_list, axis_list, step_list)

    if 0 in reshape_list:
      if not 1 in reshape_list:
        return result.data()[0]
      return (result.reshape([x for x, y in zip(result.shape(), reshape_list) if y == 1]))

    return result

  else :
    errorMsg = "Doesn't support " + str(index) + " of " + str(type(index)) + " as a slicing argument!"
    raise TypeError(errorMsg)

  return intTensor()



def __setitem__(self, index, input_tensor):
  """
     setitem method for tensor.
  """
  value_tensor = ""  # declaring value_tensor which will be a deep copy of input_tensor
  if "Tensor" not in str(type(input_tensor)).split("'")[1]:
    try:
      value_tensor = array([input_tensor])  # passing single number as tensor of length 1
    except:
      errorMsg = "could not convert " + str(value_tensor) + " of type " + str(type(value_tensor)) + " to dnnc tensor"
      raise ValueError(errorMsg)
      return
  else:
    value_tensor = input_tensor.copy()
  input_tensor_shape = value_tensor.shape()  # storing input tensor shape
  def set_item_helper_int(item, axis):
    flag = 0
    start = item
    if (start < 0):
      start += self.shape()[axis]
    stop = start+1
    step = 1
    if(start >= self.shape()[axis]):
      errorMsg = "index value " + str(start) + " along axis " + str(axis) + " is beyond the size " + str(self.shape()[axis]) + " of input tensor along that axis"
      raise ValueError(errorMsg)
      flag = 1
    return start, stop, step, flag

  def set_item_helper_slice(item, axis):
    flag = 0   # to check if all index items are supported or not
    start = 0
    stop = self.shape()[axis]
    step = 1
    if type(item.step) == int:
      step = item.step
      if step == 0:
        errorMsg = "slice step cannot be zero"
        raise TypeError(errorMsg)
        flag = 1
      elif step < 0:
        start = self.shape()[axis] -1
        stop = -1
    elif str(type(item.step)).split("'")[1] == "NoneType":
      pass
    else:
      errorMsg = "step of " + str(type(item.step)) + " not supported!"
      raise TypeError(errorMsg)
      flag = 1
    if type(item.start) == int:
      start = item.start
      if (start < 0):
        start += self.shape()[axis]
    elif str(type(item.start)).split("'")[1] == "NoneType":
      pass
    else:
      errorMsg = "start of " + str(type(item.start)) + " not supported!"
      raise TypeError(errorMsg)
      flag = 1
    if type(item.stop) == int:
      stop = item.stop
      if (stop < 0):
        stop += self.shape()[axis]
    elif str(type(item.stop)).split("'")[1] == "NoneType":
      pass
    else:
      errorMsg = "stop of " + str(type(item.stop)) + " not supported!"
      raise TypeError(errorMsg)
      flag = 1
    # if(start > self.shape()[axis]):
    #   errorMsg = "index value " + str(start) + " along axis " + str(axis) + " is beyond the size " + str(self.shape()[axis]) + " of input tensor along that axis"
    #   raise IndexError(errorMsg)
    #   flag = 1
    # if(stop > self.shape()[axis]):
    #   errorMsg = "index value " + str(stop) + " along axis " + str(axis) + " is beyond the size " + str(self.shape()[axis]) + " of input tensor along that axis"
    #   raise IndexError(errorMsg)
    #   flag = 1
    # if (step < 0) and not (start > stop):
    #   errorMsg = "stop index " + str(stop) + " along axis " + str(axis) + " is greater than start index " + str(start) + " while step is negative"
    #   raise IndexError(errorMsg)
    #   flag = 1
    # elif (step > 0) and not (start < stop):
    #   errorMsg = "stop index " + str(stop) + " along axis " + str(axis) + " is smaller than start index " + str(start) + " while step is positive"
    #   raise IndexError(errorMsg)
    #   flag = 1

    return start, stop, step, flag

  if str(type(index)).split("'")[1] == "int":
    if self.rank() < 1:
      errorMsg = "cannot set items to a null tensor"
      raise TypeError(errorMsg)
      return
    axis = 0
    start, stop, step, flag = set_item_helper_int(index, axis)
    if flag:
      return
    start = array([start]).asTypeInt()
    stop = array([stop]).asTypeInt()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeInt()
    result = slice(self, start, stop, axis, step)

    try:
      if (result.len() == value_tensor.len()):
        value_tensor = value_tensor.reshape(result.shape())
    except:
      pass
    if result.shape() == value_tensor.shape():
      set_slice(self, value_tensor, start, stop, axis, step)
    else:
      errorMsg = "could not broadcast input array from shape "+str(input_tensor_shape)+" into shape "+str(result.shape())
      raise ValueError(errorMsg)
    return

  elif str(type(index)).split("'")[1] == "slice":
    if self.rank() < 1:
      self = value_tensor
      return

    axis = 0
    start, stop, step, flag = set_item_helper_slice(index, axis)
    if flag:
      return
    start = array([start]).asTypeInt()
    stop = array([stop]).asTypeInt()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeInt()
    result = slice(self, start, stop, axis, step)

    try:
      if (result.len() == value_tensor.len()):
        value_tensor = value_tensor.reshape(result.shape())
    except:
      pass
    if result.shape() == value_tensor.shape():
      set_slice(self, value_tensor, start, stop, axis, step)
    else:
      errorMsg = "could not broadcast input array from shape "+str(input_tensor_shape)+" into shape "+str(result.shape())
      raise ValueError(errorMsg)
    return

  elif str(type(index)).split("'")[1] == "ellipsis":
    if self.shape() != value_tensor.shape():
      errorMsg = "could not broadcast input array from shape "+str(input_tensor_shape)+" into shape "+str(self.shape())
      raise ValueError(errorMsg)
      return
    elif (self.rank() < 1) or (self.shape() == value_tensor.shape()):
      self = value_tensor
      return

  elif str(type(index)).split("'")[1] == "tuple":
    if (len(index) > self.rank()):
      errorMsg = "Takes maximum " + str(self.rank()) + "arguments, " + str(len(index)) + " were given!"
      raise IndexError(errorMsg)
      return
    elif Ellipsis not in index and (len(index) < self.rank()):
      errorMsg = "Takes minimum " + str(self.rank()) + "arguments, " + str(len(index)) + " were given!"
      raise IndexError(errorMsg)
      return

    # checks if any float or bool or complex is not present
    if any(isinstance(x,(bool,float,complex)) for x in index):
      errorMsg = "Restrict to only integers as a slicing argument!"
      raise ValueError(errorMsg)
      return

    start_list = []
    stop_list = []
    step_list = []
    axis_list = []
    axis = -1   # -1 for starting axis as 0 in the next loops
    reshape_list = []   # reshape list to reshape
    replace_start = replace_stop = 0   # replace ellipsis with slice methods by index

    if Ellipsis in index:
      if (index.count(Ellipsis) > 1):
        errorMsg = str(index.count(Ellipsis)) + " 'Ellipsis' found, maximum 1 is supported!"
        raise IndexError(errorMsg)
        return
      elif (index.count(Ellipsis) == 1):
        non_ellipsis_count = 0
        for item in index:
          if str(type(item)).split("'")[1] == "int" or str(type(item)).split("'")[1] == "slice":
            non_ellipsis_count += 1
        # replace holds start and stop index which will be replaced by slice method in place of ellipsis
        replace_start = index.index(Ellipsis)
        replace_stop = replace_start + self.rank() - non_ellipsis_count
      else:
        errorMsg = "Error occured while handling ellipsis!"
        raise ValueError(errorMsg)
        return

    for item in index:
      axis += 1
      if str(type(item)).split("'")[1] == "ellipsis":
        while (axis >= replace_start and axis < replace_stop):
          start = 0
          stop = self.shape()[axis]
          step = 1
          start_list.append(start)
          stop_list.append(stop)
          step_list.append(step)
          axis_list.append(axis)
          reshape_list.append(1)  # This shape will be taken
          axis += 1
        axis -= 1   # recovering from last axis increment
      elif str(type(item)).split("'")[1] == "int":
        start, stop, step, flag = set_item_helper_int(item, axis)
        if flag:
          return
        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)
        axis_list.append(axis)
        reshape_list.append(0)  # This shape will not be taken
      elif str(type(item)).split("'")[1] == "slice":
        start, stop, step, flag = set_item_helper_slice(item, axis)
        if flag:
          return
        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)
        axis_list.append(axis)
        reshape_list.append(1)  # This shape will be taken
      else:
        errorMsg = "Doesn't support " + str(item) + " of " + str(type(item)) + " as a slicing argument!"
        raise TypeError(errorMsg)
        return

    while (axis < self.rank()-1):
      axis += 1
      start = 0
      stop = self.shape()[axis]
      step = 1
      start_list.append(start)
      stop_list.append(stop)
      step_list.append(step)
      axis_list.append(axis)
      reshape_list.append(1)  # This shape will be taken

    start_list = array(start_list).asTypeInt()
    stop_list = array(stop_list).asTypeInt()
    axis_list = array(axis_list).asTypeInt()
    step_list = array(step_list).asTypeInt()

    # print("test start list :  ", start_list)
    # print("test stop list :  ", stop_list)
    # print("test axis list :  ", axis_list)
    # print("test step list :  ", step_list)

    result = slice(self, start_list, stop_list, axis_list, step_list)

    if 0 in reshape_list:
      if not 1 in reshape_list:
        reshape_list = [1,]
      result = result.reshape([x for x, y in zip(result.shape(), reshape_list) if y == 1])

    try:
      if (result.len() == value_tensor.len()):
        value_tensor = value_tensor.reshape(result.shape())
    except:
      pass
    if result.shape() == value_tensor.shape():
      set_slice(self, value_tensor, start_list, stop_list, axis_list, step_list)
    else:
      errorMsg = "could not broadcast input array from shape "+str(input_tensor_shape)+" into shape "+str(result.shape())
      raise ValueError(errorMsg)
    return
  else :
    errorMsg = "Doesn't support " + str(index) + " of " + str(type(index)) + " as a slicing argument!"
    raise TypeError(errorMsg)

  return


def __add__(self, other):
  return add(self, other)

def __radd__(self, other):
  return add(other, self)

def __iadd__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return add(self, other)

def __sub__(self, other):
  return sub(self, other)

def __rsub__(self, other):
  return sub(other, self)

def __isub__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return sub(self, other)

def __mul__(self, other):
  return mul(self, other)

def __rmul__(self, other):
  return mul(other, self)

def __imul__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return mul(self, other)

def __floordiv__(self, other):
  return floor_div(self, other)

def __rfloordiv__(self, other):
  return floor_div(other, self)

def __ifloordiv__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return floor_div(self, other)

def __truediv__(self, other):
  return true_div(self, other)

def __rtruediv__(self, other):
  return true_div(other, self)

def __itruediv__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return true_div(self, other)

def __mod__(self, other):
  return remainder(self, other)

def __rmod__(self, other):
  return remainder(other, self)

def __imod__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return remainder(self, other)

def __pow__(self, other):
  return power(self, other)

def __rpow__(self, other):
  return power(other, self)

def __ipow__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return power(self, other)

def __and__(self, other):
  return bitwise_and(self, other)

def __rand__(self, other):
  return bitwise_and(other, self)

def __iand__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return bitwise_and(self, other)

def __or__(self, other):
  return bitwise_or(self, other)

def __ror__(self, other):
  return bitwise_or(other, self)

def __ior__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return bitwise_or(self, other)

def __xor__(self, other):
  return bitwise_xor(self, other)

def __rxor__(self, other):
  return bitwise_xor(other, self)

def __ixor__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = {'double': 16, 'float': 14, 'size_t': 12, 'long': 10, 'int': 8, 'short': 6, 'bool': 4, 'char': 2}
  left_operand_dtype = right_operand_dtype = ""
  if "Tensor" in str(type(self)):
    left_operand_dtype = str(type(self)).split(".")[-1].split("Tensor")[0]
  else:
    left_operand_dtype = str(type(self)).split("'")[1]
  if "Tensor" in str(type(other)):
    right_operand_dtype = str(type(other)).split(".")[-1].split("Tensor")[0]
  else:
    right_operand_dtype = str(type(other)).split("'")[1]
  if (dtype_precedence_dict[left_operand_dtype] < dtype_precedence_dict[right_operand_dtype]):
    errorMsg = "cannot modify left hand operand datatype."
    raise TypeError(errorMsg)
  return bitwise_xor(self, other)

def __eq__(self, other):
  return equal(self, other)

def __lt__(self, other):
  return less(self, other)

def __gt__(self, other):
  return greater(self, other)

def __ne__(self, other):
  return not_equal(self, other)

def __le__(self, other):
  return less_equal(self, other)

def __ge__(self, other):
  return greater_equal(self, other)

%}// <\/>

%pythoncode %{
def numpy(self) :
  """
     convert tensor to numpy array.
  """
  import numpy as np
  return np.array(self.data()).reshape(self.shape())

def len(self):
  """
     return length of tensor.
  """
  return self.length()

def __iter__(self):
  """
     iterates over the tensor.
  """
  axis = 0
  while (axis < self.shape()[0]):
    yield self[axis]
    axis += 1

def __int__(self):
  """
     convert tensor to int tensor.
  """
  if (self.len() > 1):
    raise TypeError("only length-1 tensor can be converted to Python scalars")
  elif(self.len() == 0):
    return
  return self.asTypeInt()

def __float__(self):
  """
     convert tensor to float tensor.
  """
  if (self.len() > 1):
    raise TypeError("only length-1 tensor can be converted to Python scalars")
  elif(self.len() == 0):
    return
  return self.asTypeFloat()

def __bool__(self):
  """
     convert tensor to bool tensor.
  """
  if (self.len() > 1):
    raise ValueError("The truth value of an array with more than one element is ambiguous. Use dc.any() or dc.all()")
  elif(self.len() == 0):
    return False
  return self.asTypeBool()

%}


%feature("shadow") reshape(std::vector<size_t> &shape) %{
def reshape(self, *args):
  """
    reshape tensor to a new shape permitting same length of old and new shape.
  """
  new_shape = []
  for arg in args:
    if isinstance(arg, int):
      new_shape.append(arg)
    else:
      return $action (self, vectorSizeT(arg))

  if len(new_shape):
    return $action (self, vectorSizeT(new_shape))

  return $action (args)
%}

}
%template(boolplaceHolder)   dnnc::placeHolder<bool>;
%template(intplaceHolder)    dnnc::placeHolder<int>;
%template(uLongplaceHolder)  dnnc::placeHolder<size_t>;
%template(floatplaceHolder)  dnnc::placeHolder<float>;
%template(doubleplaceHolder) dnnc::placeHolder<double>;
%template(boolTensor)   dnnc::tensor<bool>;
%template(intTensor)    dnnc::tensor<int>;
%template(uLongTensor)  dnnc::tensor<size_t>;
%template(floatTensor)  dnnc::tensor<float>;
%template(doubleTensor) dnnc::tensor<double>;
namespace std {
  %template(vectorTensorBool) vector<dnnc::tensor<bool> >;
  %template(vectorTensorInt) vector<dnnc::tensor<int> >;
  %template(vectorTensorSizeT) vector<dnnc::tensor<size_t> >;
  %template(vectorTensorFloat) vector<dnnc::tensor<float> >;
}
