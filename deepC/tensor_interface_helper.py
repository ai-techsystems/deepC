# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
#
# This file is part of DNN compiler maintained at
# https://github.com/ai-techsystems/dnnCompiler

# This file is required by ../dnnCompiler/swig/op_gen.py


def build_python_file():
  s = """

%pythoncode %{

def __getitem__(self, index):

  def get_item_helper_int(item, axis):
    flag = 0
    start = item
    stop = item+1
    step = 1
    if(start >= self.shape()[axis] or start < - self.shape()[axis]):
      errorMsg = "index value " + str(start) + " along axis " + str(axis) + " is beyond the size " + str(self.shape()[axis]) + " of input tensor along that axis"
      raise ValueError(errorMsg)
      flag = 1
    return start, stop, step, flag

  def get_item_helper_slice(item, axis):
    flag = 0   # to check if all index items are supported or not
    start = 0
    stop = self.shape()[axis]
    step = 1
    if type(item.start) == int:
      start = item.start
    elif str(type(item.start)).split("'")[1] == "NoneType":
      pass
    else:
      errorMsg = "start of " + str(type(item.start)) + " not supported!"
      raise TypeError(errorMsg)
      flag = 1
    if type(item.stop) == int:
      stop = item.stop
    elif str(type(item.stop)).split("'")[1] == "NoneType":
      pass
    else:
      errorMsg = "stop of " + str(type(item.stop)) + " not supported!"
      raise TypeError(errorMsg)
      flag = 1
    if type(item.step) == int:
      step = item.step
    elif str(type(item.step)).split("'")[1] == "NoneType":
      pass
    else:
      errorMsg = "step of " + str(type(item.step)) + " not supported!"
      raise TypeError(errorMsg)
      flag = 1
    if(start >= self.shape()[axis] or start < - self.shape()[axis]):
      errorMsg = "index value " + str(start) + " along axis " + str(axis) + " is beyond the size " + str(self.shape()[axis]) + " of input tensor along that axis"
      raise ValueError(errorMsg)
      flag = 1
    if(stop >= self.shape()[axis]+1 or stop < - (self.shape()[axis]+1)):
      errorMsg = "index value " + str(stop) + " along axis " + str(axis) + " is beyond the size " + str(self.shape()[axis]) + " of input tensor along that axis"
      raise ValueError(errorMsg)
      flag = 1
    return start, stop, step, flag

  if str(type(index)).split("'")[1] == "int":
    if self.rank() < 1:
      return self.copy()
    axis = 0
    start, stop, step, flag = get_item_helper_int(index, axis)
    if flag:
      return
    start = array([start]).asTypeULong()
    stop = array([stop]).asTypeULong()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeULong()
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
    start = array([start]).asTypeULong()
    stop = array([stop]).asTypeULong()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeULong()
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

    start_list = array(start_list).asTypeULong()
    stop_list = array(stop_list).asTypeULong()
    axis_list = array(axis_list).asTypeInt()
    step_list = array(step_list).asTypeULong()
    result = slice(self, start_list, stop_list, axis_list, step_list)

    if 0 in reshape_list:
      if not 1 in reshape_list:
        return result.data()[0]
      return (result.reshape([x for x, y in zip(result.shape(), reshape_list) if y == 1]))
    
    return result

  else :
    errorMsg = "Doesn't support " + str(index) + " of " + str(type(index)) + " as a slicing argument!"
    raise TypeError(errorMsg)

  return empty(0)



def __iter__(self):
  axis = 0
  while (axis < self.shape()[0]):
    yield self[axis]
    axis += 1

"""
  return s


def overload_python_operator(dc_operator, operator_python, dtype_precedence_dict, flag):
  s = ""
  if flag == "logical" or flag == "binary":
    s = '''
def __<operand>__(self, other):
  return <operator>(self, other)

def __r<operand>__(self, other):
  return <operator>(other, self)

def __i<operand>__(self, other):
  """
    making sure left hand operand is immutable
  """
  dtype_precedence_dict = '''
    s += str(dtype_precedence_dict) + '''
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
  return <operator>(self, other)
'''
  elif flag == "comparison":
    s = '''
def __<operand>__(self, other):
  return <operator>(self, other)
'''
  s = s.replace("<operator>",dc_operator).replace("<operand>",operator_python)
  return s
