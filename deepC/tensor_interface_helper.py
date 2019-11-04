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
    start = item
    stop = item+1
    step = 1
    return start, stop, step

  def get_item_helper_slice(index, axis):
    start = 0
    stop = self.shape()[axis]
    step = 1
    if type(item.start) == int:
      start = item.start
    elif str(type(item.start)).split("'")[1] == "NoneType":
      pass
    else:
      print("item.start ", type(item.start), "not supported!")
    if type(item.stop) == int:
      stop = item.stop
    elif str(type(item.stop)).split("'")[1] == "NoneType":
      pass
    else:
      print("item.stop ", type(item.stop), "not supported!")
    if type(item.step) == int:
      step = item.step
    elif str(type(item.step)).split("'")[1] == "NoneType":
      pass
    else:
      print("item.step ", type(item.step), "not supported!")
    return start, stop, step


  if str(type(index)).split("'")[1] == "int":
    axis = 0
    start, stop, step = get_item_helper_int(index, axis)
    start = array([start]).asTypeULong()
    stop = array([stop]).asTypeULong()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeULong()
    return slice(self, start, stop, axis, step)
    # return self.getitem_helper(index)

  elif isinstance(index, list) or isinstance(index, tuple):
    start_list = []
    stop_list = []
    step_list = []
    axis_list = []
    for axis,item in enumerate(index):
      if str(type(item)).split("'")[1] == "int":
        start, stop, step = get_item_helper_int(item, axis)
        # print("int", start, stop, axis, step)
        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)
        axis_list.append(axis)
      elif str(type(item)).split("'")[1] == "slice":
        start, stop, step = get_item_helper_slice(item, axis)
        # print("slice", start, stop, axis, step)
        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)
        axis_list.append(axis)
      
      else:
        print("else", type(item))
    
    # print(start_list, stop_list, axis_list, step_list)
    start_list = array(start_list).asTypeULong()
    stop_list = array(stop_list).asTypeULong()
    axis_list = array(axis_list).asTypeInt()
    step_list = array(step_list).asTypeULong()
    return slice(self, start_list, stop_list, axis_list, step_list)
  
  elif str(type(index)).split("'")[1] == "slice":
    axis = 0
    start, stop, step = get_item_helper_slice(index, axis)
    start = array([start]).asTypeULong()
    stop = array([stop]).asTypeULong()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeULong()
    return slice(self, start, stop, axis, step)
    
  else :
    print("else")
    print(type(index))

  return

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
    raise TypeError("cannot modify left hand operand datatype.")
  return <operator>(self, other)
'''
  elif flag == "comparison":
    s = '''
def __<operand>__(self, other):
  return <operator>(self, other)
'''
  s = s.replace("<operator>",dc_operator).replace("<operand>",operator_python)
  return s
