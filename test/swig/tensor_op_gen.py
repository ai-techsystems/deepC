# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for divitional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License") you may not use this file except in compliance
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

import os, argparse
from tensor_op_dict import *


def assignment(assignment):
	py_file = ""
	
	for key, vals in assignment.items():
		
		operator = operators["assignment_"+key]
		py_file += "\n\t# Assignment "+key.title()+"\n"
		
		for key_operand, value_operands in vals.items():
			
			for value_operand in value_operands:
				
				py_file += "\n\t# "+key_operand + " " + operator + " " + value_operand + "\n"
				py_file += "\tdef test_Assignment_" + key.title() + "_" + key_operand + "_" + value_operand + " (self):\n"
				
				py_file += "\t\ttemp_np = self.np_" + tensorOperands[key_operand] + ".copy()\n"
				if "tensor" in value_operand:
					py_file += "\t\ttemp_np "+ operator +" self.np_" + tensorOperands[value_operand] + "\n"
				elif "scalar" in value_operand:
					py_file += "\t\ttemp_np "+ operator + " " + tensorOperands[value_operand] + "\n"
				
				py_file += "\t\ttemp_dc = self.dc_" + tensorOperands[key_operand] + ".copy()\n"
				if "tensor" in value_operand:
					py_file += "\t\ttemp_dc "+ operator +" self.dc_" + tensorOperands[value_operand] + "\n"
				elif "scalar" in value_operand:
					py_file += "\t\ttemp_dc "+ operator + " " + tensorOperands[value_operand] + "\n"
				
				py_file += "\t\tnp.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))\n"
	
	return py_file

def binary(binary):
	py_file = ""
	
	for key, vals in binary.items():
		
		operator = operators["binary_"+key]
		py_file += "\n\t# Binary "+key.title()+"\n"
		
		for key_operand, value_operands in vals.items():
			
			for value_operand in value_operands:
				
				py_file += "\n\t# "+key_operand + " " + operator + " " + value_operand + "\n"
				py_file += "\tdef test_Binary_" + key.title() + "_" + key_operand + "_" + value_operand + " (self):\n"
				
				py_file += "\t\ttemp_np = self.np_" + tensorOperands[key_operand] + " "
				if "tensor" in value_operand:
					py_file += operator +" self.np_" + tensorOperands[value_operand] + "\n"
				elif "scalar" in value_operand:
					py_file += operator + " " + tensorOperands[value_operand] + "\n"
				
				py_file += "\t\ttemp_dc = self.dc_" + tensorOperands[key_operand] + " "
				if "tensor" in value_operand:
					py_file += operator +" self.dc_" + tensorOperands[value_operand] + "\n"
				elif "scalar" in value_operand:
					py_file += operator + " " + tensorOperands[value_operand] + "\n"
				
				py_file += "\t\tnp.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))\n"
	
	return py_file

def unary(unary):
	py_file = ""
	
	for key, value_operands in unary.items():
		
		operator = operators["unary_"+key]
		py_file += "\n\t# Unary "+key.title()+"\n"
		
		for value_operand in value_operands:

			py_file += "\n\t# " + operator + " " + value_operand + "\n"
			py_file += "\tdef test_Unary_" + key.title() + "_" + value_operand + " (self):\n"
			
			py_file += "\t\ttemp_np = " + operator + " self.np_" + tensorOperands[value_operand] + "\n"
			
			py_file += "\t\ttemp_dc = " + operator + " self.dc_" + tensorOperands[value_operand] + "\n"
			
			py_file += "\t\tnp.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))\n"
	
	return py_file

def comparison(comparison):
	py_file = ""
	
	for key, vals in comparison.items():
		
		operator = operators["comparison_"+key]
		py_file += "\n\t# Comparison "+key.title()+"\n"
		
		for key_operand, value_operands in vals.items():
			
			for value_operand in value_operands:
				
				py_file += "\n\t# "+key_operand + " " + operator + " " + value_operand + "\n"
				py_file += "\tdef test_Comparison_" + key.title() + "_" + key_operand + "_" + value_operand + " (self):\n"
				
				py_file += "\t\ttemp_np = self.np_" + tensorOperands[key_operand] + " "
				if "tensor" in value_operand:
					py_file += operator +" self.np_" + tensorOperands[value_operand] + "\n"
				elif "scalar" in value_operand:
					py_file += operator + " " + tensorOperands[value_operand] + "\n"
				
				py_file += "\t\ttemp_dc = self.dc_" + tensorOperands[key_operand] + " "
				if "tensor" in value_operand:
					py_file += operator +" self.dc_" + tensorOperands[value_operand] + "\n"
				elif "scalar" in value_operand:
					py_file += operator + " " + tensorOperands[value_operand] + "\n"
				
				py_file += "\t\tnp.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))\n"
	
	return py_file


def main():

	py_file = '''# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for divitional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License") you may not use this file except in compliance
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


# This file is auto generated by tensor_op_gen.py

import common

import dnnc as dc
import numpy as np
import unittest

class tensorOperatorsGeneratedTest(unittest.TestCase):

	def setUp(self):

		self.np_bool_0_4 = np.arange(5).astype(np.bool)
		self.np_bool_5_9 = np.arange(5,10).astype(np.bool)

		self.np_int_0_4 = np.arange(5).astype(np.int)
		self.np_int_5_9 = np.arange(5,10).astype(np.int)

		self.np_float_0_4 = np.arange(5).astype(np.float)
		self.np_float_5_9 = np.arange(5,10).astype(np.float)

		self.np_double_0_4 = np.arange(5).astype(np.double)
		self.np_double_5_9 = np.arange(5,10).astype(np.double)

		self.dc_bool_0_4 = dc.arange(5).asTypeBool()
		self.dc_bool_5_9 = dc.arange(5,10).asTypeBool()

		self.dc_int_0_4 = dc.arange(5).asTypeInt()
		self.dc_int_5_9 = dc.arange(5,10).asTypeInt()

		self.dc_float_0_4 = dc.arange(5).asTypeFloat()
		self.dc_float_5_9 = dc.arange(5,10).asTypeFloat()

		self.dc_double_0_4 = dc.arange(5).asTypeDouble()
		self.dc_double_5_9 = dc.arange(5,10).asTypeDouble()

'''
	
	parser = argparse.ArgumentParser(description="generate and run tensor operators' unittests")
	parser.add_argument("-a", "--assignment", action="store_true", help="add assignment tensor operators in unittest file")
	parser.add_argument("-u", "--unary", action="store_true", help="add unary tensor operators in unittest file")
	parser.add_argument("-b", "--binary", action="store_true", help="add binary tensor operators in unittest file")
	parser.add_argument("-c", "--comparison", action="store_true", help="add comaparison tensor operators in unittest file")
	parser.add_argument("-r", "--run", action="store_true", help="run the tensor unittest file after generation")
	parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity of unittests")
	args = parser.parse_args()
	
	if not args.assignment and not args.unary and not args.binary and not args.comparison:
		args.assignment = args.unary = args.binary = args.comparison = True
	
	if args.assignment:
		try:
			py_file += assignment(tensorOperators['assignment'])
		except:
			py_file += "\n\n\t# something went wrong while handling Assignment tensor operators.\n\n"
	
	if args.binary:
		try:
			py_file += binary(tensorOperators['binary'])
		except:
			py_file += "\n\n\t# something went wrong while handling Binary tensor operators.\n\n"
	
	if args.unary:
		try:
			py_file += unary(tensorOperators['unary'])
		except:
			py_file += "\n\n\t# something went wrong while handling Unary tensor operators.\n\n"
	
	if args.comparison:
		try:
			py_file += comparison(tensorOperators['comparison'])
		except:
			py_file += "\n\n\t# something went wrong while handling Comparison tensor operators.\n\n"
	
	py_file += '''

	def tearDown(self):
		return "test finished"

if __name__ == '__main__':
	
	unittest.main()
	'''

	with open ("tensorOperatorsGenerated.py", "w") as f:
		f.write(py_file)
	
	if args.run:
		if args.verbose:
			os.system("python3 tensorOperatorsGenerated.py -v")
		else:
			os.system("python3 tensorOperatorsGenerated.py")

	return


if __name__ == "__main__":
	main()