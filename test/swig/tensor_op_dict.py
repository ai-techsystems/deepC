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

'''
This file is required by tensorOperatorsGenerator.py
This is the structure of the dictionary

"operator":{
		"left operand" : [
			 		"right operand 1",
			 		"right operand 2",
			 		"right operand 3",
			 		"right operand 4",
			 	],
		}
'''

tensorOperators = {
	"assignment" : {
		"add" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"int_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				# "int_tensor_2",		# Dnnc doesn't support
			],
		},
		"sub" : {
			"bool_tensor_1" : [
				# "bool_scalar",		# Numpy doesn't support
				# "bool_tensor_2",		# Numpy doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"int_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				# "int_tensor_2",		# Dnnc doesn't support
			],
		},
		"mul" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"int_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				# "int_tensor_2",		# Dnnc doesn't support
			],
		},
		"true_div" : {
			"bool_tensor_1" : [
				# "bool_scalar",		# Numpy doesn't support
				# "bool_tensor_2",		# Numpy doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"int_tensor_1" : [
				# "bool_scalar",		# Numpy doesn't support
				# "bool_tensor_2",		# Numpy doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"float_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Dnnc != Numpy
				# "float_tensor_2",		# Dnnc != Numpy
				# "int_scalar",			# Dnnc != Numpy
				# "int_tensor_2",		# Dnnc doesn't support
			],
		},
		"floor_div" : {
			"bool_tensor_1" : [
				# "bool_scalar",		# Numpy doesn't support
				# "bool_tensor_2",		# Numpy doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"int_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				# "int_tensor_2",		# Dnnc doesn't support
			],
		},
		"pow" : {
			"bool_tensor_1" : [
				# "bool_scalar",		# Numpy doesn't support
				# "bool_tensor_2",		# Numpy doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"int_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				# "int_tensor_2",		# Dnnc doesn't support
			],
		},
		"and" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"int_tensor_1" : [
				# "bool_scalar",		# Dnnc != Numpy
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Dnnc != Numpy
				# "int_tensor_2",		# Dnnc != Numpy
			],
			"float_tensor_1" : [
				# "bool_scalar",		# Numpy doesn't support
				# "bool_tensor_2",		# Numpy doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
		},
		"or" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"int_tensor_1" : [
				# "bool_scalar",		# Dnnc != Numpy
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Dnnc != Numpy
				# "int_tensor_2",		# Dnnc != Numpy
			],
			"float_tensor_1" : [
				# "bool_scalar",		# Numpy doesn't support
				# "bool_tensor_2",		# Numpy doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
		},
		"xor" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
			"int_tensor_1" : [
				# "bool_scalar",		# Dnnc != Numpy
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Dnnc != Numpy
				# "int_tensor_2",		# Dnnc != Numpy
			],
			"float_tensor_1" : [
				# "bool_scalar",		# Numpy doesn't support
				# "bool_tensor_2",		# Numpy doesn't support
				# "float_scalar",		# Numpy doesn't support
				# "float_tensor_2",		# Numpy doesn't support
				# "int_scalar",			# Numpy doesn't support
				# "int_tensor_2",		# Numpy doesn't support
			],
		},
	},
	"binary" : {
		"add" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				# "float_scalar",		# Dnnc doesn't support
				# "float_tensor_2",		# Dnnc doesn't support
				# "int_scalar",			# Dnnc doesn't support
				# "int_tensor_2",		# Dnnc doesn't support
			],
			"int_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				# "float_scalar",		# Dnnc doesn't support
				# "float_tensor_2",		# Dnnc doesn't support
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				# "bool_tensor_2",		# Dnnc doesn't support
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				# "int_tensor_2",		# Dnnc doesn't support
			],
		},
		"sub" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"mul" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"true_div" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"floor_div" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"mod" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"pow" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"and" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"or" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
	},
	"unary" : {
		"pos" : [
			# "bool_tensor_1",			# Numpy doesn't support
			"int_tensor_1",
			"float_tensor_1",
		],
		"neg" : [
			# "bool_tensor_1",			# Numpy doesn't support
			"int_tensor_1",
			"float_tensor_1",
		],
	},
	"comparison" : {
		"greater_than" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"greater_equal" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"less_than" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"less_equal" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"equal" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
		"not_equal" : {
			"bool_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"int_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
			"float_tensor_1" : [
				"bool_scalar",
				"bool_tensor_2",
				"float_scalar",
				"float_tensor_2",
				"int_scalar",
				"int_tensor_2",
			],
		},
	},
}


# Don't change the section below, until you know what you are doing

operators = {
	"assignment_add" : "+=",
	"assignment_sub" : "-=",
	"assignment_mul" : "*=",
	"assignment_true_div" : "/=",
	"assignment_floor_div" : "//=",
	"assignment_mod" : "%=",
	"assignment_pow" : "**=",
	"assignment_lshift" : "<<=",
	"assignment_rshift" : ">>=",
	"assignment_and" : "&=",
	"assignment_or" : "|=",
	"assignment_xor" : "^=",
	"binary_add" : "+",
	"binary_sub" : "-",
	"binary_mul" : "*",
	"binary_true_div" : "/",
	"binary_floor_div" : "//",
	"binary_mod" : "%",
	"binary_pow" : "**",
	"binary_lshift" : "<<",
	"binary_rshift" : ">>",
	"binary_and" : "&",
	"binary_or" : "|",
	"binary_xor" : "^",
	"unary_pos" : "+",
	"unary_neg" : "-",
	"unary_invert" : "~",
	"comparison_greater_than" : ">",
	"comparison_greater_equal" : ">=",
	"comparison_less_than" : "<",
	"comparison_less_equal" : "<=",
	"comparison_equal" : "==",
	"comparison_not_equal" : "!=",
}

tensorOperands = {
	"bool_scalar" : "True",
	"bool_tensor_1" : "bool_0_4",
	"bool_tensor_2" : "bool_5_9",
	"int_scalar" : "5",
	"float_scalar" : "5.0",
	"int_tensor_2" : "int_5_9",
	"float_tensor_2" : "float_5_9",
	"int_tensor_1" : "int_0_4",
	"float_tensor_1" : "float_0_4",
}