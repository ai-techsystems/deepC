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


from tensor_interface_helper import *
import ast


def check_comments(s):
	if "*/" in s:
		print("\nUnmatched '*/' comment syntax at:\n\n"+s[s.find("*/")-100:s.find("*/")+100])
		return 1
	if "/*" in s:
		print("\nUnmatched '/*' comment syntax at:\n\n"+s[s.find("/*")-100:s.find("/*")+100])
		return 1
	return 0


def remove_comments(s):
	# going with string replace
	for i in range(s.count("/*")):
		comment_block = s[s.find("/*"):s.find("*/\n")+4]   # +4 is to remove "\n" after "*/"
		s = s.replace(comment_block,"")
	return s


def get_dtype_dictionary(s):
	string_dict = s.replace(" ","").split("dtype=")[1].split("}")[0].replace("\n","").replace("\t","")+"}"
	dtype = ast.literal_eval(string_dict)
	return dtype


def remove_dtype(s):
	dtype_string = ""

	if "\tdtype" in s:
		dtype_string = s[s.find("\tdtype"):s.find("}",s.find("dtype"))+2]
	elif "  dtype" in s:
		dtype_string = s[s.find("  dtype"):s.find("}",s.find("dtype"))+2]
	else:
		dtype_string = s[s.find("dtype"):s.find("}",s.find("dtype"))+2]

	s = s.replace("} ","}").replace(dtype_string,"")
	return s


def get_swig_extern(dc_operator, s):
	s = "\textern "+s.split("{")[0].replace(dc_operator,"\\\n\t\t"+dc_operator,1)+";\n"
	return s


def change_dtype(output,i):
	s = ""
	if i == 1:
		s += "\ttensor<"+output+"> "+output+"_a = a.asType<"+output+">();\n"
	elif i == 2:
		s += "\ttensor<"+output+"> "+output+"_b = b.asType<"+output+">();\n"
	return s


def change_compute(s):
	dtype = s.split(".asType<")[1].split(">")[0]
	if s.count("asType<") == 1:
		if s[s.find(".asType")-1:s.find(".asType")] == "a":
			s = s.replace("return op.compute(a, b);","return op.compute("+dtype+"_a, b);")
		elif s[s.find(".asType")-1:s.find(".asType")] == "b":
			s = s.replace("return op.compute(a, b);","return op.compute(a, "+dtype+"_b);")
	elif s.count("asType<") == 2:
		s = s.replace("return op.compute(a, b);","return op.compute("+dtype+"_a, "+dtype+"_b);")
	else:
		raise Exception("asType() count is wrong, try again!")
	return s


def overload_python_operator(dc_operator, operator_python):
	s = '''
	def __<operand>__(self, other):
		return dc.<operator>(self, other)

	def __r<operand>__(self, other):
		return dc.<operator>(other, self)
'''
	s = s.replace("<operator>",dc_operator).replace("<operand>",operator_python)
	return s


def get_scalar(dc_operator, i):
	s = ""
	if i == 1:
		s = '''tensor<output> dc_operator(tensor<input1> &a, input2 b) {
	tensor<input2> tensor_b(1);
	tensor_b.load(&b);
	return dc_operator(a, tensor_b);
}
'''

	if i == 2:
		s = '''tensor<output> dc_operator(input1 a, tensor<input2> &b) {
	tensor<input1> tensor_a(1);
	tensor_a.load(&a);
	return dc_operator(tensor_a, b);
}
'''

	if i == 3:
		s = '''output dc_operator(input1 a, input2 b) {
	tensor<input1> tensor_a(1);
	tensor<input2> tensor_b(1);
	tensor_a.load(&a);
	tensor_b.load(&b);
	return dc_operator(tensor_a, tensor_b)[0];
}
'''
	s = s.replace("dc_operator", dc_operator)
	return s


def binary_operators(s):
	cpp_file = swig_extern_file = tensor_swig_helper_file = py_file = ""

	operator_list = ast.literal_eval(s.split("\n\n")[0].split("operator_list = ")[1])

	dtype = ast.literal_eval(s.split("\n\n")[1].split("dtype = ")[1])

	temp_content = s.split("\n\n")[2]
	for dc_operator, dc_operator_values in operator_list.items():
		
		for i in range (4):
			if i==0:

				content = temp_content[:]

				operator_header, operator_python = dc_operator_values
				content = content.replace("dc_operator", dc_operator).replace("operator_header", operator_header)

				if "dtype" in content:
					raise Exception("dtype block could not be removed, try again!")

				# tensor interface for true_div and floor_div are written manually
				if (dc_operator != "true_div") and (dc_operator != "floor_div"):
					tensor_swig_helper_file += tensor_swig_helper_binary(dc_operator, operator_header, operator_python)
				py_file += overload_python_operator(dc_operator, operator_python)

				for output, input_2d in dtype.items():
					
					# true_div only outputs in float
					if (dc_operator == "true_div"):
						output = "double" 
					
					# floor_div only outputs in int
					if (dc_operator == "floor_div"):
						output = "int" 
					
					for input_1d in input_2d:

						input1, input2 = input_1d
						temp_typecast = ") {\n"

						if (input1 != output):
							temp_typecast += change_dtype(output,1)
						if (input2 != output):
							temp_typecast += change_dtype(output,2)

						temp = content.replace("input1",input1).replace("input2",input2).replace("input",output).replace("output",output) + "\n\n"
						temp = temp.replace(") {\n",temp_typecast)
						
						if "asType" in temp:
							temp = change_compute(temp)
						cpp_file += temp.replace("\n","\n\t")
						temp = get_swig_extern(dc_operator, temp)
						swig_extern_file += temp

			if i>0 and i<4:
				content = get_scalar(dc_operator, i)

				for output, input_2d in dtype.items():
					# true_div only outputs in float
					if (dc_operator == "true_div"):
						output = "double" 
					# floor_div only outputs in int
					if (dc_operator == "floor_div"):
						output = "int" 
					
					for input_1d in input_2d:
						input1, input2 = input_1d
						temp = content.replace("input1",input1).replace("input2",input2).replace("output",output) + "\n"
						cpp_file += temp.replace("\n","\n\t")
						temp = get_swig_extern(dc_operator, temp)
						swig_extern_file += temp

	return cpp_file, swig_extern_file, tensor_swig_helper_file, py_file


def logical_operators(s):
	cpp_file = swig_extern_file = tensor_swig_helper_file = py_file = ""

	operator_list = ast.literal_eval(s.split("\n\n")[0].split("operator_list = ")[1])

	dtype = ast.literal_eval(s.split("\n\n")[1].split("dtype = ")[1])

	temp_content = s.split("\n\n")[2]
	for dc_operator, dc_operator_values in operator_list['logical'].items():
		
		for i in range (4):
			if i==0:

				content = temp_content[:]
				
				operator_header, operator_python = dc_operator_values
				content = content.replace("dc_operator", dc_operator).replace("operator_header", operator_header)

				if "dtype" in content:
					raise Exception("dtype block could not be removed, try again!")

				tensor_swig_helper_file += tensor_swig_helper_logical(dc_operator, operator_header, operator_python)
				py_file += overload_python_operator(dc_operator, operator_python)

				for output, input_2d in dtype.items():
					for input_1d in input_2d:
						input1, input2 = input_1d
						temp_typecast = ") {\n"
						
						if (input1 != output):
							temp_typecast += change_dtype(output,1)
						if (input2 != output):
							temp_typecast += change_dtype(output,2)

						temp = content.replace("input1",input1).replace("input2",input2).replace("input",output).replace("output",output) + "\n\n"
						temp = temp.replace(") {\n",temp_typecast)

						if "asType" in temp:
							temp = change_compute(temp)
						cpp_file += temp.replace("\n","\n\t")
						temp = get_swig_extern(dc_operator, temp)
						swig_extern_file += temp

			
			if i>0 and i<4:
				content = get_scalar(dc_operator, i)
				for output, input_2d in dtype.items():

					for input_1d in input_2d:
						input1, input2 = input_1d
						temp = content.replace("input1",input1).replace("input2",input2).replace("output",output) + "\n"
						cpp_file += temp.replace("\n","\n\t")
						temp = get_swig_extern(dc_operator, temp)
						swig_extern_file += temp

	return cpp_file, swig_extern_file, tensor_swig_helper_file, py_file



def comparison_operators(s, dtype_precedence_dict):
	cpp_file = swig_extern_file = tensor_swig_helper_file = py_file = ""

	operator_list = ast.literal_eval(s.split("\n\n")[0].split("operator_list = ")[1])
	dtype = ast.literal_eval(s.split("\n\n")[1].split("dtype = ")[1])

	temp_content = s.split("\n\n")[2]
	for dc_operator, dc_operator_values in operator_list['comparison'].items():
		
		for i in range (4):
			if i==0:

				content = temp_content[:]
				
				operator_header, operator_python = dc_operator_values
				content = content.replace("dc_operator", dc_operator).replace("operator_header", operator_header)

				if "dtype" in content:
					raise Exception("dtype block could not be removed, try again!")

				tensor_swig_helper_file += tensor_swig_helper_comparison(dc_operator, operator_header, operator_python)
				py_file += overload_python_operator(dc_operator, operator_python)

				for output, input_2d in dtype.items():
					for input_1d in input_2d:
						input1, input2 = input_1d
						temp_typecast = ") {\n"
						
						input = ""
						if (input1 != input2):
							if (dtype_precedence_dict[input1] > dtype_precedence_dict[input2]):
								input = input1
								temp_typecast += change_dtype(input,2)
							elif (dtype_precedence_dict[input1] < dtype_precedence_dict[input2]):
								input = input2
								temp_typecast += change_dtype(input,1)
							else:
								raise Exception("different datatypes can't have same precedence, try again!")
						else:
							input = input1

						temp = content.replace("input1",input1).replace("input2",input2).replace("input",input).replace("output",output) + "\n\n"
						temp = temp.replace(") {\n",temp_typecast)

						if "asType" in temp:
							temp = change_compute(temp)
						cpp_file += temp.replace("\n","\n\t")
						temp = get_swig_extern(dc_operator, temp)
						swig_extern_file += temp


			if i>0 and i<4:
				content = get_scalar(dc_operator, i)
				for output, input_2d in dtype.items():

					for input_1d in input_2d:
						input1, input2 = input_1d
						temp = content.replace("input1",input1).replace("input2",input2).replace("output",output) + "\n"
						cpp_file += temp.replace("\n","\n\t")
						temp = get_swig_extern(dc_operator, temp)
						swig_extern_file += temp

	return cpp_file, swig_extern_file, tensor_swig_helper_file, py_file


def normal_operators(s):
	cpp_file = swig_extern_file = ""
	
	for content in s.split("\n\n"):
		dc_operator = content.split("> ")[1].split("(")[0]
		if "<output>" not in content and "<input>" not in content:

			if "dtype" in content:
				raise Exception("input output not mentioned, try again!")

			temp = content + "\n\n"
			cpp_file += temp.replace("\n","\n\t")
			temp = get_swig_extern(dc_operator, temp)
			swig_extern_file += temp
			continue

		if "dtype" not in content:
			raise Exception("dtype not mentioned, try again!")

		dtype = get_dtype_dictionary(content)
		content = remove_dtype(content)

		if "dtype" in content:
			raise Exception("dtype block could not be removed, try again!")

		for output, input in dtype.items():
			temp = ""
			if type(input) is tuple:
				for input_items in input:
					if type(input_items) is list:
						temp = content.replace("output",output) + "\n\n"
						for i, input_item in enumerate(input_items):
							temp = temp.replace("input"+str(i+1), input_item)
					elif type(input_items) is str:
						temp = content.replace("input",input_items).replace("output",output) + "\n\n"

					cpp_file += temp.replace("\n","\n\t")
					temp = get_swig_extern(dc_operator, temp)
					swig_extern_file += temp
			else:
				if type(input) is list:
					temp = content.replace("output",output) + "\n\n"
					for i, input_item in enumerate(input):
						temp = temp.replace("input"+str(i+1), input_item)
				elif type(input) is str:
					temp = content.replace("input",input).replace("output",output) + "\n\n"

				cpp_file += temp.replace("\n","\n\t")
				temp = get_swig_extern(dc_operator, temp)
				swig_extern_file += temp

	return cpp_file, swig_extern_file


def generate_py_file(s):
	s += '''
import dnnc as dc

class mydnnc(dc):
'''
	return s


def main():
	try:
		with open ( "dnnc.api" , "r") as f:
			print("Reading 'dnnc.api'")
			contents = f.read()
	except:
		print("'dnnc.api' not found !")
		return 1

	else:

		split_string = "\n<\\/>\n\n"
		split_position = contents.find(split_string,1)
		cpp_file = contents[:split_position] + "\nnamespace dnnc {\n\n\t"
		swig_extern_file = contents.split("#include")[0] + "namespace dnnc {\n"
		py_file = generate_py_file(contents.split("#include")[0])
		tensor_swig_helper_file = ""
		
		contents = remove_comments(contents)
		if check_comments(contents):
			return 1

		dtype_precedence_dict = ast.literal_eval(contents[split_position:].split(split_string)[1].split("dtype_precedence_dict = ")[1])

		temp_cpp_file, temp_swig_extern_file, temp_tensor_swig_helper_file, temp_py_file = binary_operators(contents[split_position:].split(split_string)[2][:-1])
		cpp_file += temp_cpp_file
		swig_extern_file += temp_swig_extern_file
		tensor_swig_helper_file += temp_tensor_swig_helper_file
		py_file += temp_py_file

		temp_cpp_file, temp_swig_extern_file, temp_tensor_swig_helper_file, temp_py_file = logical_operators(contents[split_position:].split(split_string)[3][:-1])
		cpp_file += temp_cpp_file
		swig_extern_file += temp_swig_extern_file
		tensor_swig_helper_file += temp_tensor_swig_helper_file
		py_file += temp_py_file

		temp_cpp_file, temp_swig_extern_file, temp_tensor_swig_helper_file, temp_py_file = comparison_operators(contents[split_position:].split(split_string)[3][:-1], dtype_precedence_dict)
		cpp_file += temp_cpp_file
		swig_extern_file += temp_swig_extern_file
		tensor_swig_helper_file += temp_tensor_swig_helper_file
		py_file += temp_py_file

		temp_cpp_file, temp_swig_extern_file = normal_operators(contents[split_position:].split(split_string)[4])
		cpp_file += temp_cpp_file
		swig_extern_file += temp_swig_extern_file

		cpp_file += "\n}\n"
		swig_extern_file += "}\n"
		py_file = "# " + py_file.replace("\n//", "\n#")[3:]

		with open ("dnnc_api.cpp" ,"w") as f:
			print("Saving 'dnnc_api.cpp'")
			f.write(cpp_file)

		with open ("dnnc_swig_externs.h" ,"w") as f:
			print("Saving 'dnnc_swig_externs.h'")
			f.write(swig_extern_file)

		# with open ("mydnnc.py" ,"w") as f:
		# 	print("Saving 'mydnnc.py'")
		# 	f.write(py_file)

		try:
			with open ("tensor.i", "r") as f:
				s = f.read()
		except:
			print("'tensor.i' not found !")
			return 1
		else:
			comment = "// <\\/>"

			# Uncomment the below line to stop adding these operators in 'tensor.i'
			# tensor_swig_helper_file = "\n\n\n"
			
			try:
				s = s.split(comment)[0] + comment + tensor_swig_helper_file + comment + s.split(comment)[2]
			except:
				print("'"+comment+"' not found 'tensor.i'!")
				return 1
			else:
				with open ("tensor.i" ,"w") as f:
					print("Saving 'tensor.i'")
					f.write(s)


if __name__=="__main__":
	main()
