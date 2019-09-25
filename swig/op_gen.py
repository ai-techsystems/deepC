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

import ast

''' we don't need it as of now

dtype_precedence_dict = {
	"double" : 16, 	# 8
	"float" : 14, 	# 4
	"long" : 10,	# 4
	"int" : 8, 		# 4
	"short" : 6, 	# 2
	"bool" : 4,		# 2
	"char" : 2,  	# 1 
}
'''

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


def get_swig_extern(s):
	s = "\textern "+s.split("{")[0].replace(">","> \\\n\t\t",1)+";\n"
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
	if s[s.find(".asType")-1:s.find(".asType")] == "a":
		s = s.replace("return op.compute(a, b);","return op.compute("+dtype+"_a, b);")
	elif s[s.find(".asType")-1:s.find(".asType")] == "b":
		s = s.replace("return op.compute(a, b);","return op.compute(a, "+dtype+"_b);")
	return s


def binary_operators(s):
	cpp_file = swig_extern_file = ""

	for content in s.split("\n\n"):
		if "<output>" not in content:

			if "dtype" in content:
				raise Exception("input output not mentioned, try again!")

			temp = content + "\n\n"
			temp = change_dtype(temp)
			cpp_file += temp.replace("\n","\n\t")
			temp = get_swig_extern(temp)
			swig_extern_file += temp
			continue

		if "dtype" not in content:
			raise Exception("dtype not mentioned, try again!")

		dtype = get_dtype_dictionary(content)
		content = remove_dtype(content)

		if "dtype" in content:
			raise Exception("dtype block could not be removed, try again!")

		for output, input_2d in dtype.items():
			for i, input_1d in enumerate(input_2d):

				temp_typecast = ") {\n"
				temp = content.replace("output",output) + "\n"
				for j, input in enumerate(input_1d):
					
					temp = temp.replace("input"+str(j+1),input)
					if (output != input):
						temp_typecast += change_dtype(output,(j+1))
				
				temp = temp.replace(") {\n",temp_typecast)
				if "asType" in temp:
					temp = change_compute(temp)
				cpp_file += temp.replace("\n","\n\t")
				temp = get_swig_extern(temp)
				swig_extern_file += temp

	return cpp_file, swig_extern_file


def normal_operators(s):
	cpp_file = swig_extern_file = ""
	
	for content in s.split("\n\n"):
		if "<output>" not in content and "<input>" not in content:

			if "dtype" in content:
				raise Exception("input output not mentioned, try again!")

			temp = content + "\n\n"
			cpp_file += temp.replace("\n","\n\t")
			temp = get_swig_extern(temp)
			swig_extern_file += temp
			continue

		if "dtype" not in content:
			raise Exception("dtype not mentioned, try again!")

		dtype = get_dtype_dictionary(content)
		content = remove_dtype(content)

		if "dtype" in content:
			raise Exception("dtype block could not be removed, try again!")

		for input, output in dtype.items():
			temp = content.replace("input",input) .replace("output",output) + "\n\n"
			cpp_file += temp.replace("\n","\n\t")
			temp = get_swig_extern(temp)
			swig_extern_file += temp

	return cpp_file, swig_extern_file


def main():
	try:
		with open ( "dnnc.api" , "r") as f:
			print("Reading dnnc.api")
			contents = f.read()
	except:
		print("'dnnc.api' not found !")
		return 1

	else:

		split_string = "\n<\\/>\n\n"
		split_position = contents.find(split_string,1)
		cpp_file = contents[:split_position] + "\nnamespace dnnc {\n\n\t"
		swig_extern_file = contents.split("#include")[0] + "namespace dnnc {\n"
		
		contents = remove_comments(contents)
		if check_comments(contents):
			return 1
		
		temp_cpp_file , temp_swig_extern_file = binary_operators(contents[split_position:].split(split_string)[1])
		cpp_file += temp_cpp_file
		swig_extern_file += temp_swig_extern_file

		temp_cpp_file , temp_swig_extern_file = normal_operators(contents[split_position:].split(split_string)[2])
		cpp_file += temp_cpp_file
		swig_extern_file += temp_swig_extern_file

		cpp_file += "\n}\n"
		swig_extern_file += "}\n"

		with open ("dnnc_api.cpp" ,"w") as f:
			print("Saving 'dnnc_api.cpp'")
			f.write(cpp_file)

		with open ("dnnc_swig_externs.h" ,"w") as f:
			print("Saving 'dnnc_swig_externs.h'")
			f.write(swig_extern_file)


if __name__=="__main__":
	main()
