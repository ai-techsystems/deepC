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

def check_comments(s):
	if "/*" in s:
		print("\nUnmatched '/*' comment syntax at:\n\n"+s[s.find("*/")-100:s.find("*/")+100])
		return 1
	if "*/" in s:
		print("\nUnmatched '*/' comment syntax at:\n\n"+s[s.find("*/")-100:s.find("*/")+100])
		return 1
	return 0

def remove_comments(s):
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
	s = "extern "+s.split("{")[0].replace(">","> \\\n\t\t",1).replace("tensor","dnnc::tensor")+";\n"
	return s

def main():
	try:
		with open ( "dnnc.api" , "r") as f:
			print("Reading dnnc.api")
			contents = f.read()
	except:
		print("'dnnc.api' not found !")
		return
	
	else:
		split_position = contents.find("/*##")
		cpp_file = contents[:split_position]
		swig_extern_file = contents.split("#include")[0]
		contents = remove_comments(contents)
		
		if check_comments(contents):
			return

		for content in contents[split_position:].split("\n\n"):

			if "<output>" not in content and "<input>" not in content:
				
				if "dtype" in content:
					print("input output not mentioned, try again!")
					return
				
				temp = content + "\n\n"
				cpp_file += temp
				temp = get_swig_extern(temp)
				swig_extern_file += temp
				continue

			if "dtype" not in content:
				print("dtype not mentioned, try again!")
				return

			dtype = get_dtype_dictionary(content)
			content = remove_dtype(content)

			if "dtype" in content:
				print("dtype block could not be removed, try again!")
				return

			for input, output in dtype.items():
				temp = content.replace("input",input) .replace("output",output) + "\n\n"
				cpp_file += temp
				temp = get_swig_extern(temp)
				swig_extern_file += temp

		with open ("dnnc_api.cpp" ,"w") as f:
			print("Saving 'dnnc_api.cpp'")
			f.write(cpp_file)
		
		with open ("dnnc_swig_externs.h" ,"w") as f:
			print("Saving 'dnnc_swig_externs.h'")
			f.write(swig_extern_file)

if __name__=="__main__":
	main()