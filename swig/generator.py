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


def main():
	try:
		with open ( "dnnc.api" , "r") as f:
			print("Reading dnnc.api")
			contents = f.read()
	except:
		print("'dnnc.api' not found !")
		return
	
	else:
		contents = contents[:contents.rfind("}")]
		cpp_file = contents.split("using namespace dnnc;\n\n")[0]+"using namespace dnnc;\n\n"
		swig_extern = contents.split("#include")[0]

		for content in contents.split("using namespace dnnc;\n\ntensor")[1].split("}\n\ntensor"):

			# if float is the dtype, then it also supports int (most of the cases)
			if "<float>" in content:
				for dtype in ['<float>', '<int>']:
					temp = ("tensor"+content.replace("<float>",dtype)+"}\n\n")
					cpp_file += temp
					temp = ("extern "+temp.split("{")[0].replace(">","> \\\n\t\t",1).replace("tensor","dnnc::tensor")+";\n")
					swig_extern += temp

			# if int or bool is the dtype, it won't support any other dtypes (most of the cases)
			else:
				temp= ("tensor"+content+"}\n\n")
				cpp_file += temp
				temp = ("extern "+temp.split("{")[0].replace(">","> \\\n\t\t",1).replace("tensor","dnnc::tensor")+";\n")
				swig_extern += temp


		with open ("dnnc_api.cpp" ,"w") as f:
			print("Saving 'dnnc_api.cpp'")
			f.write(cpp_file)
		
		with open ("dnnc_swig_externs.h" ,"w") as f:
			print("Saving 'dnnc_swig_externs.h'")
			f.write(swig_extern)

		return


if __name__=="__main__":
	
	consent = input("Are you sure you want to overwrite the current 'dnnc_api.cpp'? [y/N] : " )
	
	if consent == "y" or consent == "Y":
		main()
	else:
		print("Closing program!")