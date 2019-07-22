import os
from parser import *

testcase_dir = "./testcases/"
output_dir = "./output/"

testcases = []
for r, d, f in os.walk(testcase_dir):
	for filename in f:
		if ".onnx" in filename:
			testcases.append(testcase_dir + filename)

for onnx_testcase in testcases:
	name = onnx_testcase.split('/')[-1]
	print(name)
	output_filename = output_dir + name + ".sym" + ".gold"
	parse(onnx_testcase, output_filename)
