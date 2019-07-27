import pytest
import new_parser
import os

def test_parser():
	testcase_dir = "./pytorch-to-onnx/testcases/"
	gold_dir = "./pytorch-to-onnx/gold_files/"

	testcases = []
	for r, d, f in os.walk(testcase_dir):
		for filename in f:
			path = r + '/' + filename		
			if ".onnx" in filename:
				testcases.append(path)

	errors = []
	for onnx_testcase in testcases:
		name = onnx_testcase.split('/')[-1][:-5]
		gold_filename = gold_dir + name + ".sym" + ".gold"
		new_st = new_parser.get_symbol_table(onnx_testcase)
		with open(gold_filename, "r") as f:
			gold_st = f.read()

		if new_st != gold_st:
			errors.append(name)
	
	assert errors == []