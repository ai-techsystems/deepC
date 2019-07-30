from generate_output import *

operators = {}

operators['Sigmoid'] = {
'nodes':['helper.make_node("Sigmoid", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

for operator in operators.keys():
	operator_info = operators[operator]
	create_testcase(operator, operator_info['inputs'], operator_info['outputs'], operator_info['nodes'])
