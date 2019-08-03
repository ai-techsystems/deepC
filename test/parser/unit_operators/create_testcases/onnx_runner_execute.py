from onnx_runner_generator import generate_onnx_runner

operators = {}

operators['Sigmoid'] = {
'inputs':'[np.array([1, 2, 3]).astype(np.float32)]'
}

operators['BitShift'] = {
'inputs':'[np.random.randn((2,3,4)),np.random.randn((2,3,4))]'
}

for operator, operator_info in operators.items():
	generate_onnx_runner(operator, operator_info['inputs'])
