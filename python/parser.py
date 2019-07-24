import onnx
import sys

def get_node_symbol(node):
	symbol = ""
	symbol += node.op_type + '\n'
	symbol += '\t"' + node.name + '"\n'
	
	symbol += '\t[ '
	for elem in node.input:
		symbol += '[\"' + elem + '\"] '
	symbol += ']\n'

	symbol += '\t[ '
	for elem in node.output:
		symbol += '[\"' + elem + '\"] '
	symbol += ']\n'

	for attr in node.attribute:
		symbol += '\tAttr\n'
		symbol += '\t\t"' + node.name + '/' + attr.name + '"\n'
		symbol += '\t\t[ '

		if attr.type == onnx.AttributeProto.INT:
			symbol += str(attr.i)
		elif attr.type == onnx.AttributeProto.INTS:
			symbol += '['
			for index, num in enumerate(attr.ints):
				symbol += str(num)
				if index != len(attr.ints) - 1:
					symbol += ', '
			symbol += ']'
		elif attr.type == onnx.AttributeProto.FLOAT:
			symbol += str(attr.f)
		elif attr.type == onnx.AttributeProto.FLOATS:
			symbol += '['
			for index, num in enumerate(attr.floats):
				symbol += str(num)
				if index != len(attr.floats) - 1:
					symbol += ', '
			symbol += ']'
		symbol += ' ]\n'

	symbol += '\n'
	return symbol

def get_io_symbol(io, is_input):
	symbol = ""

	if is_input:
		symbol += "Input"
	else:
		symbol += "Output"
	symbol += '\n'
	symbol += '\t"' + io.name + '"\n'

	symbol += '\t[ '
	dims = io.type.tensor_type.shape.dim
	for i, dim in enumerate(dims):
		if dim.dim_param != "":
			symbol += dim.dim_param
		else:
			symbol += str(dim.dim_value)
		if i != len(dims) - 1:
			symbol += ', '
	symbol += ' ]\n'

	symbol += '\n'
	return symbol

def get_initializer_symbol(initializer):
	symbol = "Initializer\n"
	symbol += '\t"' + initializer.name + '"\n'

	symbol += '\t[ '
	for i, dim in enumerate(initializer.dims):
		symbol += str(dim)
		if i != len(initializer.dims) - 1:
			symbol += ', '
	symbol += ' ]\n'

	symbol += '\t[ ' + str(initializer.raw_data) + ' ]\n'

	symbol += '\n'
	return symbol

def parse(onnx_filename, output_file, onnx_output_file=None):
	model = onnx.load(onnx_filename)

	graph = model.graph
	nodes = graph.node
	
	print("writing to", output_file)
	f = open(output_file, "w")
	for node in nodes:
		f.write(get_node_symbol(node))

	for i in graph.input:
		f.write(get_io_symbol(i, True))

	for o in graph.output:
		f.write(get_io_symbol(o, False))

	for initl in graph.initializer:
		f.write(get_initializer_symbol(initl))

	f.close()


	if onnx_output_file is not None:
		with open(onnx_output_file, "w") as f:
			f.write(str(model))

if __name__ == "__main__":
	if len(sys.argv) >= 4:
		parse(sys.argv[1], sys.argv[2], sys.argv[3])
	else:
		parse(sys.argv[1], sys.argv[2])