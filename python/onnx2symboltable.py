from onnx import proto3_pb2 as pb
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
		symbol += 'Attr\n'
		symbol += '\t"' + node.name + '/' + attr.name + '"\n'
		symbol += '\t[ '

		if attr.type == pb.AttributeProto.INT:
			symbol += str(attr.i)
		elif attr.type == pb.AttributeProto.INTS:
			symbol += '['
			for index, num in enumerate(attr.ints):
				symbol += str(num)
				if index != len(attr.ints) - 1:
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


# currently discarding elem_type information on node attributes, input/output, and initializers
# in initializers, not totally sure about how to handle dim_param if it is not unknown, so currently just printing it (if they are all unknown, then can mark with -1)
# not sure whether metadata and version info should be included as symbols

model = pb.ModelProto()
with open("model.onnx", "rb") as f:
	model.ParseFromString(f.read())
	# print(model)

graph = model.graph
nodes = graph.node


f = open("symboltable.txt", "w")
for node in nodes:
	f.write(get_node_symbol(node))

for i in graph.input:
	f.write(get_io_symbol(i, True))

for o in graph.output:
	f.write(get_io_symbol(o, False))

for initl in graph.initializer:
	f.write(get_initializer_symbol(initl))

f.close()