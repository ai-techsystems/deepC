import onnx
import caffe2.python.onnx.backend as backend
import numpy as np
import re

model_file = "asl_model.onnx"
input_file = "gesture.data"

gesture = None;
with open(input_file, "r") as fp:
    lineList = fp.read().replace('\n', ',')[:-1]
    ary = [float(num) for num in lineList.split(',')]
    gesture  = np.array(ary, dtype=np.float32)

# Load the ONNX model
model = onnx.load(model_file)

# Check that the IR is well formed
onnx.checker.check_model(model)

rep = backend.prepare(model, device="CPU")

outputs = rep.run(gesture)

print(('hi', 'sup')[np.argmax(outputs[0])])


