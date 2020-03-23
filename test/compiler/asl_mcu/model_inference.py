import onnx
import caffe2.python.onnx.backend as backend
import numpy as np

def infer_gesture(input_file):
    model_file = "asl_model.onnx"

    gesture = None;
    with open(input_file, "r") as fp:
        lineList = fp.read().replace('\n', ',')[:-1]
        ary = [float(num) for num in lineList.split(',')]
        # normalize
        for i in range(0, 714, 6):
            ary[i+0] = (ary[i+0] + 4.)/8.
            ary[i+1] = (ary[i+1] + 4.)/8.
            ary[i+2] = (ary[i+2] + 4.)/8.
            ary[i+3] = (ary[i+3] + 2000.)/4000.
            ary[i+4] = (ary[i+4] + 2000.)/4000.
            ary[i+5] = (ary[i+5] + 2000.)/4000.
        gesture  = np.array(ary, dtype=np.float32)


    # Load the ONNX model
    model = onnx.load(model_file)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    rep = backend.prepare(model, device="CPU")

    outputs = rep.run(gesture)

    print(input_file + ": " + ('hi', 'sup')[np.argmax(outputs[0])], end='')
    print("\t", outputs[0], end='')
    print("")


#"./gesture.data",
input_files = [
        "gestures/hi00.csv",
        "gestures/hi01.csv",
        "gestures/hi02.csv",
        "gestures/hi03.csv",
        "gestures/hi04.csv",
        "gestures/hi05.csv",
        "gestures/sup00.csv",
        "gestures/sup01.csv",
        "gestures/sup02.csv",
        "gestures/sup03.csv",
        "gestures/sup04.csv",
        "gestures/sup05.csv"
        ]
for input_file in input_files:
    infer_gesture(input_file)
