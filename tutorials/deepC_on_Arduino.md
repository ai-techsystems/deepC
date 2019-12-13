# deepC on ![Arduino](https://www.arduino.cc/en/uploads/Trademark/ARDUINO_logo.jpg)

## Bring your own Model (BYOM)
1. Train your machine learnig model.
1. [Convert it to onnx](https://github.com/onnx/tutorials#converting-to-onnx-format)

## deepC Compile to C++
Download and install deepC and run ```onnx2cpp``` command

```% onnx2cpp <model>.onnx```

## Embed deepC model

Use setup and loop to include C++ file compiled by deepC
1. [setup()](https://www.arduino.cc/reference/en/language/structure/sketch/setup/)
1. [loop()](https://www.arduino.cc/reference/en/language/structure/sketch/loop/)
1. [Build on Arduino](https://github.com/arduino/Arduino/wiki/Build-Process)

Consult [Arduino Reference](https://www.arduino.cc/reference/en/) to complete your [application/sketch](https://www.arduino.cc/en/tutorial/sketch).

## Run
1. Upload the sketch: Sketch -> Upload
1. Open the Serial Monitor: Tools -> Serial Monitor

![Run on Arduino](https://cdn.sparkfun.com/assets/d/4/e/3/c/51df245ece395f0728000001.png)

### KPNS
1. [Replace Eigen lib of deepC with Arduino port](https://github.com/bolderflight/Eigen), in case you run into Eigen errors.

### Reference
1. [Getting Started with Arduino products](https://www.arduino.cc/en/Guide/HomePage)
