# Developer's getting started guide

### Our work is divided in 3 parts: 

---
##### 1) Implementing these **[ONNX operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md)**.  

* We are working on **[dnnc-operators](https://github.com/ai-techsystems/dnnc-operators)** repo in C++. Our development status is mentioned **[here](https://github.com/ai-techsystems/dnnc-operators/blob/master/README.md)**

* You can study [Matmul.h](https://github.com/ai-techsystems/dnnc-operators/blob/master/include/operators/MatMul.h), [Add.h](https://github.com/ai-techsystems/dnnc-operators/blob/master/include/operators/Add.h) and [Threshholdrelu.h](https://github.com/ai-techsystems/dnnc-operators/blob/master/include/operators/ThresholdedRelu.h) to get a demo of what we are trying to achieve.

* Here is a **[Tutorial](https://dritchie.github.io/csci2240/assignments/eigen_tutorial.pdf)** for eigen library.  

* Here is more **[indepth discussion of philosophy and features of Eigen](http://downloads.tuxfamily.org/eigen/eigen_CGLibs_Giugno_Pisa_2013.pdf)**

* Also check out **[Eigen documentation](http://eigen.tuxfamily.org/dox/.)**
---

##### 2) Test the above **[ONNX operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md)**.

* Here are the **[test cases](https://github.com/ai-techsystems/dnnc-operators/tree/master/src/operators)**

* After making this branch stable it will be merged in the official repo **[dnnCompiler](https://github.com/ai-techsystems/dnnCompiler/tree/operators)**.

---
##### 3) Implement SWIG for python interface.

* Here is a **[SWIG Tutorial](http://www.swig.org/tutorial.html)**.

* DNNC [operators](https://github.com/ai-techsystems/dnnc-operators/tree/master/include/operators) and tensors should be implemented for the python interface with **[SWIG](http://www.swig.org/exec.html)**.

* Check out **[Numpy](https://docs.scipy.org/doc/numpy/reference/)** for the implementation of our tensor and it's simplicity.
