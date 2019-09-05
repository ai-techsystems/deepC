# Automating the Test cases and Swig interface

## Test Case Automation:

##### We have created 2 files which will keep track of our operators, which passes or fails the test cases:
* **[test / swig / passingTests.txt](../test/swig/passingTests.txt)**
* **[test / swig / failingTests.txt](../test/swig/failingTests.txt)**

##### We have created 2 python scripts to run the tests at ease:
* **[test / run_all.py](../test/run_all.py)** (It will run all the testcases mentioned on the `passingTests.txt`)
* **[test / run_one.py](../test/run_one.py)** (It will run only one testcase opearator at a time)

##### Why do we need them?
In a distant future in dnnCompiler development, we will come at a point, when pull request can only be done when the make command builds successfully. Currently in top level make, the `run_all.py` is already implemented. You can check that with command

```console
make TEST
```
This will help us to get rid of the tension when it comes to merging a update, whether the update will break the functionality or not.

---
## Operator Interface Automation:

We are currently automating the `dnnc.i` and `dnnc_api.cpp` file, to save you some time, and repeatative works.
In the process of automation we will be needing two files, 

* **[swig / dnnc.api](../swig/dnnc.api)** (pseudo cpp/python file which you will be adding your opearators in)
* **[swig / op_gen.py](../swig/op_gen.py)** (which will generate `dnnc_swig_externs.h` and `dnnc_api.cpp` file from the above `dnnc.api` file)

**op_gen.py is integrated in Makefile, so running make at the top-level or in swig/ dir will generate required files.**

## Explicit Usage :
```console
python op_gen.py
```

### I have tried to pick and write some diverse examples below to give you an idea how the `dnnc.api` file will look like.

---
##### MatMul and Add operators has input and output of same dtypes
```python

tensor<output> matmul(tensor<input> &a, tensor<input> &b) {
  MatMul<input> op;
  return op.compute(a, b);
  dtype = {
    "float" : "float",
    "int" : "int"
  }
}

tensor<output> add(tensor<input> &a, tensor<input> &b) {
  Add<input> op;
  return op.compute(a, b);
  dtype = {
    "float" : "float",
    "int" : "int"
  }
}
```
---
##### DequantizeLinear takes b tensor as float, and it's fixed, so declared the b tensor as `<float>`, instead of `<input>`

```python
tensor<output> dequantize_linear(tensor<input> &a, tensor<float> &b, tensor<input> &c) {
  DequantizeLinear<input> op;
  return op.compute(a, b, c);
  dtype = {
    "int" : "float"
  }
}
```
---
##### Elu has fixed input and output, `<float>` only, either you can write `<float>` instead of `<input>` and `<output>`, or specify dtype, both works.

```python
tensor<output> elu(tensor<input> &a, float alpha = 1.0) {
  Elu<input> op("localOpName", alpha);
  return op.compute(a);
  dtype = {
    "float" : "float"
  }
}
```
---
##### Equal only outputs in `<bool>`

```python
tensor<output> equal(tensor<input> &a, tensor<input> &b) {
  Equal<input> op;
  return op.compute(a, b);
  dtype = {
    "float" : "bool",
    "int" : "bool",
    "bool" : "bool"
  }
}
```
---
##### This should give you a rough idea how the dnnc.api file will look like. If you like to see the whole picture, see below

<details>
<summary>Example</summary>

```python


tensor<output> matmul(tensor<input> &a, tensor<input> &b) {
  MatMul<input> op;
  return op.compute(a, b);
  dtype = {
    "float" : "float",
    "int" : "int"
  }
}

tensor<output> add(tensor<input> &a, tensor<input> &b) {
  Add<input> op;
  return op.compute(a, b);
  dtype = {
    "float" : "float",
    "int" : "int"
  }
}

tensor<output> dequantize_linear(tensor<input> &a, tensor<float> &b, tensor<input> &c) {
  DequantizeLinear<input> op;
  return op.compute(a, b, c);
  dtype = {
    "int" : "float"
  }
}

tensor<output> elu(tensor<input> &a, float alpha = 1.0) {
  Elu<input> op("localOpName", alpha);
  return op.compute(a);
  dtype = {
    "float" : "float"
  }
}

tensor<output> equal(tensor<input> &a, tensor<input> &b) {
  Equal<input> op;
  return op.compute(a, b);
  dtype = {
    "float" : "bool",
    "int" : "bool",
    "bool" : "bool"
  }
}
```
</details>

### Caution :
* Everything inside `dnnc.api` is **whitespace** and **newline** sensitive, so try to keep the structure similar.
* Make sure to add a blank line between 2 operators.
* Don't leave any blank lines inside operators' functions.
* Don't leave more than one blank line anywhere.
* Use comment syntax (`/*` or `*/`) in the same line as the code. 
  - Like at front ` /* tensor<output> ... ` and at end ` } */ `
* After adding new operator in `dnnc.api`, run the `op_gen.py` at the same directory.

* It will generate `dnnc_swig_externs.h` and `dnnc_api.cpp` file, and you are done here.
  - If you are curious, why we are creating `dnnc_swig_externs.h`, earlier we were adding externs inside `dnnc.i` file twice,
    but now, we will create a cpp header file and call that file twice inside `dnnc.i`. Not everything inside `dnnc.i` can
    be automated, but we used automation, where we could.
* Run Make command now, like you did previously. And everything will be same after that.

* Everything except **dtype** block is a cpp block, and **dtype** is a python dictionary which contains all kinds of input output datatype combination possible for the operators

```python
dtype = {
  "input1" : "output1",
  "input2" : "output2",
  "input2" : "output1",
  ...
}
```
