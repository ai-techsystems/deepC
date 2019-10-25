# dnnc.api

## Brief understanding of working principle.

## Example 1

#### Structure

```python
dtype = {
		"output" : "input",
	}
```
#### Code

```python
tensor<output> example(tensor<input> &a, float alpha = 1.0) {
	Example<output, input> op("localOpName", alpha);
	return op.compute(a, b);
	dtype = {
		"double" : "double",
		"float" : "int",
		"int" : "bool",
	}
}
```
#### Output

```cpp
tensor<double> example(tensor<double> &a, float alpha = 1.0) {
	Example<double, double> op("localOpName", alpha);
	return op.compute(a, b);
}

tensor<float> example(tensor<int> &a, float alpha = 1.0) {
	Example<float, int> op("localOpName", alpha);
	return op.compute(a, b);
}

tensor<int> example(tensor<bool> &a, float alpha = 1.0) {
	Example<int, bool> op("localOpName", alpha);
	return op.compute(a, b);
}
```
## Example 2

#### Structure

```python
dtype = {
		"output" : ("input1","input2","input3"),
	}
```
#### Code

```python
tensor<output> example(tensor<input> &a, float alpha = 1.0) {
	Example<output, input> op("localOpName", alpha);
	return op.compute(a, b);
	dtype = {
		"double" : ("double","float","int"),
	}
}
```
#### Output

```cpp
tensor<double> example(tensor<double> &a, float alpha = 1.0) {
	Example<double, double> op("localOpName", alpha);
	return op.compute(a, b);
}

tensor<double> example(tensor<float> &a, float alpha = 1.0) {
	Example<double, float> op("localOpName", alpha);
	return op.compute(a, b);
}

tensor<double> example(tensor<int> &a, float alpha = 1.0) {
	Example<double, int> op("localOpName", alpha);
	return op.compute(a, b);
}
```

## Example 3

#### Structure

```python
dtype = {
		"output" : ["input1","input2","input3"],
	}
```
#### Code

```python
tensor<output> example(tensor<input1> &a, tensor<input2> &b, tensor<input3> &c, float alpha = 1.0) {
	Example<output, input1, input2> op("localOpName", alpha);
	return op.compute(a, b, c);
	dtype = {
		"double" : ["double","float","double"],
		"float" : ["float","float","double"],
	}
}
```
#### Output

```cpp
tensor<double> example(tensor<double> &a, tensor<float> &b, tensor<double> &c, float alpha = 1.0) {
	Example<double, double, float> op("localOpName", alpha);
	return op.compute(a, b, c);
}

tensor<float> example(tensor<float> &a, tensor<float> &b, tensor<double> &c, float alpha = 1.0) {
	Example<float, float, float> op("localOpName", alpha);
	return op.compute(a, b, c);
}
```

## Example 4

#### Structure

```python
dtype = {
		"output" :(["input1","input2","input3"],["input1","input2","input3"]),
	}
```
#### Code

```python
tensor<output> example(tensor<input1> &a, tensor<input2> &b, tensor<input3> &c, float alpha = 1.0) {
	Example<output, input1, input2, input3> op("localOpName", alpha);
	return op.compute(a, b, c);
	dtype = {
		"double" : (["double","float","double"], ["float","float","double"]),
		"float" : ["float","double","float"],
	}
}
```
#### Output

```cpp
tensor<double> example(tensor<double> &a, tensor<float> &b, tensor<double> &c, float alpha = 1.0) {
	Example<double, double, float, double> op("localOpName", alpha);
	return op.compute(a, b, c);
}

tensor<double> example(tensor<float> &a, tensor<float> &b, tensor<double> &c, float alpha = 1.0) {
	Example<double, float, float, double> op("localOpName", alpha);
	return op.compute(a, b, c);
}

tensor<float> example(tensor<float> &a, tensor<double> &b, tensor<float> &c, float alpha = 1.0) {
	Example<float, float, double, float> op("localOpName", alpha);
	return op.compute(a, b, c);
}
```