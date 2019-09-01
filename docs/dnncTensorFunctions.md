# DNNC functions to mimic numpy using Eigen

---
### Most remarkable fluid features (Array creation):

**Legends** ✔️: ToBeDone, ✅:Done, 🚫:Rejected
```python

✅a = numpy.array([1,2,3])

✅numpy.array([1,2]) #1D
✅numpy.array([[1,2],[10,20]]) #2D

# For complex types
🚫numpy.array([1,2], dtype=complex) #1D complex

# For randomized 3d array
🚫Array3d = numpy.random.randint(10, size=(3, 4, 5))

# generate uniformly distributed numbers
✅a = numpy.random.rand(3,2)   #(3 rows, 2 cols)

# Create empty 2D array (2 rows, 3 columns)
✅a_empty = numpy.empty(2,3)

# Create 0 initiallized 2D array (3 rows, 2 columns)
✅numpy.zeros(3,2) 

# Create 1 initiallized 2D array (3 rows, 2 columns)
✅numpy.ones(3,2) 

# Create a range of elements
✅array = numpy.arange(3) # array will contain 0,1,2

# Create a Numpy array from Python sequence of elements
✔️a = numpy.asarray([1,2])

# Create an array with values that are evenly spaced
✔️a = numpy.array(0,6,2) # create 0-5, 2 apart, returns [0,2,4]

# Ccreate an array where the values are linearly spaced between an interval    numpy.linspace(first, last, number)
✔️a = numpy.linspace(0,10,5) # returns [0,2.5,5,7.5,10]

```
---
### Some advanced features (Array manipulation)

```python

# Add
a = [3,4,5]
a = numpy.append(a, [1,2]) #returns [3,4,5,1,2]

#Join
numpy.concatenate(a,b)
numpy.stack(a,b)
numpy.hstack(a,b)
numpy.vstack(a,b)

# Delete
a = numpy.delete(array,2) # 2 is going to be deleted from the array

# Sort
numpy.sort(array1, axis=1, kind = 'quicksort', order ='column name')

# Deep copy
new_array = numpy.copy(array)

```
---
### Array functions and attributes

##### Shape and Dimension

```python

# Shape
array = numpy.array([[..],[..]])
array.shape

# Reshape by setting shape property
array.shape = (1,2) # (1 row, 2 columns)
# resize(x,y) can also be used to resize an array

# Dimensions of an array:
array.dim

# Find length of each element of an array:
array.itemsize

```
##### Slicing

```python

array = numpy.arange(100)

# Get 3rd element:
array[2]

# Get items within indexes
array[3:5] #3 is start, 5 is end

# Get 3-10 element, step size 4 increments:
array[2:9:4]

# Get all elements from 2nd element onwards
array[1:]

# Can also pass in N-Dimensional Index
array[[0,1],[1,2]]

# Get all NAN elements
array[numpy.isnan(array)]

# Using where()
numpy.where(array > 5) # will return all elements that meet the criteria

```
##### Broadcasting

```python

# 5 rows, 3 columns array
bigger_array = arange(5,3) 

# 5 rows, 1 column array
smaller_array = arange(5)

final_array = bigger_array + smaller_array

```
---
### Mathematical functions
---
#### Can be done through DNNC opearators:

* **:heavy_check_mark: numpy.sin()**
* **:heavy_check_mark: numpy.cos()**
* **:heavy_check_mark: numpy.tan()**
* **:heavy_check_mark: numpy.sinh()**
* **:heavy_check_mark: numpy.cosh()**
* **:heavy_check_mark: numpy.tanh()**
* **:heavy_check_mark: numpy.arcsin()**
* **:heavy_check_mark: numpy.arccos()**
* **:heavy_check_mark: numpy.arctan()**
* **:heavy_check_mark: numpy.arcsinh()**
* **:heavy_check_mark: numpy.arccosh()**
* **:heavy_check_mark: numpy.arctanh()**
* **:heavy_check_mark: numpy.add()**
* **:heavy_check_mark: numpy.subtract()**
* **:heavy_check_mark: numpy.cross()**
* **:heavy_check_mark: numpy.divide()**
* **:x: numpy.power()**
* **:heavy_check_mark: numpy.round()**
* **:heavy_check_mark: numpy.floor()**
* **:heavy_check_mark: numpy.ceil()**
* **:heavy_check_mark: numpy.exp()**
* **:heavy_check_mark: numpy.log()**
* **:heavy_check_mark: numpy.sqrt()**
* **:heavy_check_mark: numpy.absolute()**
* **:x: numpy.clip()**
* **:x: numpy.convolve()**

---

#### Have to be implemented:

* **numpy.dot()**
	 > dot product of 2 arrays
* **numpy.inner()**
	 > inner product of 2 arrays
* **numpy.determinant()**
	 > determinant of an array
* **numpy.transpose()**
	 > permute the dimensions of matrix
* **numpy.inverse()**
	 > inverse of a matrix
* **numpy.solve()**
	 > solves matrix equation
* **numpy.multiply()**
     > element wise multiplication of 2 arrays (not to be confused with matrix multiplication)
* **numpy.true_divide()**
     > element wise division of 2 arrays (uses `/` in python)
* **numpy.floor_divide()**
     > element wise division of 2 arrays (uses `//` in python)
* **numpy.degrees() / numpy.rad2deg()**
     > radian to degree converter
* **numpy.radians() / numpy.deg2rad()**
     > degree to radian converter
* **numpy.median()**
	 > Finds the median
* **numpy.average()**
	 > Finds average
* **numpy.mean()**
	 > Finds mean
* **numpy.var()**
	 > Finds variance
* **numpy.rint()**
     > round elements of the array to the nearest integer
* **numpy.fix()**
     > round elements of the array to the nearest integer towards zero
* **numpy.trunc()**
     > returns the truncated value of the elements of array
* **numpy.log10()**
     > return the base 10 logarithm of the input array, element-wise
* **numpy.log2()**
     > return the base 10 logarithm of the input array, element-wise
* **numpy.expm1()**
     > calculate exp(x) – 1 for all elements in the array
* **numpy.exp2()**
     > calculate (2^p) for all p in the input array
* **numpy.logaddexp()**
     > logarithm of the sum of exponentiations of the inputs
* **numpy.logaddexp2()**
     > logarithm of the sum of exponentiations of the inputs in base-2
* **numpy.reciprocal()**
     > calculate (1/x) for all x in the input array
* **numpy.positive()**
     > make every element positive
* **numpy.negetive()**
     > make every element negetive
* **numpy.remainder()**
     > return element wise remainder of division
* **numpy.divmod()**
     > return element-wise quotient and remainder simultaneously
* **numpy.isreal()**
     > test element-wise whether it is a real number or not(not infinity or not Not a Number) and return the result as a boolean array
* **numpy.conj()**
     > The conjugate of a complex number is obtained by changing the sign of its imaginary part. If the complex number is (2+5j) then its conjugate is (2-5j)
* **numpy.cbrt()**
     > mathematical function helps user to calculate cube root of x for all x being the array elements
* **numpy.square()**
	 > return the non-negative square-root of an array, element-wise
* **numpy.maximum()**
	 > find the element-wise maximum of array elements
* **numpy.minimum()**
	 > find the element-wise minimum of array elements
* **numpy.interp()**
	 > returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x
* **numpy.nan_to_num()**
	 > replace NaN with zero and infinity with large finite numbers
* **numpy.real_if_close()**
	 > if complex input returns a real array if complex parts are close to zero
* **numpy.heaviside()**
	 > heaviside(x1, x2) = {0   if x1 < 0}, {x2   if x1 == 0}. {1   if x1 > 0}       




---
## Resource:

* Basic Functions are taken from **[Medium](https://medium.com/fintechexplained/why-should-we-use-numpy-c14a4fb03ee9)**
* Mathematical functions are taken from **[GeeksForGeeks](https://www.geeksforgeeks.org/numpy-mathematical-function/)**
