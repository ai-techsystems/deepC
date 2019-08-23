# Help for writing Python Tests

## Third party software used

1. [Python unit testing framework](https://docs.python.org/2/library/unittest.html) aka PyUnit
1. [Numpy Testing assertions](https://docs.scipy.org/doc/numpy/reference/routines.testing.html)

## DNNC Python APIs
DNNC framework is easy to get started since most of APIs closely resemble numpy.
For the sake of simplicity, team has chosen to go with *Row Major* form of tensors for the beta releases. 
DNNC supports upto 4D tensors in the beta release

### DNNC Tensor APIs

```
>>> import dnnc as dc

>>> a=dc.arange(5) ;                                     # create a vector of 5 elements
>>> print(a)
[0.000000 1.000000 2.000000 3.000000 4.000000]
>>> a[1]                                                 # print second element of the array.
1.0
>>> a[1] = 100.1                                         # assign second element a new value 100.1
>>> print(a)                                             # print and check new value.
[0.000000 100.099998 2.000000 3.000000 4.000000]


>>> a=dc.array([[10,11,12],[20,21,22]])                  # create 2D arrary from python list
>>> a.shape()                                            # check shape.
(2, 3)
>>> print(a)                                             # print 2D arrary.
[[10.000000 11.000000 12.000000]
 [12.000000 20.000000 21.000000]]


>>> a.reshape(dc.ivec([3,2]))                            # reshape 2x3 matrix to 3x2 matrix
>>> a.shape()
(3, 2)
>>> print(a)
[[10.000000 11.000000]
 [20.000000 21.000000]
 [0.000000 0.000000]]
```


**Other tensor functions**
```
>>> a.<tab><tab>
a.broadcast(       a.empty(           a.name(            a.this             
a.data(            a.flatten(         a.rank(            a.to_proto(        
a.dtype(           a.length(          a.reshape(         a.to_string(       
a.eigen_to_numpy(  a.load(            a.shape(           a.transpose(  

```

### DNNC APIs

**Matrix Multiplication Example**

```
>>> a=dc.array([[10,11,12],[20,21,22]])                  # create 'a' 2x3 matrix 
>>> b=dc.array([[10,11,12],[20,21,22]])                  # create 'b' 2x3 matrix 
>>> dc.reshape(b,(3,2))                                  # reshape matrix 'b' to 3x2
>>> b.shape()
(3, 2)
>>> y=dc.matmul(a,b)                                     # multiply 'a' and 'b'.
>>> print(y)
[[484.000000 594.000000]
 [914.000000 1124.000000]]
 ```
 
**Other DNNC APIs**

```
>>> dc.<tab><tab>
dc.add(                          dc.matmul(
dc.arange(                       dc.ones(
dc.array(                        dc.random(
dc.dTensor(                      dc.reshape(
dc.empty(                        dc.zeros(
dc.ivec                          dc.fvec
dc.iTensor(                      dc.fTensor(
dc.thresholded_relu(
```

## Writing a unit test for Operator MatMul 

```
import dnnc as dc
import numpy as np
import unittest

class MatMulTest(unittest.TestCase):
    def __init__(self):
        self.len = 12
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.np_b = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a));
        self.dc_b = dc.array(list(self.np_b));

    def MatMul_1D (self):
        npr = np.matmul(self.np_a, self.np_b)
        dcr = dc.matmul(self.dc_a, self.dc_b)
        np.testing.assert_allclose(npr, np.array(dcr.data()[0]).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def MatMul_2D (self):
        np_a = np.reshape(self.np_a, (3,4))
        np_b = np.reshape(self.np_b, (4,3))
        dc_a = dc.reshape(self.dc_a, (3,4));
        dc_b = dc.reshape(self.dc_b, (4,3));
        npr = np.matmul(np_a, np_b);
        dcr = dc.matmul(dc_a, dc_b);
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def MatMul_3D (self):
        np_a = np.reshape(self.np_a, (2,2,3))
        np_b = np.reshape(self.np_b, (2,3,2))
        dc_a = dc.reshape(self.dc_a, (2,2,3));
        dc_b = dc.reshape(self.dc_b, (2,3,2));

        npr = np.matmul(np_a, np_b);
        dcr = dc.matmul(dc_a, dc_b);

        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    unittest.main()
```
