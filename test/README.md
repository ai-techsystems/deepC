# Help for writing Python Tests

## Third party software used

1. [Python unit testing framework](https://docs.python.org/2/library/unittest.html) aka PyUnit
1. [Numpy Testing assertions](https://docs.scipy.org/doc/numpy/reference/routines.testing.html)
1. **[Skip examples and jump straight to the Instructions](#instruction-for-your-implementations)** (Go through the examples at least once)

## DNNC Python APIs
DNNC framework is easy to get started since most of APIs closely resemble numpy.
For the sake of simplicity, team has chosen to go with *Row Major* form of tensors for the beta releases. 
DNNC supports upto 4D tensors in the beta release

### DNNC Tensor APIs

```python
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
```python
>>> a.<tab><tab>
a.broadcast(       a.empty(           a.name(            a.this             
a.data(            a.flatten(         a.rank(            a.to_proto(        
a.dtype(           a.length(          a.reshape(         a.to_string(       
a.eigen_to_numpy(  a.load(            a.shape(           a.transpose(  

```

### DNNC APIs

**Matrix Multiplication Example**

```python
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

```python
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

```python
import dnnc as dc
import numpy as np
import unittest

class MatMulTest(unittest.TestCase):
    
    def setUp(self):
        self.len = 12
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.np_b = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a));
        self.dc_b = dc.array(list(self.np_b));

    def test_MatMul_1D (self):
        npr = np.matmul(self.np_a, self.np_b)
        dcr = dc.matmul(self.dc_a, self.dc_b)
        np.testing.assert_allclose(npr, np.array(dcr.data()[0]).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_MatMul_2D (self):
        np_a = np.reshape(self.np_a, (3,4))
        np_b = np.reshape(self.np_b, (4,3))
        dc_a = dc.reshape(self.dc_a, (3,4));
        dc_b = dc.reshape(self.dc_b, (4,3));
        
        npr = np.matmul(np_a, np_b);
        dcr = dc.matmul(dc_a, dc_b);
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_MatMul_3D (self):
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
---

# Instruction for your implementations:

- I have moved your work on the python interface i.e. **dnnc.i** and **dnnc_api.cpp** to a **[temporary folder](https://github.com/ai-techsystems/dnnc-operators/tree/master/temp/swig)**

- **Because not all the operators work with the interface, so be sure to only push the repository when your** `make` **command doesn't fail. This is very important, as this could take your fellow developers' valuable time trying to fix the bug, rather than implementing test cases.**

- So only push when you add the header files in the interface, and that works with python interface.

- So below is the cycle you have to go through, in the next days.
    - Add one of your operator in the **dnnc.i** and **dnnc_api.cpp**.
    - Go to the **dnnc-operator/swig** folder, and run `make` command to compile the newly added operators in the interface.
    - If everything works fine, then you successfully added your operator in the **dnnc python interface**. And you can check it by importing **dnnc** in python bash.
    - Now go to the **dnnc-operator/test/swig** folder, here you will see the **MatMul.py** and **Add.py** examples. Add your operators like the above examples.
    - After adding your code, run the code from the same directory like this.
        
        ```console
        python MatMul.py
        ```
        or you can also use
        
        ```console
        python -m unittest MatMul.MatMulTest
        ```
    - If you want to see **Verbose output** of your execution, add `-v` at the end of your command. Like
        
        ```console
        python MatMul.py -v
        ```
    - It will show which test cases work, and which doesn't and shows the reason too.

    - To run all of the unittests at once, use this.

        ```console
        python -m unittest *py -v
        ```

    - Then update the **[README](https://github.com/ai-techsystems/dnnc-operators/blob/master/README.md)** to show which test case is not working, or if all are working.

    - Now look at the code, here are some things, if you need to know about the lines of the implementation.

---
#### Test case for Add operator is given below

```python
    
    import os,sys

    # adding "dnnc-operators/swig" folder to the import directory, we will import "dnnc.py" from here
    sys.path.append(os.path.abspath('..'+os.path.sep+'..'+os.path.sep+'swig'));

    import dnnc as dc       # import dnnc
    import numpy as np      # import numpy to cross check against dnnc
    import unittest         # to test our testcases

    class AddTest(unittest.TestCase):
        
        # This is like __init__ method of class, but for unittest, it just 
        # declares the variables everytime before testcases are tested.
        def setUp(self):
            
            # declare the total size of the tensor
            self.len = 24

            # use numpy random to generate random numbers
            self.np_a = np.random.randn(self.len).astype(np.float32)
            self.np_b = np.random.randn(self.len).astype(np.float32)
            #self.np_a = np.arange(self.len).astype(np.float32)
            #self.np_b = np.arange(self.len).astype(np.float32)
            
            # set the same numbers as input to dnnc, so cross checking happens on same input
            self.dc_a = dc.array(list(self.np_a));
            self.dc_b = dc.array(list(self.np_b));

        # First test case for !D
        # remember, for unittest module to know which method should be tested, you have to give 
        # test_as prefix to the method
        def test_Add1D (self):

            # npr is the tensor which stores the result of numpy add of those two input tensors
            npr = np.add(self.np_a, self.np_b)
            
            # dcr is the tensor which stores the result of dnnc add of those same input tensors
            dcr = dc.add(self.dc_a, self.dc_b)

            # assert_allclose checks the two tensor values, with tolerance of 
            # atol (Absolute tolerance), rtol (Relative tolerance)
            np.testing.assert_allclose(npr, np.array(dcr.data()[0]).astype(np.float32),
                    rtol=1e-3, atol=1e-3)

        # Second test case for 2D
        def test_Add2D (self):

            #for 2d we have to reshape each of np_a, np_b, dc_a, dc_b
            np_a = np.reshape(self.np_a, (6,4))
            np_b = np.reshape(self.np_b, (6,4))
            dc_a = dc.reshape(self.dc_a, (6,4));
            dc_b = dc.reshape(self.dc_b, (6,4));
            
            npr = np.add(np_a, np_b);
            dcr = dc.add(dc_a, dc_b);
            
            np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                    rtol=1e-3, atol=1e-3)

        def test_Add3D (self):
            np_a = np.reshape(self.np_a, (2,4,3))
            np_b = np.reshape(self.np_b, (2,4,3))
            dc_a = dc.reshape(self.dc_a, (2,4,3));
            dc_b = dc.reshape(self.dc_b, (2,4,3));

            npr = np.add(np_a, np_b);
            dcr = dc.add(dc_a, dc_b);

            np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                    rtol=1e-3, atol=1e-3)

        def test_Add4D (self):
            np_a = np.reshape(self.np_a, (2,2,2,3))
            np_b = np.reshape(self.np_b, (2,2,2,3))
            dc_a = dc.reshape(self.dc_a, (2,2,2,3));
            dc_b = dc.reshape(self.dc_b, (2,2,2,3));

            npr = np.add(np_a, np_b);
            dcr = dc.add(dc_a, dc_b);

            np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                    rtol=1e-3, atol=1e-3)

    # when the program gets called
    if __name__ == '__main__':
        unittest.main()

```
---
##### This should give you the idea of how the code is working.