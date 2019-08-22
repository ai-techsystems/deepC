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
>>>
>>> # create a vector of 5 elements
>>> a=dc.arange(5)
>>> print(a)
[0.000000 1.000000 2.000000 3.000000 4.000000]
>>>
>>> # create 2D arrary from python list
>>> a=dc.array([[10,11,12],[20,21,22]])
>>> a.shape()
(2, 3)
>>> print(a)
[[10.000000 11.000000 12.000000]
 [12.000000 20.000000 21.000000]]

>>> # reshape 2x3 matrix to 3x2 matrix
>>> a=a.reshape(dc.ivec([3,2]))
>>> a.shape()
(3, 2)
>>> print(a)
[[10.000000 11.000000]
 [20.000000 21.000000]
 [0.000000 0.000000]]

>>>
>>> # Other tensor functions availiable
>>> a.<tab><tab>
a.broadcast(       a.empty(           a.name(            a.this             
a.data(            a.flatten(         a.rank(            a.to_proto(        
a.dtype(           a.length(          a.reshape(         a.to_string(       
a.eigen_to_numpy(  a.load(            a.shape(           a.transpose(  

```

### DNNC APIs

``` 
>>> 
>>> b=dc.reshape(a,(3,2))
>>> b.shape()
(3, 2)
>>> y=dc.matmul(a,b)
>>> print(y)
[[484.000000 594.000000]
 [914.000000 1124.000000]]
>>>
>>> # Other DNNC APIs
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
