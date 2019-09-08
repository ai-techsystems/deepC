# What is DNNC Tensor
Tensor variable is a fixed-length multi-dimensional array. 
It is a mutable object with ability to transform constrained by algebraic rules. 
Similar to NumPy ndarray objects, dnnc.tensor objects have a data type and a shape. 
Additionally, dnnc.tensors can reside in accelerator memory (like a GPU). 
DNNC will offers a rich library of operations (dnnc.add, dnnc.matmul, dnnc.transpose etc.) that consume and produce dnnc.tensors.
Here is a partial list.

# Tensor Operations

## Assignment Operators:
```
Operator          Method
+=                object.__iadd__(self, other)
-=                object.__isub__(self, other)
*=                object.__imul__(self, other)
/=                object.__idiv__(self, other)
//=               object.__ifloordiv__(self, other)
%=                object.__imod__(self, other)
**=               object.__ipow__(self, other[, modulo])
<<=               object.__ilshift__(self, other)
>>=               object.__irshift__(self, other)
&=                object.__iand__(self, other)
^=                object.__ixor__(self, other)
|=                object.__ior__(self, other)
```

## Comparison Operators
```
Operator          Method
<                 object.__lt__(self, other)
<=                object.__le__(self, other)
==                object.__eq__(self, other)
!=                object.__ne__(self, other)
>=                object.__ge__(self, other)
>                 object.__gt__(self, other)
```

## Unary Operators:
```
Operator          Method
-                 object.__neg__(self)
+                 object.__pos__(self)
abs()             object.__abs__(self)
~                 object.__invert__(self)
complex()         object.__complex__(self)
int()             object.__int__(self)
long()            object.__long__(self)
float()           object.__float__(self)
oct()             object.__oct__(self)
hex()             object.__hex__(self)
```

## Binary Operators
```
Operator           Method
+                  object.__add__(self, other)
-                  object.__sub__(self, other)
*                  object.__mul__(self, other)
//                 object.__floordiv__(self, other)
/                  object.__div__(self, other)
%                  object.__mod__(self, other)
**                 object.__pow__(self, other[, modulo])
<<                 object.__lshift__(self, other)
>>                 object.__rshift__(self, other)
&                  object.__and__(self, other)
^                  object.__xor__(self, other)
|                  object.__or__(self, other)
```

**Reference:** 
1. [Python Data model](https://docs.python.org/3/reference/datamodel.html)
