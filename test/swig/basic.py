import os,sys
sys.path.append(os.path.abspath('.'));

import dnnc
t1=dnnc.make_tensor(2,3)
t2=dnnc.make_tensor(3,2)

mul = dnnc.multiply(t1,t2)
#print ("multiplication : " , mul.to_string())
add = dnnc.add(t1,t1)
#print ("addition : " , add.to_string())

t3 = dnnc.make_tensor(2,3,4)
#print("old shape", t1.shape())
new_shape = dnnc.ivec([2,12])
t3.reshape(new_shape)
#print("new shape", t1.shape())

#t4 = dnnc.thresholded_relu(t1);
#print("relu", t4.to_string())

#replace first few values in tensor with new values.
data = dnnc.fvec([1.0, 2.0, 3.0, 4.0])
t3.load(data)
#print(t3.to_string())



def test_multiply(a,b):
    c = dnnc.multiply(a, b)
    #print(c.to_string())


#3D MatMul Test1
a = dnnc.make_tensor(2, 2, 2)
b = dnnc.make_tensor(2, 2, 2)
adata = dnnc.fvec([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0])
bdata = dnnc.fvec([8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0])
a.load(adata)
b.load(bdata)
test_multiply(a,b)

#3D MatMul Test2
a = dnnc.make_tensor(2, 3, 2)
b = dnnc.make_tensor(3, 2, 2)
adata = dnnc.fvec([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0])
bdata = dnnc.fvec([12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0])
a.load(adata)
b.load(bdata)
test_multiply(a,b)
