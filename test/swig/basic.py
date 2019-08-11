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
#new_shape = dnnc.ivec([2,12])
#@t3.reshape(new_shape)
#print("new shape", t1.shape())
