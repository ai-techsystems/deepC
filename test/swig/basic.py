import os,sys
sys.path.append(os.path.abspath('.'));

import dnnc
a=dnnc.make_tensor(2,3)
b=dnnc.make_tensor(3,2)

mul = dnnc.multiply(a,b)
add = dnnc.add(a,a)
