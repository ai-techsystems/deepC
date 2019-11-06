import os, argparse
import common

import deepC.dnnc as dc
import numpy as np

def test_multiply(a,b):
    c = dc.matmul(a, b)
    #print(c)

def test_non_detailed():

	t1=dc.array(2,3)
	t2=dc.array(3,2)

	mul = dc.matmul(t1,t2)
	#print ("multiplication : " , mul.to_string())
	
	t3 = dc.array(2,3,4)
	#print("old shape", t1.shape())
	new_shape = dc.vectorSizeT([2,12])
	t3.reshape(new_shape)
	t3.reshape(4,6)
	t3.reshape((4,6))
	t3.reshape([4,6])
	#print("new shape", t1.shape())

	# bug ref: #98
	# py_list = list(t3);   # convert tensor to python list
	# py_tuple = tuple(t3); # convert tensor to python tuple
	np_ary = t3.numpy();  # convert to numpy array

	#t4 = dc.thresholded_relu(t1);
	#print("relu", t4.to_string())

	#replace first few values in tensor with new values.
	data = dc.vectorFloat([1.0, 2.0, 3.0, 4.0])
	t3.load(data)
	#print(t3.to_string())

	arr = dc.array([1, 2])
	#print(arr)
	arr2D = dc.array([[1, 2], [10, 20]]).asTypeInt()
	#print(arr2D)
	arrRand = dc.random(2, 3);
	#print(arrRand)
	empty = dc.empty(3, 2);
	#print(empty)
	zeros = dc.zeros(2, 2);
	#print(zeros);
	ones = dc.ones(2, 2);
	#print(ones)
	ranges = dc.arange(15, 3, 2)
	#print(ranges)

	dc.reshape(arr2D, (1,4))

	#3D MatMul Test1
	a = dc.array(2, 2, 2)
	b = dc.array(2, 2, 2)
	adata = dc.vectorFloat([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0])
	bdata = dc.vectorFloat([8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0])
	a.load(adata)
	b.load(bdata)
	test_multiply(a,b)

	#3D MatMul Test2
	a = dc.array(2, 2 ,3)
	b = dc.array(2 ,3, 2)
	adata = dc.vectorFloat([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0])
	bdata = dc.vectorFloat([12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0])
	a.load(adata)
	b.load(bdata)
	test_multiply(a,b)


def test_detailed():

	t1 = dc.array(2,3).asTypeFloat()
	t2 = dc.array(2,3).asTypeInt()

	add = dc.add(t1,t1)
	add = t1 + t1

	#print ("addition : " , add.to_string())

	t_dc = dc.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])

	t_dc[2]
	t_dc[2,1]
	int(t_dc[2,1])
	t_dc[2:3,:]
	t_dc[2]
	t_dc[2,:]
	t_dc[2:3,1:2]
	t_dc[1,::2]
	t_dc[1:2:1,1:2]
	t_dc[1:2:1,...]
	t_dc[...,1]


def main():

	parser = argparse.ArgumentParser(description="basic testing of deepC.dnnc")
	parser.add_argument("-dev", "--developer", action="store_true", help="skip testing binary operators only for faster development purposes")
	args = parser.parse_args()

	test_non_detailed()
	
	if not args.developer:
		test_detailed()

if __name__ == "__main__":
	main()