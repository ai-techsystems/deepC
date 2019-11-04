import time
import dnnc as dc
import numpy as np

t1 = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
t3 = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]],[[12,13,14],[15,16,17]]])
t2 = dc.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
start = dc.array([1,1]).asTypeULong()
stop = dc.array([2,2]).asTypeULong()
axis = dc.array([0,1]).asTypeInt()
step = dc.array([1,1]).asTypeULong()

result = ""

def run(function):
	start_time = time.time()
	print(function)
	exec(function)
	stop_time = time.time()
	elapsed_time = stop_time - start_time
	print("Took %.8s" % elapsed_time, "secs\n\n")
	return

def main():
	
	run("print(t1)")
	# t1[2:3,1:2] = 8
	# run("print(t1)")
	
	
	
	# run("print(t1[1::1,:])")
	# run("print(t1[2,1:3])")
	
	run("print(t1[2])")
	# run("print(t2[2])")
	
	run("print(t1[2,:])")
	run("print(t1[5:6,2])")
	
	run("print(t1[2:3,1:2])")
	run("print(t2[2:3,1:2])")
	run("print(t2[1,:])")
	
	run("print(t1[1:2:1,1:2])")
	run("print(t2[1:2:1,1:2])")
	
	# run("print(dc.slice(t2, start, stop, axis, step))")


if __name__ == "__main__":
	main()