import time
import dnnc as dc
import numpy as np

t_np = np.array([[12,1,2],[3,4,5],[6,7,8],[9,10,11]])
t_dc = dc.array([[12,1,2],[3,4,5],[6,7,8],[9,10,11]])
t_np1 = np.array([[21,22],[23,24],[26,27]])
t_dc1 = dc.array([[21,22],[23,24],[26,27]])
t_dc2 = dc.arange(10)


start = dc.array([1,1]).asTypeInt()
stop = dc.array([-1,-1]).asTypeInt()
axis = dc.array([0,1]).asTypeInt()
step = dc.array([-1,-1]).asTypeInt()


def run(function):
	print("\n>>>", function)
	start_time = time.time()
	exec(function)
	stop_time = time.time()
	elapsed_time = stop_time - start_time
	print("Took %.8s" % elapsed_time, "secs\n")
	return

def tested():

	run("print(t_np[2])")
	run("print(t_dc[2])")

	run("print(t_np[2,1])")
	run("print(t_dc[2,1])")

	run("print(t_np[2:3,:])")
	run("print(t_dc[2:3,:])")

	run("print(t_np[2])")
	run("print(t_dc[2])")

	run("print(t_np[2,:])")
	run("print(t_dc[2,:])")

	run("print(t_np[2:3,1:2])")
	run("print(t_dc[2:3,1:2])")

	run("print(t_np[1,::2])")
	run("print(t_dc[1,::2])")

	run("print(t_np[1:2:1,1:2])")
	run("print(t_dc[1:2:1,1:2])")

	run("print(t_np[...])")
	run("print(t_dc[...])")

	run("print(t_np[2:3,...])")
	run("print(t_dc[2:3,...])")

	run("print(t_np[...,2:3:3])")
	run("print(t_dc[...,2:3:3])")

	run("print(t_np[:,1])")
	run("print(t_dc[:,1])")

	run("print(t_np[...,1])")
	run("print(t_dc[...,1])")

	run("print(t_np[-2])")
	run("print(t_dc[-2])")

	run("print(t_np[-1:,-2:])")
	run("print(t_dc[-1:,-2:])")

	run("print(t_np[1::-1,1::-1])")
	run("print(dc.slice(t_dc, start, stop, axis, step))")

	run("print(t_np[2::-1,2::-1])")
	run("print(t_dc[2::-1,2::-1])")

	run("print(t_np[::-2,1::-1])")
	run("print(t_dc[::-2,1::-1])")

	run("print(t_np[...,3:-1:-1])")
	run("print(t_dc[...,3:-1:-1])")

	return

def testing():
	run("print(t_np)")
	run("print(t_dc)")
	t_np[0:3,0:2] = t_np1
	t_dc[0:3,0:2] = t_dc1
	run("print(t_np)")
	print(t_dc[2:3,1:2])
	run("print(t_dc)")
	t_np[2:3,1:2] = 30
	run("print(t_np)")
	t_dc[2:3,1:2] = 30
	run("print(t_dc)")
	t_dc[2,1] = 1
	run("print(t_dc)")
	run("print(t_dc2)")
	run("print(t_dc2[2])")
	t_dc2[2] = True
	run("print(t_dc2)")

	return

def main():
	run("print(t_np)")

	tested()
	testing()


if __name__ == "__main__":
	main()