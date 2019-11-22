import os

def main():

	# Mac and Linux
	if os.name == "posix":
		os.system('sudo docker build -t dnnc .')
		os.system('sudo docker run -it dnnc /bin/bash -c "cd /dnnCompiler && make clean && make all"')

	# Windows
	elif os.name == "nt":
		os.system('docker build -t dnnc .')
		# don't use single quotes inside command, always use duble quotes, similar problem listed below
		# https://stackoverflow.com/questions/24673698/unexpected-eof-while-looking-for-matching-while-using-sed
		os.system('docker run -it dnnc /bin/bash -c "cd /dnnCompiler && make clean && make all"')


if __name__ == "__main__":
	main()