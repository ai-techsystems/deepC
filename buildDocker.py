import os, argparse

def main():

	parser = argparse.ArgumentParser(description="run docker in any os")
	parser.add_argument("-dev", "--developer", action="store_true", help="ssh inside docker container without running make")
	args = parser.parse_args()

	if args.developer:
		# Mac and Linux
		if os.name == "posix":
			try:
				os.system('systemctl start docker')
			except:
				os.system('systemctl unmask docker.service && systemctl unmask docker.socket && systemctl start docker.service')
			os.system('sudo docker build -t dnnc .')
			os.system('sudo docker run -it dnnc /bin/bash')

		# Windows
		elif os.name == "nt":
			os.system('docker build -t dnnc .')
			# don't use single quotes inside command, always use duble quotes, similar problem listed below
			# https://stackoverflow.com/questions/24673698/unexpected-eof-while-looking-for-matching-while-using-sed
			os.system('docker run -it dnnc /bin/bash -c "cd /dnnCompiler && make clean && make all"')

	else:
		# Mac and Linux
		if os.name == "posix":
			try:
				os.system('systemctl start docker')
			except:
				os.system('systemctl unmask docker.service && systemctl unmask docker.socket && systemctl start docker.service')
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