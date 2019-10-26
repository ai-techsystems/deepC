![](https://www.docker.com/sites/default/files/social/docker_facebook_share.png)

# [Docker](https://www.docker.com/)

Read the official documentation **[here](https://docs.docker.com/)**

## Downloads 
#### Windows 10
* **[Docker for Windows 10](https://docs.docker.com/v17.09/docker-for-windows/install/#download-docker-for-windows)**

#### Mac
* **[Docker for Mac](https://docs.docker.com/v17.09/docker-for-mac/install/#download-docker-for-mac)**

#### Ubuntu
```bash
sudo apt-get update && apt-get install docker 
```

#### Arch Linux/ Manjaro
```bash
sudo pacman -Syu docker
```
## Usage

```bash
git clone "https://github.com/ai-techsystems/dnnCompiler/"
```
#### Then depending upon your workflow, if you want to do a top level `make`

```bash
cd dnnCompiler
python buildDocker.py
```
#### If you want to run `make` from swig directory

```bash
cd dnnCompiler/swig
python buildDocker.py
```

## Explicit Usage

#### If you want to know the workflow:

* Go to the base directory
	```bash
	cd dnnCompiler
	```
* You should be able to see the **[Dockerfile](../Dockerfile)**. That has the instruction to create a Ubuntu 18.04 image and download required depedencies on top of that.
* Now to create the image from the **[Dockerfile](../Dockerfile)** run
	
	```bash
	sudo docker build -t dnnc .
	```
* Now the image is created. But need to run the image as container that will execute your code. For that
	
	```bash
	sudo docker run -it dnnc /bin/bash -c "cd /dnnCompiler && make clean && make"
	```
  What this does is, runs the image as container, goes inside dnnCompiler directory, and runs `make` as normally you would.