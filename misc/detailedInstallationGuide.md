# Detailed Installation guide

## Windows

#### Prerequisites:
* anaconda
* cmake
* MinGW

*Why we need this much work? As cmake doesn't work well with protobuf specially in windows.*

#### Step 1:
* Open **anaconda prompt** in **administrator mode** and run the following command:
```
conda install -c anaconda protobuf
```
* Open **powershell/cmd** and try to see protoc (proto compiler) version run the following command:
```
protoc --version
```
It will return something like 3.8.0 (in my case) or 3.9.0 or any other.
* Go to this **[protobuf Github release](https://github.com/protocolbuffers/protobuf/releases)** and download the specific protobuf release as the same version of the proto compiler you have.
* Extract the zip and put the **protobuf folder** anywhere you like, but we will be using the downloaded **folder path** later.

#### Step 2:
* Now look for the directory which is like `YOUR PATH WHERE YOU CLONED PROTOBUF/protobuf/src/google/protobuf`.
Copy this path and paste it in the `CMakeLists.txt` file in the dnncompiler repository under `protobuf_INCLUDE_PATH`. (You can see it in the 1st line)

> This will set up the path for protobuf for cmake

* Now look for the directory which is like `YOUR PATH WHERE YOU CLONED PROTOBUF/protobuf/src`.
Copy this path and paste it in the `CMakeLists.txt` file in the dnncompiler repository under `protobuf_LIBRARY_PATH`. (You can see it in the 2nd line)

> This will set up the proto compiler path for protobuf for cmake


**Now you should be able to use `cmake ..` without error.**

#### Step 3:
* To use `make` command in windows, you need MinGW. **[This stackverflow thread](https://stackoverflow.com/questions/23723364/make-is-not-recognized-as-an-internal-or-external-command-operable-program-or/23734705)** has the answers.
* After installing MinGW, set system path for "C:\MinGW\bin" (Default location for MinGW installation)
* Then use `MinGW32-make` not `make`.

### So after all of these the steps are:

Inside the folder "dnnCompiler/build":

```
cmake ..
MinGW32-make
```

## Known bugs:
* When trying MinGW32-make:

```
Makefile:35: *** missing separator.  Stop.
```

**Possible reason:** 
Make command doesn't support `<4 SPACES>` while using indentation. You have to give `<TAB>` in that place.


## Contribution:

#### If you find anymore issues and solution, please append them to this file.