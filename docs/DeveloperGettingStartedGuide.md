# Developer's getting started guide

## Contents

* **[Abstract](#abstract)**

* **[Setting up repository](#setting-up-repository)**

* **[Add new operators](#add-new-operators)**

* **[Why use Eigen](#why-use-eigen)**

* **[Add documentation for the operators](#add-documentation-for-the-operators)**

* **[Add operators in python interface](#add-operators-in-python-interface)**

* **[Add unittests for operator testing](#add-unittests-for-operator-testing)**

* **[Work-Flow (Usage and commands for updating code)](#work-flow)**

* **[Pull latest updates](#pull-latest-updates)**

* **[Create pull request](#create-pull-request)**

## Abstract
#### Our work is divided in 5 parts: 

1. Implementing these **[ONNX operators](https://github.com/onnx/onnx/blob/rel-1.5.0/docs/Operators.md)**.	

	* We are working on **[dnnc-operators](https://github.com/ai-techsystems/dnnc-operators)** repo in C++. Our development status is mentioned **[here](https://github.com/ai-techsystems/dnnc-operators/blob/master/README.md)**

	* You can study [Matmul.h](https://github.com/ai-techsystems/dnnc-operators/blob/master/include/operators/MatMul.h), [Add.h](https://github.com/ai-techsystems/dnnc-operators/blob/master/include/operators/Add.h) and [Threshholdrelu.h](https://github.com/ai-techsystems/dnnc-operators/blob/master/include/operators/ThresholdedRelu.h) to get a demo of what we are trying to achieve.

	* Here is a **[Tutorial](https://dritchie.github.io/csci2240/assignments/eigen_tutorial.pdf)** for eigen library.	

	* Here is more **[indepth discussion of philosophy and features of Eigen](http://downloads.tuxfamily.org/eigen/eigen_CGLibs_Giugno_Pisa_2013.pdf)**

	* Also check out **[Eigen documentation](http://eigen.tuxfamily.org/dox/.)**

2. Test the above **[ONNX operators](https://github.com/onnx/onnx/blob/rel-1.5.0/docs/Operators.md)**.

	* Here are the **[test cases](https://github.com/ai-techsystems/dnnc-operators/tree/master/src/operators)**

	* After making this branch stable it will be merged in the official repo **[dnnCompiler](https://github.com/ai-techsystems/dnnCompiler/tree/operators)**.

3. Add documentation to the operators with the help of **[Doxygen](http://www.doxygen.nl/index.html)**.
	
	* Here is a **[tutorial for Doxygen](https://www.youtube.com/watch?v=44Ja2X_fzv4)**

4. Implement SWIG for python interface.

	* Here is a **[SWIG Tutorial](http://www.swig.org/tutorial.html)**.

	* DNNC [operators](https://github.com/ai-techsystems/dnnc-operators/tree/master/include/operators) and tensors should be implemented for the python interface with **[SWIG](http://www.swig.org/exec.html)**.

	* To understand how we are wrapping operators written in cpp, with python see [usage](#usage) guide below.

	* Check out **[Numpy](https://docs.scipy.org/doc/numpy/reference/)** for the implementation of our tensor and it's simplicity.

5. Test the operators with python unittest.

	* Here is the guide of **[python unittest](https://docs.python.org/3/library/unittest.html)**.

	* For reference go to **[test / swig](https://github.com/ai-techsystems/dnnCompiler/tree/master/test/swig)** and see [MatMul.py](https://github.com/ai-techsystems/dnnCompiler/blob/master/test/swig/MatMul.py), [Exp.py](https://github.com/ai-techsystems/dnnCompiler/blob/master/test/swig/Exp.py) to get a demo.

## Setting up repository

### Forking:	

* Go to **[dnnCompiler](https://github.com/ai-techsystems/dnnCompiler)**
* Click **Fork** to your own repository.
	- This will take 10 sec or so.
	- Now you will be redirected to a copy of **dnnCompiler** under your username
	- And it will be written :	
		> your_username/dnnCompiler	
		> forked from ai-techsystems/dnnCompiler
* Choose active development branch (e.g. `operators`),  click on the **Clone or Download button** and copy the link.
	- It will look like (https://github.com/your_username/dnnCompiler.git)
* Choose active development Go to your terminal and go to any directory under which you want to clone the repo and open terminal.
	- Paste the link you copied after typing `git clone `. It will look like this :	
		```console
		git clone --single-branch -b operators https://github.com/your_username/dnnCompiler.git
		```

### Changing branch

- Go inside the repo	
	```console
	cd dnnCompiler
	```
* Now you will be inside the repository.

	- Check how many branches this repository has.
		```console
		git branch -r
		``` 
		- You will see something like:
			```bash
			origin/HEAD -> origin/master
			origin/master
			origin/operators
			```
	- Check on which branch you are currently on
		```console
		git branch
		```
		- You will see something like:
			```bash
			* master
			operators
			```
		- The `*` shows your current branch.
	- Change the branch to the operators as all the newer development is done on that branch.
		```console
		git checkout operators
		```
		- You will see something like
			```bash
			Switched to a new branch 'operators'
			Branch 'operators' set up to track remote branch 'operators' from 'origin'.
			```
	- Now if you do 
		```console
		git branch
		```
		- You will see:
			```bash
			master
			* operators
			```
		- Now you are on operators branch.

#### Add synchronization steps to get latest updates from `AITS dnnCompiler`

* Now you will have to setup your repo so that it can sync new updates from the original **dnnCompiler** repo under **AITS**. As there will be other developers working on that. To do that you have to set **dnnCompiler** repo of **AITS** as an **upstream**.
	- Add a remote upstream of the original **dnnCompiler** (You only need to do this upstream setup once! But **fetching** and **merging** should be done everytime)
	
		```console
		git remote add upstream https://github.com/ai-techsystems/dnnCompiler
		```
	- This will add original **dnnCompiler** as upstream.

#### Update code	

* Now you are set to change and update your code.

## Add new operators

This is a tutorial for adding new operator implementation in C++ using Eigen and Swig for interface to Python. Video explains in more detail how implementation is carried out for each operator.

1. Create header file (.h) in **[include / operators](https://github.com/ai-techsystems/dnnCompiler/tree/master/include/operators)** (_see other files for example_)

2. Create test file (.cpp) in **[src / operators](https://github.com/ai-techsystems/dnnCompiler/tree/master/src/operators)** (_see other files for example_)

3. Compile and run .cpp file.	

For reference look at this tutorial, and just watch till **8:33 minutes**, as after that he shows how to add them in swig, but the process of adding the operators in the swig has changed to a much easier convenient way.

[<img src="https://img.youtube.com/vi/2CITO2SEAfE/maxresdefault.jpg" width="100%">](https://youtu.be/2CITO2SEAfE)

---

## Why use Eigen

#### Below is a snippet code only for **2D**. One uses Eigen, and another just uses loop.

<details>
<summary>With Eigen</summary>

```cpp
tensor<T> eigen_compute(tensor<T> &a, tensor<T> &b){
		
		if (a.shape() != b.shape())
			throw std::invalid_argument(
					"tensor dimenions not appropriate for Div operator.");
		if (a.rank() == 2 && b.rank() == 2) {
		
			tensor<T> result(a.shape()[0], b.shape()[1]);

			DNNC_EIGEN_MATRIX(eigenMatrixA, a);
			DNNC_EIGEN_MATRIX(eigenMatrixB, b);

			Matrix<T, Dynamic, Dynamic, RowMajor> eResult =
					eigenMatrixA.array() / eigenMatrixB.array();

			result.load(eResult.data());
			return result;
		}
		return tensor<T>();
	}

```
</details>

<details>
<summary>Without Eigen</summary>

```cpp
tensor<T> without_eigen_compute(tensor<T> &a, tensor<T> &b) {
		if (a.shape() != b.shape())
			throw std::invalid_argument(
					"tensor dimenions not appropriate for Div operator.");

		tensor<T> result(a.shape(), a.name());
		for (size_t i = 0; i < a.length(); i++)
			result[i] = a[i] / b[i];

		return result;
	}
```
</details>

### Now let's see the performance

<details>
<summary>Random array generation funtion</summary>

```cpp
void generate_random(float* a,int size){
	srand(time(0)); 
	int i;
	for (i=0;i<size;i++){
		a[i]=rand();
	}
}
```
</details>

#### Going with relatively small matrix
<details>
<summary>Small matrix input</summary>

```cpp
int main() {
	float d1[100],d2[100];
	generate_random(d1,100);
	generate_random(d2,100);

	tensor<float> a(10, 10);
	a.load(d1);
	tensor<float> b(10, 10);
	b.load(d2);
	Div<float> m("localOpName");

	clock_t t;
	
	t = clock();
	auto result_1 = m.without_eigen_compute(a, b);
	t = clock() - t;
	double time_taken_1 = ((double)t)/CLOCKS_PER_SEC;
	
	t = clock();
	auto result_2 = m.eigen_compute(a, b);
	t = clock() - t;
	double time_taken_2 = ((double)t)/CLOCKS_PER_SEC;
	
	std::cout << time_taken_1 << " seconds took without eigen " << std::endl;
	std::cout << time_taken_2 << " seconds took with eigen" << std::endl;

	return 0;
}
```
</details>

##### Here Eigen is **~10x** faster than looping
#### Going with relatively large matrix
<details>
<summary>Large matrix input</summary>

```cpp
int main() {
	float d1[1000000],d2[1000000];
	generate_random(d1,1000000);
	generate_random(d2,1000000);

	tensor<float> a(1000, 1000);
	a.load(d1);
	tensor<float> b(1000, 1000);
	b.load(d2);
	Div<float> m("localOpName");

	clock_t t;
	
	t = clock();
	auto result_1 = m.without_eigen_compute(a, b);
	t = clock() - t;
	double time_taken_1 = ((double)t)/CLOCKS_PER_SEC;
	
	t = clock();
	auto result_2 = m.eigen_compute(a, b);
	t = clock() - t;
	double time_taken_2 = ((double)t)/CLOCKS_PER_SEC;
	
	std::cout << time_taken_1 << " seconds took without eigen " << std::endl;
	std::cout << time_taken_2 << " seconds took with eigen" << std::endl;

	return 0;
```
</details>

##### Here Eigen is **~2x** faster than looping

#### Eigen is excellent in memory handling and efficiency, rather than us looping through the tensor.


## Add documentation for the operators

This is a tutorial for documenting your operator implementation in C++.
We will be using [Doxygen](http://www.doxygen.nl/index.html) for our documentation purpose.
Install doxygen in your system by following this [tutorial](https://www.youtube.com/watch?v=44Ja2X_fzv4).
Here's how to run doxygen.

```console
doxygen doxygen.cfg
```

This will create a 'docs' folder outside your local repo folder
Search for 'index.html' in docs/html and run it on your browser.
#### Steps to follow for documentation
1. This is how we to put documentation for the operator class. Notice the '!' i the comment box.
	
	```cpp
	/*! <Put your operator description here>
			...
	 */
	template <typename T> class <operator> : public baseOperator<T> {
	...
	};
	```
	
2. Here's how you can put formulas in your operator [link](http://www.doxygen.nl/manual/formulas.html).
We will be using MathJax so no need to installing LaTeX in your system. [You can use this site to help generate LaTex code](https://www.codecogs.com/latex/eqneditor.php).
	
	```cpp
	/*! \f$ \max (0,\min(1,alpha*x+beta)) \f$
	 */
	 template <typename T> class HardSigmoid : public baseOperator<T> {
	```

3. You can implement all your member functions and protected attributes
 Here's a full [manual](http://www.doxygen.nl/manual/docblocks.html#cppblock) for documentation using doxygen.
 I will be giving quick examples to document attributes and member functions.
 Notice the '!<' i the comment box. Attributes-
		
	```cpp
	float epsilon = 1e-05; /*!< In case variance goes to zero and to avoid division by zero. */
	```
	
	Member functions- documenting the inputs and outputs
	
	```cpp
	tensor<T> compute(tensor<T> &input /*!< [float,double]: ND tensor of shape ( NxCxD1xD2…Dk ).*/){
		...
	}
	/*!<
	\return The output tensor of the same shape as input.
	*/
	```
	
	Note that this is only for class members. For documenting non-members and static members see point 1

You can look at **[include / operators / InstanceNormalization.h](../include/operators/InstanceNormalization.h)** for a full example.
You might want to delete the docs folder outside your local repo after work.



## Add operators in python interface


#### Operator Interface Automation:

We are currently automating the `dnnc.i` and `dnnc_api.cpp` file, to save you some time, and repeatative works.
In the process of automation we will be needing two files, 

* **[swig / dnnc.api](../swig/dnnc.api)** (pseudo cpp/python file which you will be adding your opearators in)
* **[swig / op_gen.py](../swig/op_gen.py)** (which will generate `dnnc_swig_externs.h` and `dnnc_api.cpp` file from the above `dnnc.api` file)

#### op_gen.py is integrated in Makefile, so running make at the top-level or in [swig /](../swig) will generate required files.

* So here is the **[Guide](#guide)** to follow while writing **dnnc.api**, there are some examples shown below.

* After adding your operator inside dnnc.api, run make clean to clean previous compilations
	```console
	make clean
	```
* Then run make again, to compile it with your addition.
	```console
	make
	```
	- This will generate the required swig files and compile them so that we can use them from pyhton interface too. 

##### Explicit Usage of automation:
```console
python op_gen.py
```

#### I have tried to pick and write some diverse examples below to give you an idea how the `dnnc.api` file will look like.

---
##### MatMul and Add operators has input and output of same dtypes
```cpp

tensor<output> matmul(tensor<input> &a, tensor<input> &b) {
	MatMul<input> op;
	return op.compute(a, b);
	dtype = {
		"float" : "float",
		"int" : "int"
	}
}

tensor<output> add(tensor<input> &a, tensor<input> &b) {
	Add<input> op;
	return op.compute(a, b);
	dtype = {
		"float" : "float",
		"int" : "int"
	}
}
```
---
##### DequantizeLinear takes b tensor as float, and it's fixed, so declared the b tensor as `<float>`, instead of `<input>`

```cpp
tensor<output> dequantize_linear(tensor<input> &a, tensor<float> &b, tensor<input> &c) {
	DequantizeLinear<input> op;
	return op.compute(a, b, c);
	dtype = {
		"float" : "int"
	}
}
```
---
##### Elu has fixed input and output, `<float>` only, either you can write `<float>` instead of `<input>` and `<output>`, or specify dtype, both works.

```cpp
tensor<output> elu(tensor<input> &a, float alpha = 1.0) {
	Elu<input> op("localOpName", alpha);
	return op.compute(a);
	dtype = {
		"float" : "float"
	}
}
```
---
##### Equal only outputs in `<bool>`

```cpp
tensor<output> equal(tensor<input> &a, tensor<input> &b) {
	Equal<input> op;
	return op.compute(a, b);
	dtype = {
		"bool" : "bool",
		"bool" : "int",
		"bool" : "float"
	}
}
```
---
##### This should give you a rough idea how the dnnc.api file will look like. If you like to see the whole picture, see below

<details>
<summary>Example</summary>

```cpp


tensor<output> matmul(tensor<input> &a, tensor<input> &b) {
	MatMul<input> op;
	return op.compute(a, b);
	dtype = {
		"float" : "float",
		"int" : "int"
	}
}

tensor<output> add(tensor<input> &a, tensor<input> &b) {
	Add<input> op;
	return op.compute(a, b);
	dtype = {
		"float" : "float",
		"int" : "int"
	}
}

tensor<output> dequantize_linear(tensor<input> &a, tensor<float> &b, tensor<input> &c) {
	DequantizeLinear<input> op;
	return op.compute(a, b, c);
	dtype = {
		"float" : "int"
	}
}

tensor<output> elu(tensor<input> &a, float alpha = 1.0) {
	Elu<input> op("localOpName", alpha);
	return op.compute(a);
	dtype = {
		"float" : "float"
	}
}

tensor<output> equal(tensor<input> &a, tensor<input> &b) {
	Equal<input> op;
	return op.compute(a, b);
	dtype = {
		"bool" : "float",
		"bool" : "int",
		"bool" : "bool"
	}
}
```
</details>

### Guide :
* Everything except **dtype** block is a cpp block, and **dtype** is a python dictionary which contains all kinds of input output datatype combination possible for the operators:
	```python
	dtype = {
		"output1" : "input1",
		"output2" : "input2",
		"output2" : "input1",
		...
	}
	```
* Everything inside `dnnc.api` is **whitespace** and **newline** sensitive, so try to keep the structure similar.
* Make sure to add a blank line between 2 operators.
* Don't leave any blank lines inside operators' functions.
* Don't leave more than one blank line anywhere.
* Use comment syntax (`/*` or `*/`) in the same line as the code. See the example below
	
	```cpp
	tensor<output> less_equal(tensor<input> &a, tensor<input> &b) {
		LessEqual<input> op;
		return op.compute(a, b);
		dtype = {
			"bool" : "bool",
			"bool" : "int",
			"bool" : "float",
			"bool" : "double"
		}
	}

	/* The below operators need to change accroding to above operators */

	tensor<float> thresholded_relu(tensor<float> &a) {
		ThresholdedRelu<float> op;
		return op.compute(a);
	}

	/* tensor<output> logical_xor(tensor<input> &a, tensor<input> &b) {
		Xor<input> op;
		return op.compute(a, b);
		dtype = {
			"bool" : "double",
			"bool" : "float",
			"bool" : "bool",
			"bool" : "int"
		}
	} */

	tensor<output> transpose(tensor<input> &a) {
		Transpose<input> op;
		return op.compute(a);
		dtype = {
			"double" : "double",
			"float" : "float",
			"int" : "int",
			"bool" : "bool"
		}
	}
	```


## Add unittests for operator testing

#### Test Case Automation:
##### We have created 2 files which will keep track of our operators, which passes or fails the test cases:
* **[test / swig / passingTests.txt](../test/swig/passingTests.txt)**
* **[test / swig / failingTests.txt](../test/swig/failingTests.txt)**

##### We have created 2 python scripts to run the tests at ease:
* **[test / run_all.py](../test/run_all.py)** (It will run all the testcases mentioned on the `passingTests.txt`)
* **[test / run_one.py](../test/run_one.py)** (It will run only one testcase opearator at a time)

##### Why do we need them?
In a distant future in dnnCompiler development, we will come at a point, when pull request can only be done when the make command builds successfully. Currently in top level make, the `run_all.py` is already implemented. You can check that with command

```console
make TEST
```
This will help us to get rid of the tension when it comes to merging a update, whether the update will break the functionality or not.

#### How to add your unittest

* Go to **[test / swig /](https://github.com/ai-techsystems/dnnCompiler/tree/operators/test/swig)**

* Here are all the **[python unittest](https://docs.python.org/3/library/unittest.html)** files. Go add yours too by looking at others as demo.

* you can run them by (if your operator name is MatMul.py)
	
	```console
	python MatMul.py
	```

## Work-FLow

* For adding new opeartors you have add your code in as mentioned in **[Add new operators](#add-new-operators)**
	
	* **[include / operators /](https://github.com/ai-techsystems/dnnCompiler/tree/master/include/operators)** (The .h file)
	
	* **[src / operators /](https://github.com/ai-techsystems/dnnCompiler/tree/master/src/operators)** (The .cpp file)

* Now to wrap them in python interface go to **[swig / ](https://github.com/ai-techsystems/dnnCompiler/tree/operators/swig)** folder

* Look for a file named **[dnnc.api](https://github.com/ai-techsystems/dnnCompiler/blob/master/swig/dnnc.api)**

* It's a pseudo (cpp/python) code. There are some things which you need to remember before adding your operator in this file. Head towards **[guide](#guide-)** section to learn how to add your operator inside **[dnnc.api](https://github.com/ai-techsystems/dnnCompiler/blob/master/swig/dnnc.api)** file.

* After that, run **make** command followed by a **make clean** in the same directory.
	
	```console
	make clean
	```

	```console
	make
	```

* If everything went fine, go to **[test / swig /](https://github.com/ai-techsystems/dnnCompiler/tree/operators/test/swig)**

* Here are all the **[python unittest](https://docs.python.org/3/library/unittest.html)** files. Go add yours too by looking at others as demo.

* To test your **unittest** file, there are 2 ways.
	- `Option 1`: Inside **[test / swig /](https://github.com/ai-techsystems/dnnCompiler/tree/operators/test/swig)** (If your operator is **Reciprocal.py**) run the following command:
		```console
		python Reciprocal.py
		```
	
	- `Option 2`: Inside **[test /](https://github.com/ai-techsystems/dnnCompiler/tree/operators/test/)** (If your operator is **Reciprocal.py**) run the following command:
		```console
		python run_one.py Reciprocal.py
		```
* If your operator's unittest was successful, go to **[test / swig / passingTests.txt](https://github.com/ai-techsystems/dnnCompiler/blob/master/test/swig/passingTests.txt)** and append your operator's unittest name there, in a new line.

* If your operator's unittest was unsuccessful, go to **[test / swig / failingTests.txt](https://github.com/ai-techsystems/dnnCompiler/blob/master/test/swig/failingTests.txt)** and append your operator's unittest name there, in a new line.

* After that go to **[test /](https://github.com/ai-techsystems/dnnCompiler/tree/operators/test/)** and run the following command, which will run all the passing tests listed in the **[test / swig / passingTests.txt](https://github.com/ai-techsystems/dnnCompiler/blob/master/test/swig/passingTests.txt)**. If you added your operator there, your unittest will run too. Command:
	```console
	python run_all.py
	```
	- If everything goes well, you have successfully added your operator and integrated it with python.

## Pull latest updates

* If you don't want to keep any changes you made, and just pull the upstream, use this:

	```console
	git fetch upstream
	```
*	Followed by

	```console
	git reset --hard upstream/operators
	```
* To read more, go to this **[StackOverflow link](https://stackoverflow.com/questions/1125968/how-do-i-force-git-pull-to-overwrite-local-files)**.

* If you want to keep your work, and pull update from upstream, follow below.

#### Backing up uncommitted work:	
* First back up your current work:
	```console
	git stash
	```

#### Pull latest updates from `AITS dnnCompiler`

* Remember you added Upstream while setting up your repo, we will be using that now. If you haven't done that yet, go to [this section](#add-synchronization-steps-to-get-latest-updates-from-aits-dnncompiler)
	- To fetch the latest updates from the **dnnCompiler** repo from **AITS**, use
		```console
		git fetch upstream
		```
		- You will see something like
			```bash
			From https://github.com/ai-techsystems/dnnCompiler
			* [new branch]			master		 -> upstream/master
			* [new branch]			operators	-> upstream/operators
			```
* Now based on which branch you are currently on, you have to merge `origin/branch_name` with `upstream/branch_name`. **Origin** means your forked local repo, and **Upstream** means the original repo from **AITS** here.

#### Merging the update from upstream	

* If you followed all previous steps, you will be currently on `origin/operators` branch.
* Now we will merge the upstream operators branch.
	
	```console
	git merge upstream/operators
	```
	
	- There can be 2 possibilities:
		
		1. If you are already upto date, you will see something like this.

			```bash
			Already up to date.
			```
		
		2. If there was some updates from upstream repo, you will see somthing like this.
			
			```bash
			Updating 5e128bb..daa1019
			Fast-forward
			include/operators/Reciprocal.h | 19 +++++++++++++++++--
			src/operators/Reciprocal.cpp   | 13 +++++++++++++
			swig/dnnc.api                  | 12 ++++++++++++
			swig/dnnc_api.cpp              | 15 +++++++++++++++
			swig/dnnc_swig_externs.h       |	4 ++++
			5 files changed, 61 insertions(+), 2 deletions(-)
			```

	- Else every update will be merged from operators branch.
* We will not merge the `upstream/master` as it is not required, but if you want to do that too, follow the steps below.
	- First change to master branch	
		```console
		git checkout master
		```	
	- If you did `git fetch` previously, don't bother to do that again, or do a `git fetch upstream`.	
	- Then merge master branch	
		```console	
		git merge upstream/master
		```	
	- Now your master branch will also be updated, before you forget, go back to `operators` branch, as we will modify that only.	
		```console	
		git checkout operators
		```	
			- Now both of your branches are synchronized with the latest update from **AITS dnnCompiler** repo.	
* Now your repo is synchronized with the latest update from upstream. Now sync your forked repo with upstream. Till now you synced your local repo with upstream, but not published it in your github forked repo, to do that simply type
	```console
	git push
	```
* Now everything is in sync.
#### Get uncomitted code back
* Now get back the local changes you saved earlier with `git stash` command.
	```console
	git stash pop
	```
* Here 2 things can happen:
	- Either it will merge your saved work with recent update automatically, which will say like this, and doesn't need attention:

		```bash
		On branch operators
		Your branch is ahead of 'origin/operators' by 25 commits.
		  (use "git push" to publish your local commits)

		Changes not staged for commit:
		  (use "git add <file>..." to update what will be committed)
		  (use "git restore <file>..." to discard changes in working directory)
		        modified:   docs/DeveloperGettingStartedGuide.md

		no changes added to commit (use "git add" and/or "git commit -a")
		```
	- Or it will show conflict while merge like this, this **needs your attention**.

		```bash
		Auto-merging swig/dnnc_swig_externs.h
		CONFLICT (content): Merge conflict in swig/dnnc_swig_externs.h
		Auto-merging swig/dnnc_api.cpp
		CONFLICT (content): Merge conflict in swig/dnnc_api.cpp
		Auto-merging swig/dnnc.api
		CONFLICT (content): Merge conflict in swig/dnnc.api
		Auto-merging include/operators/Or.h
		```
#### Resolve merge conflict issue

In the previous step, if you hava faced the **merge conflict**, this is what you need to do:

* See the above message, says you have conflict in 3 files. So if you open these 3 files, you will see something like this:
	
	```cpp
	<<<<<<< Updated upstream
	#include "operators/Mod.h"
	#include "operators/Mul.h"
	#include "operators/Neg.h"
	#include "operators/Not.h"
	#include "operators/NotEqual.h"
	#include "operators/Or.h"
	#include "operators/Pow.h"
	=======
	#include "operators/Reciprocal.h"
	>>>>>>> Stashed changes
	```
	- What this means is, 
		
		```cpp
		<<<<<<< Updated upstream
		// the code you fetched from the upstream/or remote repository
		=======
		// the code you wrote earlier and stashed, which now creates
		// merge conflict upon doing `git stash pop`
		>>>>>>> Stashed changes
		```
		
		- So, change what necessary, and delete those symbols, git creates this to show you where the conflict is. So after removing conflict, the snippet will look like this:

		```cpp
		#include "operators/Mod.h"
		#include "operators/Mul.h"
		#include "operators/Neg.h"
		#include "operators/Not.h"
		#include "operators/NotEqual.h"
		#include "operators/Or.h"
		#include "operators/Pow.h"
		#include "operators/Reciprocal.h"
		```
* By doing this procedure to every file, which is showing conflict, you can manage to resolve the conflict.

#### Push your modified code to your forked repo in GitHub
* Now you will have your uncommitted work over the synced repo, just as you wanted. Do more modifications if required. And then do the usual commands to push your changes in your forked repo.
	```console
	git add .
	git commit -m "commit message"
	git push
	```
* This will update your forked repo with your additions, Now if you want them to be added in the **AITS dnnCompiler** repo, see the Pull request sectionbelow.

## Create pull request

* If you followed previous instructions, you will have a forked repo which has the latest update from **AITS dnnCompiler** with your further modifications.

* Now go to your forked repo in GitHub in your browser.

* Change branch from master to operators.

* You will see something like
  > Your branch is ahead of n commits of ai-techsystems:operators.

* Click on **pull request**

* You will be taken to a new page where in the top you can see
  > merge [operator branch] [aits dnnCompiler] <-- [operator branch] [your_username dnnCompiler]

* You will also be able to see the changes you made in the comparison of files below that.

* Now click on **create pull request**

* It's done!  
