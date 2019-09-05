# How to add new operators

This is a tutorial for adding new operator implementation in C++ using Eigen and Swig for interface to Python. Video explains in more detail how implementation is carried out for each operator.

1. Create header file (.h) in /include/operators _see other files for example_
2. Create test file (.cpp) in /src/operators _see other files for example_
3. Compile and run .cpp file
4. Add your opeprator in dnnc.api file for Python interface
5. Add python test in test/swig/<operatorName>.py _see other files for example_
6. Test operator from Python interface
7. Create pull request on Github

[<img src="https://img.youtube.com/vi/2CITO2SEAfE/maxresdefault.jpg" width="100%">](https://youtu.be/2CITO2SEAfE)
