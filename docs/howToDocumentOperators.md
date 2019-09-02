# How to document operators

This is a tutorial for documenting your operator implementation in C++.
We will be using [Doxygen](http://www.doxygen.nl/index.html) for our documentation purpose.
Install doxygen in your system by following this [tutorial](https://www.youtube.com/watch?v=44Ja2X_fzv4).
Here's how to run doxygen.
```
doxygen doxygenconfig
```
This will create a 'docs' folder outside your local repo folder
Search for 'index.html' in docs/html and run it on your browser.
#### Steps to follow for documentation
1. This is how we to put documentation for the operator class.
    Notice the '!' i the comment box.
    ```cpp
    /*! <Put your operator description here>
        ...
     */
    template <typename T> class <operator> : public baseOperator<T> {
    ...
    };
    ```
2. Here's how you can put formulas in your operator [link](http://www.doxygen.nl/manual/formulas.html).
We will be using MathJax so no need to installing LaTeX in your system.
[You can use this site to help generate LaTex code](https://www.codecogs.com/latex/eqneditor.php).
    ```cpp
    /*! \f$ \max (0,\min(1,alpha*x+beta)) \f$
     */
     template <typename T> class HardSigmoid : public baseOperator<T> {
    ```
3. You can implement all your member functions and protected attributes
 Here's a full [manual](http://www.doxygen.nl/manual/docblocks.html#cppblock) for documentation using doxygen.
 I will be giving quick examples to document attributes and member functions.
 Notice the '!<' i the comment box.
 Attributes-
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

You can look at include/operators/InstanceNormalization.h for a full example.
You might want to delete the docs folder outside your local repo after work.
