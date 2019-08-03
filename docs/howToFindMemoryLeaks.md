# Guide to find memory leaks

## Using Valgrind

```
% cd dnnCompiler
% make clean
% make DEBUG=Y; # debug symbols.
% valgrind --tool=memcheck --leak-check=yes --show-reachable=yes -\
    -num-callers=20 --track-fds=yes <dnnc-exe> |& tee valrind.log
```

## gcc option lmcheck 
You can ask malloc to check the consistency of dynamic memory by using the mcheck function. This function is a GNU extension, 
declared in mcheck.h.

```
% cd dnnCompiler
% make clean
% make LMCHECK=Y; % instrumentation
% cd <test-dir>
% <dnnc-exe> test.py
```

##  glibc env variable "MALLOC_CHECK_"

Another possibility to check for and guard against bugs in the use of malloc, realloc and free is to set the environment 
variable MALLOC_CHECK_.

```
% cd dnnCompiler
% make clean
% make DEBUG=Y; % instrumentation
% cd <test-dir>
% export MALLOC_CHECK_ 3
% <dnnc-exe> test.py
```

Read More at [Heap Consistency Checking](https://www.gnu.org/software/libc/manual/html_node/Heap-Consistency-Checking.html)
