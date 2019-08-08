# Guide to profile your code using gprof
```
% cd dnnCompiler
% make PROF=Y; # profiler build for profiler data
% cd <test>
% <dnnc-exe> test.py
% gprof `which <dnnc-exe>` gmon.out |& tee gprof.rpt
```
