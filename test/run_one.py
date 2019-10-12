import unittest
import os, sys
import common
import importlib

if __name__ == '__main__':
    for folder in ['swig', 'parser'] :

        # add the test name here
        if ( len(sys.argv) > 1 ):
            test = sys.argv[1];
        else:
            test = input("Enter test name (ex. 'MatMul.py'): ")

        loader = unittest.TestLoader()
        tests = []
        pkg = importlib.import_module(folder)

        pkg.load_test(loader,test,tests)
        if ( len(tests) == 0 ):
            continue;

        suite = unittest.TestSuite(tests)
        runner = unittest.TextTestRunner(verbosity=0)
        runner.run(suite)
