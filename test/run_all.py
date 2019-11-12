import os, sys
import common
import unittest
import importlib

if __name__ == '__main__':
    for folder in ['swig', 'parser', 'compiler'] :

        print("\nRunning tests in ===|" + folder + "|===")
        loader = unittest.TestLoader()
        tests = []

        pkg = importlib.import_module(folder)
        pkg.load_tests(loader,tests)
        if ( len(tests) == 0 ):
            continue;

        suite = unittest.TestSuite(tests)
        runner = unittest.TextTestRunner(verbosity=0)
        runner.run(suite)
    exit(0)
