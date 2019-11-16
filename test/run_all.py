import os, sys
import common
import unittest
import importlib

if __name__ == '__main__':
    tests_failed = 0
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
        result = runner.run(suite)
        tests_failed = tests_failed +  int(not result.wasSuccessful())
    exit(tests_failed)

