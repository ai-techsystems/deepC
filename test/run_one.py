import unittest
import os, sys
import common
import swig

if __name__ == '__main__':
    for folder in ['swig'] :

        # add the test name here
        if ( len(sys.argv) > 1 ):
            test = sys.argv[1];
        else:
            test = input("Enter test name (ex. 'MatMul.py'): ")

        folder = os.path.join(os.getcwd(), folder)
        print (folder)

        loader = unittest.TestLoader()
        tests = []
        swig.load_test(loader,test,tests)

        suite = unittest.TestSuite(tests)
        runner = unittest.TextTestRunner(verbosity=0)
        runner.run(suite)
