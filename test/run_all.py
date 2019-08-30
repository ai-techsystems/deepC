import unittest
import os, sys
import common
import swig

if __name__ == '__main__':
    for folder in ['swig'] :
        folder = os.path.join(os.getcwd(), folder)
        #module_names = load_tests(folder)
        #os.chdir(folder);
        print (folder)

        loader = unittest.TestLoader()
        tests = []
        swig.load_tests(loader,tests)
        # tests = loader.discover(folder, pattern='*.py', top_level_dir=os.getcwd())
        # # tests = load_tests(loader,tests, pattern='*.py')
        # print(tests)
        # #suites = [unittest.defaultTestLoader.loadTestsFromName(mname) for mname in module_names]

        # #testSuite = unittest.TestSuite(suites)
        # runner = unittest.TextTestRunner(verbosity=0)
        # runner.run(tests)

        suite = unittest.TestSuite(tests)
        runner = unittest.TextTestRunner(verbosity=0)
        runner.run(suite)