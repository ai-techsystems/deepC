import os, sys
import unittest

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite();

    test_file = "passingTests.txt";
    if ( os.path.isfile(test_file) == False ):
        print("no test file in ", os.getcwd());
        return suite;

    test_fp = open(test_file, "r");
    print("opened file", test_file)
    for test in test_fp.readlines():
        print("adding test", test)
        loader.loadTestsFromTestCase(test);
    test_fp.close()

    return suite;
