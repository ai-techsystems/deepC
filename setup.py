# how to run this script: python setup.py bdist_wheel
# how to test install: python -m pip install ~/dnnc/master/deepC/dist/dnnc-0.1-py3-none-any.whl --root pip_install_test

import os, sys, glob
import shutil, errno, subprocess, multiprocessing
import setuptools


NAME='deepC'
VERSION=0.13

long_description = ""
with open("README.md", "r") as fh:
    long_description = fh.read()

# to tag the whl file with platform
class binaryDist(setuptools.dist.Distribution):
    def is_pure(self):
        return False;
    def has_ext_modules(self):
        return True;

from setuptools.command.build_ext import build_ext as buildext
class make_build(setuptools.Command):
    
    def initialize_options(self):
        self.jobs = multiprocessing.cpu_count()

    def finalize_options(self):
        self.jobs = int(self.jobs)

    def run(self):
        cmd = "make CC=g++ SRC -j " + str(self.jobs)
        subprocess.call(cmd, shell=True)
        cmd = "make CC=g++ all "
        subprocess.call(cmd, shell=True)

class build_ext(buildext):
    def run(self):
        self.run_command('make_build')


#create the links to src dir inside deepC for proper installation.
def link_dir(dir_name):
  try:
    os.symlink(os.path.abspath(os.path.join(os.path.dirname(__file__),
               dir_name)),
               os.path.join(NAME, dir_name))
  except OSError as e:
    if e.errno != errno.EEXIST:
       raise e

link_dir('include')
link_dir('packages')

# add source files for model compiler
def source_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

cmdclass = {
    'make_build': make_build,
    'build_ext': build_ext,
}

ext_modules = [
    setuptools.Extension(
        name=str(NAME+".dnnc"),
        sources=[
                 str(source_files('include')) +
                 str(source_files('packages'))
                ])
]

packages = setuptools.find_packages()

tests_require = []
tests_require.append('unittest')
install_requires = []
install_requires.extend([
    'onnx==1.5.0',
])

setuptools.setup(
    name=NAME,
    version=VERSION,
    description="DeepC: Deep Neural Network Compiler",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=packages,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
    package_data={'':['_dnnc.so'] +
        source_files('include') +
        source_files('packages')
        },
    install_requires=install_requires,
    tests_require=tests_require,
    author='Rohit Sharma et. al.',
    author_email='rohit@find-domain-name-on-github.com',
    url='https://github.com/ai-techsystems/deepC',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
    ],
    python_requires='>=3.5',
    distclass=binaryDist,
    entry_points={
        'console_scripts': [
            'onnx-cpp = deepC.scripts.onnx2cpp:main',
            'compile-onnx = deepC.scripts.onnx2exe:main',
        ]
    },
    dependency_links=[]
)
