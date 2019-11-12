# how to run this script: python setup.py bdist_wheel
# hot to test install: python -m pip install ~/dnnc/master/dnnCompiler/dist/dnnc-0.1-py3-none-any.whl --root pip_install_test
# reference: https://dzone.com/articles/executable-package-pip-install

import os, sys, glob
import shutil, errno
import setuptools

NAME='deepC'
VERSION=0.12

long_description = ""
with open("README.md", "r") as fh:
    long_description = fh.read()

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

packages = setuptools.find_packages()

tests_require = []
tests_require.append('unittest')
install_requires = []
install_requires.extend([
    'numpy>=1.16.1',
    'onnx==1.5.0',
])

setuptools.setup(
    name=NAME,
    version=VERSION,
    description="DeepC: Deep Neural Network Compiler",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=packages,
    include_package_data=True,
    package_data={'':['_dnnc.so'] +
        source_files('include') +
        source_files('packages')
        },
    install_requires=install_requires,
    tests_require=tests_require,
    author='Rohit Sharma et. al.',
    author_email='rohit@find-domain-name-on-github.com',
    url='https://github.com/ai-techsystems/dnnCompiler',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
    ],
    python_requires='>=3.5',
    entry_points={
        'console_scripts': [
            'onnx-cpp = deepC.scripts.onnx2cpp:main',
            'compile-onnx = deepC.scripts.onnx2exe:main',
        ]
    },
    dependency_links=[]
)
