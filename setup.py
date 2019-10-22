import setuptools

VERSION=0.1

long_description = ""
with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages()

install_requires = []
setup_requires = []
tests_require = []

tests_require.append('unittest')
install_requires.extend([
    'onnx',
    'numpy',
])

extras_require = {}
setuptools.setup(
    name="dnnc",
    version=VERSION,
    description="Deep Neural Network Compiler",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    author='Rohit Sharma et. al.',
    author_email='rohit@find-domain-name-on-github.com',
    url='https://github.com/ai-techsystems/dnnCompiler',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    entry_points={
        'console_scripts': [
            'onnx-cpp = python.onnx2cpp:main',
            'compile-onnx = python.onnx2exe:main',
        ]
    },
    dependency_links=[
        # Make sure to include the `#egg` portion so the `install_requires` recognizes the package
        'git+ssh://git@github.com/onnx/onnx.git#egg=rel-1.5.0'
    ]
)
