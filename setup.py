import setuptools

VERSION=0.01a

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
    version=VERSION
    description="Deep Neural Network Compiler",
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
)
