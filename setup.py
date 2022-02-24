
from setuptools import setup, find_packages

MAJOR = 0
MINOR = 0
MICRO = 1
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

setup(
    name='cvxpygen',
    version=VERSION,
    license='Apache License, Version 2.0',
    description='Code generation with CVXPY',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Maximilian Schaller',
    author_email='mschall@stanford.edu',
    url='https://github.com/cvxgrp/cvxpygen',
    packages=find_packages(),
    python_requires='>=3.6',
    py_modules=['cpg', 'utils'],
    install_requires=[
        'cmake >= 3.5',
        'cvxpy >= 1.1.18',
        'ipykernel >= 6.0.0',
        'jupyter >= 1.0.0',
        'matplotlib >= 3.1.3',
        'pybind11 >= 2.8.0'
    ],
    extras_require={
        'dev': [
            'pytest == 6.2.4',
        ],
    },
)
