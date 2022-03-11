
from setuptools import setup, find_packages

MAJOR = 0
MINOR = 1
MICRO = 4
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def readme():
    with open('README.md') as f:
        content = f.read()
    return content[:content.find('## Tests')]


setup(
    name='cvxpygen',
    version=VERSION,
    license='Apache License, Version 2.0',
    description='Code generation with CVXPY',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Maximilian Schaller, '
           'Goran Banjac, '
           'Bartolomeo Stellato, '
           'Steven Diamond, '
           'Akshay Agrawal, '
           'Stephen Boyd',
    author_email='mschall@stanford.edu, '
                 'goranbanjac1989@gmail.com, '
                 'bstellato@princeton.edu, '
                 'diamond@cs.stanford.edu, '
                 'akshayka@cs.stanford.edu, '
                 'boyd@stanford.edu',
    url='https://github.com/cvxgrp/cvxpygen',
    packages=find_packages(),
    python_requires='>=3.6',
    py_modules=['cpg', 'utils'],
    include_package_data=True,
    install_requires=[
        'cmake >= 3.5',
        'cvxpy >= 1.1.18',
        'pybind11 >= 2.8.0',
        'osqp >= 0.6.2',
        'scipy >= 1.1.0',
        'numpy >= 1.15',
    ],
    extras_require={
        'dev': [
            'pytest == 6.2.4',
        ],
    },
)
