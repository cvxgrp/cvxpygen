
from setuptools import setup, find_packages

MAJOR = 0
MINOR = 5
MICRO = 2
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
    python_requires='>=3.9',
    py_modules=['cpg', 'utils'],
    include_package_data=True,
    install_requires=[
        'cmake >= 3.5',
        'cvxpy >= 1.6.4',
        'pybind11 >= 2.8',
        'osqp >= 1.0.0b3',
        'ecos >= 2.0.14',
        'clarabel >= 0.6.0',
        'scipy >= 1.13.1',
        'numpy >= 1.26.0',
        'qocogen >= 0.1.6',
        'qoco >= 0.1.4'
    ],
    extras_require={
        'dev': [
            'pytest == 6.2.4',
        ],
    },
)
