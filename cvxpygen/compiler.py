"""
Copyright 2026 Maximilian Schaller
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import importlib
from subprocess import call

import cvxpy as cp


class PythonModuleCompiler:
    """
    Compiles the generated C extension module and registers the resulting
    cpg_solve function with the CVXPY problem instance.
    """

    def __init__(self, code_dir: str, problem: cp.Problem) -> None:
        self.code_dir = code_dir
        self.problem = problem

    def compile(self) -> None:
        """Build the C extension via setup.py build_ext --inplace."""
        sys.stdout.write('Compiling python wrapper with CVXPYgen ... \n')
        p_dir = os.getcwd()
        os.chdir(self.code_dir)
        call([sys.executable, 'setup.py', '--quiet', 'build_ext', '--inplace'])
        os.chdir(p_dir)
        sys.stdout.write('CVXPYgen finished compiling python wrapper.\n')

    def register(self, name: str = 'CPG') -> None:
        """Import the compiled module and register cpg_solve with the problem."""
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        module = importlib.import_module(f'{self.code_dir}.cpg_solver')
        cpg_solve = getattr(module, 'cpg_solve')
        self.problem.register_solve(name, cpg_solve)
