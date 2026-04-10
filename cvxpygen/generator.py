"""
Copyright 2026 Maximilian Schaller
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import shutil
import sys
from typing import List, Optional

import cvxpy as cp

from cvxpygen.canonicalizer import Canonicalizer
from cvxpygen.writer import CCodeWriter
from cvxpygen.compiler import PythonModuleCompiler
from cvxpygen.mappings import Configuration


class Generator:
    """
    Orchestrates C code generation for a CVXPY problem.

    The constructor takes solver *configuration* (set once, reusable across
    problems). The :meth:`generate` method takes the problem and output
    location (per-call inputs), so a single instance can be reused.

    Example usage::

        gen = Generator(solver='OSQP')
        gen.generate(problem, code_dir='custom_solver')
    """

    def __init__(
        self,
        solver: Optional[str] = None,
        solver_opts: Optional[dict] = None,
        enable_settings: List[str] = [],
        prefix: str = '',
        gradient: bool = False,
    ) -> None:
        self._solver = solver
        self._solver_opts = solver_opts
        self._enable_settings = enable_settings
        self._prefix = prefix
        self._gradient = gradient

        # Populated during generate()
        self.config: Optional[Configuration] = None
        self.canon = None
        self.canon_gradient = None
        self.canon_solver = None
        self.solver_interface = None
        self.gradient_interface = None

    # ── public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        problem: cp.Problem,
        code_dir: str = 'cpg_code',
        wrapper: bool = True,
    ) -> None:
        """Run the full pipeline: canonicalize → write code → (optionally) compile."""
        sys.stdout.write('Generating code with CVXPYgen ...\n')

        solver, explicit = self._resolve_solver()

        # two-stage gradient (QP solved by conic solver) only applies to non-explicit mode
        gradient_two_stage = (self._gradient and solver != cp.OSQP and not explicit)
        self.config = self._build_config(code_dir, solver, gradient_two_stage, explicit)

        self._run_canonicalization(problem, solver, gradient_two_stage)
        self._setup_folder(code_dir)
        self._run_solver_code_generation(problem, code_dir, explicit)
        self._run_code_writing(problem)

        sys.stdout.write('CVXPYgen finished generating code.\n')

        if wrapper:
            self._run_compilation(code_dir, problem)

    # ── private pipeline stages ───────────────────────────────────────────────

    def _setup_folder(self, code_dir: str) -> None:

        shutil.rmtree(code_dir, ignore_errors=True)
        os.mkdir(code_dir)

        os.makedirs(os.path.join(code_dir, 'c', 'src'))
        os.makedirs(os.path.join(code_dir, 'c', 'include'))
        os.makedirs(os.path.join(code_dir, 'c', 'build'))
        os.makedirs(os.path.join(code_dir, 'cpp', 'src'))
        os.makedirs(os.path.join(code_dir, 'cpp', 'include'))

    def _run_canonicalization(self, problem: cp.Problem, solver: str, gradient_two_stage: bool) -> None:
        canonicalizer = Canonicalizer(
            solver=solver,
            solver_opts=self._solver_opts,
            enable_settings=self._enable_settings,
        )
        if gradient_two_stage:
            (self.canon, self.canon_gradient, self.canon_solver,
             self.solver_interface, self.gradient_interface) = (
                canonicalizer.canonicalize_two_stage(problem)
            )
        else:
            self.canon, self.solver_interface = canonicalizer.canonicalize(problem)
            self.gradient_interface = self.solver_interface
            self.canon_gradient = None
            self.canon_solver = None

    def _run_solver_code_generation(self, problem: cp.Problem, code_dir: str, explicit: int) -> None:
        """Call solver_interface.generate_code() to write solver-specific C files."""
        cvxpygen_dir = os.path.dirname(os.path.realpath(__file__))
        solver_code_dir = os.path.join(code_dir, 'c', 'solver_code')
        osqp_code_dir = os.path.join(code_dir, 'c', 'osqp_code')
        cfg = self.config

        if cfg.gradient_two_stage:
            self.solver_interface.generate_code(
                cfg, code_dir, solver_code_dir, cvxpygen_dir,
                self.canon, False, cfg.prefix,
            )
            self.gradient_interface.generate_code(
                cfg, code_dir, osqp_code_dir, cvxpygen_dir,
                self.canon_gradient, True, f'gradient_{cfg.prefix}',
            )
        else:
            self.solver_interface.generate_code(
                cfg, code_dir, solver_code_dir, cvxpygen_dir,
                self.canon, self._gradient, cfg.prefix,
            )

    def _run_code_writing(self, problem: cp.Problem) -> None:
        writer = CCodeWriter(
            problem=problem,
            configuration=self.config,
            canon=self.canon,
            solver_interface=self.solver_interface,
            canon_gradient=self.canon_gradient,
            canon_solver=self.canon_solver,
            gradient_interface=self.gradient_interface,
        )
        writer.write()

    def _run_compilation(self, code_dir: str, problem: cp.Problem) -> None:
        compiler = PythonModuleCompiler(code_dir=code_dir, problem=problem)
        compiler.compile()
        compiler.register()

    # ── private helpers ───────────────────────────────────────────────────────

    def _resolve_solver(self):
        """Return (solver_name, explicit_flag)."""
        solver = self._solver
        solver_opts = self._solver_opts
        if solver is None:
            return None, 0
        elif solver.lower() == 'explicit':
            if solver_opts and solver_opts.get('dual', False):
                return 'PDAQP', 2
            else:
                return 'PDAQP', 1
        else:
            return solver, 0

    def _build_config(self, code_dir, solver_name, gradient_two_stage, explicit) -> Configuration:
        prefix = self._prefix
        if prefix and not prefix[0].isalpha():
            prefix = f'_{prefix}'
        prefix = f'{prefix}_' if prefix else ''
        return Configuration(
            code_dir, solver_name, prefix, self._gradient, gradient_two_stage, explicit,
        )
