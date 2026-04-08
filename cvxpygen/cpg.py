"""
Copyright 2022-2026 Maximilian Schaller
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

from cvxpygen.generator import Generator


def generate_code(problem, code_dir='cpg_code', solver=None, solver_opts=None,
                  enable_settings=[], prefix='', gradient=False, wrapper=True):
    """
    Generate C code to solve a CVXPY problem.

    Backward-compatible entry point. Delegates to CodeGenerator.
    """
    Generator(
        solver=solver,
        solver_opts=solver_opts,
        enable_settings=enable_settings,
        prefix=prefix,
        gradient=gradient,
    ).generate(problem, code_dir=code_dir, wrapper=wrapper)
