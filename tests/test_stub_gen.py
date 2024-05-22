import ast
import itertools
import logging
from io import StringIO
from tempfile import TemporaryDirectory
from unittest import TestCase

import cvxpy as cp
import test_E2E_LP
import test_E2E_QP
import test_E2E_SOCP

from cvxpygen.cpg import (
    get_configuration,
    get_constraint_info,
    get_dual_variable_info,
    get_interface_class,
    get_parameter_info,
    get_variable_info,
    handle_sparsity,
)
from cvxpygen.utils import write_interface


class test_stub_gen(TestCase):
    def setUp(self) -> None:
        self.all_problems = itertools.chain(
            test_E2E_LP.name_to_prob.items(),
            test_E2E_QP.name_to_prob.items(),
            test_E2E_SOCP.name_to_prob.items(),
        )
        self.tempdir = TemporaryDirectory()
        return super().setUp()

    def tearDown(self) -> None:
        self.tempdir.cleanup()
        return super().tearDown()
    
    def get_codegen_context(self, problem: cp.Problem):
        # problem data
        data, solving_chain, inverse_data = problem.get_problem_data(
            solver=None,
        )
        param_prob = data['param_prob']
        solver_name = solving_chain.solver.name()
        interface_class, cvxpy_interface_class = get_interface_class(solver_name)

        # configuration
        configuration = get_configuration(self.tempdir, solver_name, False, "")

        # cone problems check
        if hasattr(param_prob, 'cone_dims'):
            cone_dims = param_prob.cone_dims
            interface_class.check_unsupported_cones(cone_dims)

        handle_sparsity(param_prob)

        solver_interface = interface_class(data, param_prob, [])  # noqa
        variable_info = get_variable_info(problem, inverse_data)
        dual_variable_info = get_dual_variable_info(inverse_data, solver_interface, cvxpy_interface_class)
        parameter_info = get_parameter_info(param_prob)
        constraint_info = get_constraint_info(solver_interface)
        return dict(
            configuration=configuration,
            solver_interface=solver_interface,
            variable_info=variable_info,
            dual_variable_info=dual_variable_info,
            parameter_info=parameter_info,
        )

    def test_stub_valid(self):
        for name, problem in self.all_problems:
            with StringIO() as f:
                write_interface(f=f, **self.get_codegen_context(problem))
                try:
                    ast.parse(f.read())
                except SyntaxError:
                    logging.exception(f"Generated stub file for problem {name} has incoorect syntax")
                    raise