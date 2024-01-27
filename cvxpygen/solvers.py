import os
import sys
import shutil
import warnings
from abc import ABC, abstractmethod
from platform import system

import numpy as np
import scipy as sp

from cvxpygen.utils import read_write_file, write_struct_prot, write_struct_def, \
    write_vec_prot, write_vec_def, multiple_replace, cut_from_expr
from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, ConstraintInfo, AffineMap, \
    ParameterCanon, WorkspacePointerInfo, UpdatePendingLogic, ParameterUpdateLogic

from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL


def get_interface_class(solver_name: str) -> "SolverInterface":
    if system() == 'Windows' and solver_name.upper() == 'CLARABEL':
        raise ValueError(f'Clarabel solver currently unsupported on Windows.')
    mapping = {
        'OSQP': (OSQPInterface, OSQP),
        'SCS': (SCSInterface, SCS),
        'ECOS': (ECOSInterface, ECOS),
        'CLARABEL': (ClarabelInterface, CLARABEL),
    }
    interface = mapping.get(solver_name.upper(), None)
    if interface is None:
        raise ValueError(f'Unsupported solver: {solver_name}.')
    return interface[0], interface[1]


class SolverInterface(ABC):

    def __init__(self, solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                 indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings):
        self.solver_name = solver_name
        self.n_var = n_var
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.indices_obj = indices_obj
        self.indptr_obj = indptr_obj
        self.shape_obj = shape_obj
        self.indices_constr = indices_constr
        self.indptr_constr = indptr_constr
        self.shape_constr = shape_constr
        self.canon_constants = canon_constants
        self.enable_settings = enable_settings

        self.configure_settings()

    @property
    @abstractmethod
    def canon_p_ids(self):
        pass

    @property
    @abstractmethod
    def canon_p_ids_constr_vec(self):
        pass

    @property
    @abstractmethod
    def stgs_names(self):
        pass

    @property
    @abstractmethod
    def stgs_types(self):
        pass

    @property
    @abstractmethod
    def stgs_defaults(self):
        pass

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return any(variable_info.sym) or any([s == 1 for s in variable_info.sizes])

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return any([s == 1 for s in dual_variable_info.sizes])

    def configure_settings(self) -> None:
        for i, s in enumerate(self.stgs_names):
            if s in self.enable_settings:
                self.stgs_enabled[i] = True
        for s in set(self.enable_settings)-set(self.stgs_names):
            warnings.warn(f'Cannot enable setting {s} for solver {self.solver_name}')

    def get_affine_map(self, p_id, param_prob, constraint_info: ConstraintInfo) -> AffineMap:
        affine_map = AffineMap()

        if p_id == 'P':
            if self.indices_obj is None: # problem is an LP
                return None
            affine_map.mapping = param_prob.reduced_P.reduced_mat
            affine_map.indices = self.indices_obj
            affine_map.shape = (self.n_var, self.n_var)
        elif p_id in ['q', 'c']:
            affine_map.mapping = param_prob.c[:-1]
        elif p_id == 'd':
            affine_map.mapping = param_prob.c[-1]
        elif p_id == 'A':
            affine_map.mapping_rows = constraint_info.mapping_rows_eq[
                constraint_info.mapping_rows_eq < constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_eq, self.n_var)
        elif p_id == 'G':
            affine_map.mapping_rows = constraint_info.mapping_rows_ineq[
                constraint_info.mapping_rows_ineq < constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_ineq, self.n_var)
        elif p_id == 'b':
            affine_map.mapping_rows = constraint_info.mapping_rows_eq[
                constraint_info.mapping_rows_eq >= constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_eq, 1)
        elif p_id == 'h':
            affine_map.mapping_rows = constraint_info.mapping_rows_ineq[
                constraint_info.mapping_rows_ineq >= constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_ineq, 1)
        else:
            raise ValueError(f'Unknown parameter name: {p_id}.')

        if p_id in ['A', 'b']:
            affine_map.indices = self.indices_constr[affine_map.mapping_rows]
        elif p_id in ['G', 'h']:
            affine_map.indices = self.indices_constr[affine_map.mapping_rows] - self.n_eq

        if p_id in ['A', 'G']:
            affine_map.mapping = param_prob.reduced_A.reduced_mat[affine_map.mapping_rows]
            affine_map.sign = -1

        return affine_map
    
    def augment_vector_parameter(self, p_id, vector_parameter):
        return vector_parameter
    
    def get_problem_data_index(self, reduced_mat):
        if reduced_mat.problem_data_index is None:
            return None, None, None
        else:
            indices, indptr, shape = reduced_mat.problem_data_index
            return indices, indptr, shape

    @property
    def stgs_names_enabled(self):
        return [name for name, enabled in zip(self.stgs_names, self.stgs_enabled) if enabled]

    @property
    def stgs_names_to_type(self):
        return {name: typ for name, typ, enabled in zip(self.stgs_names, self.stgs_types, self.stgs_enabled)
                if enabled}

    @property
    def stgs_names_to_default(self):
        return {name: typ for name, typ, enabled in zip(self.stgs_names, self.stgs_defaults, self.stgs_enabled)
                if enabled}

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        pass

    @abstractmethod
    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                      parameter_canon: ParameterCanon) -> None:
        pass

    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        pass

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        pass


class OSQPInterface(SolverInterface):
    solver_name = 'OSQP'
    solver_type = 'quadratic'
    canon_p_ids = ['P', 'q', 'd', 'A', 'l', 'u']
    canon_p_ids_constr_vec = ['l', 'u']
    parameter_update_structure = {
        'PA': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['P', 'A'], '&&', ['P', 'A']),
            function_call = 'osqp_update_data_mat(&solver, {prefix}Canon_Params.P->x, 0, 0, {prefix}Canon_Params.A->x, 0, 0)',
        ),
        'P': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['P']),
            function_call = 'osqp_update_data_mat(&solver, {prefix}Canon_Params.P->x, 0, 0, OSQP_NULL, 0, 0)',
        ),
        'A': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['A']),
            function_call = 'osqp_update_data_mat(&solver, OSQP_NULL, 0, 0, {prefix}Canon_Params.A->x, 0, 0)'
        ),
        'qlu': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['q', 'l', 'u'], '&&', ['ql', 'qu', 'lu']),
            function_call = 'osqp_update_data_vec(&solver, {prefix}Canon_Params.q, {prefix}Canon_Params.l, {prefix}Canon_Params.u)',
        ),
        'ql': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['q', 'l'], '&&', ['q', 'l']),
            function_call = 'osqp_update_data_vec(&solver, {prefix}Canon_Params.q, {prefix}Canon_Params.l, OSQP_NULL)',
        ),
        'qu': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['q', 'u'], '&&', ['q', 'u']),
            function_call = 'osqp_update_data_vec(&solver, {prefix}Canon_Params.q, OSQP_NULL, {prefix}Canon_Params.u)',
        ),
        'lu': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['l', 'u'], '&&', ['l', 'u']),
            function_call = 'osqp_update_data_vec(&solver, OSQP_NULL, {prefix}Canon_Params.l, {prefix}Canon_Params.u)',
        ),
        'q': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['q']),
            function_call = 'osqp_update_data_vec(&solver, {prefix}Canon_Params.q, OSQP_NULL, OSQP_NULL)',
        ),
        'l': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['l']),
            function_call = 'osqp_update_data_vec(&solver, OSQP_NULL, {prefix}Canon_Params.l, OSQP_NULL)',
        ),
        'u': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['u']),
            function_call = 'osqp_update_data_vec(&solver, OSQP_NULL, OSQP_NULL, {prefix}Canon_Params.u)',
        )
    }
    solve_function_call = 'osqp_solve(&solver)'

    # header and source files
    header_files = ['"osqp.h"', '"workspace.h"']
    cmake_headers = [
        '${CMAKE_CURRENT_SOURCE_DIR}/*.h',
        '${CMAKE_CURRENT_SOURCE_DIR}/inc/public/*.h',
        '${CMAKE_CURRENT_SOURCE_DIR}/inc/private/*.h'
    ]
    cmake_sources = ['${OSQP_SOURCES}',  '${CMAKE_CURRENT_SOURCE_DIR}/workspace.c']

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # workspace
    ws_statically_allocated_in_solver_code = True
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'solver.info->obj_val',
        iterations = 'solver.info->iter',
        status = 'solver.info->status',
        primal_residual = 'solver.info->prim_res',
        dual_residual = 'solver.info->dual_res',
        primal_solution = 'sol_x',
        dual_solution = 'sol_{dual_var_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = False

    # float and integer types
    numeric_types = {'float': 'double', 'int': 'int'}

    # solver settings
    stgs_dynamically_allocated = False
    stgs_requires_extra_struct_type = True
    stgs_direct_write_ptr = 'solver.settings'
    stgs_reset_function = {'name': 'osqp_set_default_settings', 'ptr': 'solver.settings'}
    stgs_names = ['max_iter', 'eps_abs', 'eps_rel', 'eps_prim_inf', 'eps_dual_inf',
                      'scaled_termination', 'check_termination', 'warm_starting',
                      'verbose', 'polishing', 'polish_refine_iter', 'delta']
    stgs_translation = "{'warm_start': 'warm_starting'}"
    stgs_types = ['cpg_int', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float',
                      'cpg_int', 'cpg_int', 'cpg_int', 'cpg_int', 'cpg_int', 'cpg_int', 'cpg_float']
    stgs_enabled = [True, True, True, True, True, True, True, True,
                        False, False, False, False]
    stgs_defaults = ['4000', '1e-3', '1e-3', '1e-4', '1e-4',
                         '0', '25', '1',
                         '0', '0', '0', '1e-6']

    # dual variables split into two vectors
    dual_var_split = False
    dual_var_names = ['y']

    # docu
    docu = 'https://osqp.org/docs/codegen/python.html'

    def __init__(self, data, p_prob, enable_settings):
        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        indices_obj, indptr_obj, shape_obj = self.get_problem_data_index(p_prob.reduced_P)
        indices_constr, indptr_constr, shape_constr = self.get_problem_data_index(p_prob.reduced_A)

        canon_constants = {}

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                  parameter_canon: ParameterCanon) -> None:
        import osqp

        # OSQP codegen
        osqp_obj = osqp.OSQP()
        osqp_obj.setup(P=parameter_canon.p_csc['P'], q=parameter_canon.p['q'],
                    A=parameter_canon.p_csc['A'], l=parameter_canon.p['l'],
                    u=parameter_canon.p['u'])

        osqp_obj.codegen(os.path.join(code_dir, 'c', 'solver_code'), parameters='matrices', force_rewrite=True)

        # copy license files
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'osqp-python', 'LICENSE'),
                        os.path.join(solver_code_dir, 'LICENSE'))
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'LICENSE'), code_dir)

        # adjust workspace.h
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'workspace.h'),
                        lambda x: x.replace('extern OSQPSolver solver;',
                                            'extern OSQPSolver solver;\n'
                                            + f'  extern OSQPFloat sol_x[{self.n_var}];\n'
                                            + f'  extern OSQPFloat sol_y[{self.n_eq + self.n_ineq}];'))

        # modify CMakeLists.txt
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                        lambda x: cut_from_expr(x, 'add_library').replace('src', '${CMAKE_CURRENT_SOURCE_DIR}/src'))
        
        # adjust top-level CMakeLists.txt
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code'
        indent = ' ' * 6
        read_write_file(os.path.join(code_dir, 'c', 'CMakeLists.txt'),
                        lambda x: x.replace(sdir + '/include',
                                            sdir + '\n'
                                            + indent + sdir + '/inc/public\n'
                                            + indent + sdir + '/inc/private'))
        
        # adjust setup.py
        indent = ' ' * 30
        read_write_file(os.path.join(code_dir, 'setup.py'),
                        lambda x: x.replace("os.path.join('c', 'solver_code', 'include'),",
                                            "os.path.join('c', 'solver_code'),\n" +
                                            indent + "os.path.join('c', 'solver_code', 'inc', 'public'),\n" + 
                                            indent + "os.path.join('c', 'solver_code', 'inc', 'private'),"))

        # modify for extra settings
        if 'verbose' in self.enable_settings:
            read_write_file(os.path.join(code_dir, 'c', 'CMakeLists.txt'),
                    lambda x: x.replace('project (cvxpygen)', 'project (cvxpygen)\nadd_definitions(-DOSQP_ENABLE_PRINTING)'))


    def get_affine_map(self, p_id, param_prob, constraint_info: ConstraintInfo) -> AffineMap:
        affine_map = AffineMap()

        if p_id == 'P':
            if self.indices_obj is None: # problem is an LP
                return None
            affine_map.mapping = param_prob.reduced_P.reduced_mat
            affine_map.indices = self.indices_obj
            affine_map.shape = (self.n_var, self.n_var)
        elif p_id == 'q':
            affine_map.mapping = param_prob.q[:-1]
        elif p_id == 'd':
            affine_map.mapping = param_prob.q[-1]
        elif p_id == 'A':
            affine_map.mapping = param_prob.reduced_A.reduced_mat[
                                 :constraint_info.n_data_constr_mat]
            affine_map.indices = self.indices_constr[:constraint_info.n_data_constr_mat]
            affine_map.mapping[affine_map.indices >= self.n_eq] *= -1
            affine_map.shape = (self.n_eq + self.n_ineq, self.n_var)
        elif p_id == 'l':
            mapping_rows_eq = np.nonzero(self.indices_constr < self.n_eq)[0]
            affine_map.mapping_rows = mapping_rows_eq[
                mapping_rows_eq >= constraint_info.n_data_constr_mat]  # mapping to the finite part of l
            affine_map.sign = -1
            affine_map.indices = self.indices_constr[affine_map.mapping_rows]
            affine_map.shape = (self.n_eq, 1)
        elif p_id == 'u':
            affine_map.mapping_rows = np.arange(constraint_info.n_data_constr_mat,
                                                constraint_info.n_data_constr)
            affine_map.sign = np.vstack((-np.ones((self.n_eq, 1)), np.ones((self.n_ineq, 1))))
            affine_map.indices = self.indices_constr[affine_map.mapping_rows]
            affine_map.shape = (self.n_eq + self.n_ineq, 1)
        else:
            raise ValueError(f'Unknown OSQP parameter name {p_id}.')

        return affine_map
    
    def augment_vector_parameter(self, p_id, vector_parameter):
        if p_id == 'l':
            return np.concatenate((vector_parameter, -np.inf * np.ones(self.n_ineq)), axis=0)
        else:
            return vector_parameter


class SCSInterface(SolverInterface):
    solver_name = 'SCS'
    solver_type = 'conic'
    canon_p_ids = ['P', 'c', 'd', 'A', 'b']
    canon_p_ids_constr_vec = ['b']
    parameter_update_structure = {
        'init': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic([], extra_condition = '!{prefix}Scs_Work', functions_if_false = ['PA']),
            function_call = '{prefix}Scs_Work = scs_init(&{prefix}Scs_D, &{prefix}Scs_K, &{prefix}Canon_Settings)',
        ),
        'PA': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['P', 'A'], '||', functions_if_false = ['bc']),
            function_call = '{prefix}Scs_Work = scs_init(&{prefix}Scs_D, &{prefix}Scs_K, &{prefix}Canon_Settings)',
        ),
        'bc': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['b', 'c'], '&&', ['b', 'c']),
            function_call = 'scs_update({prefix}Scs_Work, {prefix}Canon_Params.b, {prefix}Canon_Params.c)'
        ),
        'b': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['b']),
            function_call = 'scs_update({prefix}Scs_Work, {prefix}Canon_Params.b, SCS_NULL)'
        ),
        'c': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['c']),
            function_call = 'scs_update({prefix}Scs_Work, SCS_NULL, {prefix}Canon_Params.c)'
        )
    }
    solve_function_call = 'scs_solve({prefix}Scs_Work, &{prefix}Scs_Sol, &{prefix}Scs_Info, ({prefix}Scs_Work && {prefix}Canon_Settings.warm_start))'

    # header files
    header_files = ['"scs.h"']
    cmake_headers = [
        '${${PROJECT_NAME}_HDR}',
        '${DIRSRC}/private.h',
        '${${PROJECT_NAME}_LDL_EXTERNAL_HDR}',
        '${${PROJECT_NAME}_AMD_EXTERNAL_HDR}',
        ]
    cmake_sources = [
        '${${PROJECT_NAME}_SRC}',
        '${DIRSRC}/private.c',
        '${EXTERNAL}/qdldl/qdldl.c',
        '${${PROJECT_NAME}_AMD_EXTERNAL_SRC}'
        ]

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'Scs_Info.pobj',
        iterations = 'Scs_Info.iter',
        status = 'Scs_Info.status',
        primal_residual = 'Scs_Info.res_pri',
        dual_residual = 'Scs_Info.res_dual',
        primal_solution = 'scs_x',
        dual_solution = 'scs_{dual_var_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = False

    # float and integer types
    numeric_types = {'float': 'scs_float', 'int': 'scs_int'}

    # solver settings
    stgs_dynamically_allocated = False
    stgs_requires_extra_struct_type = False
    stgs_direct_write_ptr = None
    stgs_reset_function = {'name': 'scs_set_default_settings', 'ptr': None} # set 'ptr' to None if stgs not statically allocated in solver code
    stgs_names = ['normalize', 'scale', 'adaptive_scale', 'rho_x', 'max_iters', 'eps_abs',
                      'eps_rel',
                      'eps_infeas', 'alpha', 'time_limit_secs', 'verbose', 'warm_start',
                      'acceleration_lookback',
                      'acceleration_interval', 'write_data_filename', 'log_csv_filename']
    stgs_translation = "{'max_iters': 'maxit'}"
    stgs_types = ['cpg_int', 'cpg_float', 'cpg_int', 'cpg_float', 'cpg_int', 'cpg_float', 'cpg_float',
                      'cpg_float', 'cpg_float',
                      'cpg_float', 'cpg_int', 'cpg_int', 'cpg_int', 'cpg_int', 'const char*', 'const char*']
    stgs_enabled = [True, True, True, True, True, True, True, True, True, True, True, True,
                        True, True, True, True]
    stgs_defaults = ['1', '0.1', '1', '1e-6', '1e5', '1e-4', '1e-4', '1e-7', '1.5', '0', '0',
                         '0', '0', '1',
                         'SCS_NULL', 'SCS_NULL']
    
    # dual variables split into two vectors
    dual_var_split = False
    dual_var_names = ['y']

    # docu
    docu = 'https://www.cvxgrp.org/scs/api/c.html'

    def __init__(self, data, p_prob, enable_settings):
        n_var = p_prob.x.size
        n_eq = data['A'].shape[0]
        n_ineq = 0

        indices_obj, indptr_obj, shape_obj = self.get_problem_data_index(p_prob.reduced_P)
        indices_constr, indptr_constr, shape_constr = self.get_problem_data_index(p_prob.reduced_A)

        canon_constants = {'n': n_var, 'm': n_eq, 'z': p_prob.cone_dims.zero,
                           'l': p_prob.cone_dims.nonneg,
                           'q': np.array(p_prob.cone_dims.soc),
                           'qsize': len(p_prob.cone_dims.soc)}

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0 or len(cone_dims.psd) > 0 or len(cone_dims.p3d) > 0:
            raise ValueError(
                'Code generation with SCS and exponential, positive semidefinite, or power cones '
                'is not supported yet.')

    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                  parameter_canon: ParameterCanon) -> None:

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'linsys', 'cmake']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'scs', dtc),
                            os.path.join(solver_code_dir, dtc))
        files_to_copy = ['scs.mk', 'CMakeLists.txt', 'LICENSE.txt']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'scs', fl),
                            os.path.join(solver_code_dir, fl))
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'LICENSE'), code_dir)

        # disable BLAS and LAPACK
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'scs.mk'),
                        lambda x: x.replace('USE_LAPACK = 1', 'USE_LAPACK = 0'))

        # modify CMakeLists.txt
        cmake_replacements = [
            (' include/', ' ${CMAKE_CURRENT_SOURCE_DIR}/include/'),
            (' src/', ' ${CMAKE_CURRENT_SOURCE_DIR}/src/'),
            (' ${LINSYS}/', ' ${CMAKE_CURRENT_SOURCE_DIR}/${LINSYS}/')
        ]
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, cmake_replacements))

        # adjust top-level CMakeLists.txt
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/'
        indent = ' ' * 6
        read_write_file(os.path.join(code_dir, 'c', 'CMakeLists.txt'),
                        lambda x: x.replace(sdir + 'include',
                                            sdir + 'include\n' + indent + sdir + 'linsys'))

        # adjust setup.py
        indent = ' ' * 30
        read_write_file(os.path.join(code_dir, 'setup.py'),
                        lambda x: x.replace("os.path.join('c', 'solver_code', 'include'),",
                                            "os.path.join('c', 'solver_code', 'include'),\n" +
                                            indent + "os.path.join('c', 'solver_code', 'linsys'),"))


    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        matrices = ['P', 'A'] if parameter_canon.quad_obj else ['A']
        for m in matrices:
            f.write(f'\n// SCS matrix {m}\n')
            write_struct_prot(f, f'{prefix}scs_{m}', 'ScsMatrix')
        f.write(f'\n// Struct containing SCS data\n')
        write_struct_prot(f, f'{prefix}Scs_D', 'ScsData')
        if self.canon_constants['qsize'] > 0:
            f.write(f'\n// SCS array of SOC dimensions\n')
            write_vec_prot(f, self.canon_constants['q'], f'{prefix}scs_q', 'cpg_int')
        f.write(f'\n// Struct containing SCS cone data\n')
        write_struct_prot(f, f'{prefix}Scs_K', 'ScsCone')
        f.write(f'\n// Struct containing SCS settings\n')
        write_struct_prot(f, f'{prefix}Canon_Settings', 'ScsSettings')
        f.write(f'\n// SCS solution\n')
        write_vec_prot(f, np.zeros(self.canon_constants['n']), f'{prefix}scs_x', 'cpg_float')
        write_vec_prot(f, np.zeros(self.canon_constants['m']), f'{prefix}scs_y', 'cpg_float')
        write_vec_prot(f, np.zeros(self.canon_constants['m']), f'{prefix}scs_s', 'cpg_float')
        f.write(f'\n// Struct containing SCS solution\n')
        write_struct_prot(f, f'{prefix}Scs_Sol', 'ScsSolution')
        f.write(f'\n// Struct containing SCS information\n')
        write_struct_prot(f, f'{prefix}Scs_Info', 'ScsInfo')
        f.write(f'\n// Pointer to struct containing SCS workspace\n')
        write_struct_prot(f, f'{prefix}Scs_Work', 'ScsWork*')


    def define_workspace(self, f, prefix, parameter_canon) -> None:
        matrices = ['P', 'A'] if parameter_canon.quad_obj else ['A']
        scs_PA_fields = ['x', 'i', 'p', 'm', 'n']
        scs_PA_casts = ['(cpg_float *) ', '(cpg_int *) ', '(cpg_int *) ', '', '']
        for m in matrices:
            f.write(f'\n// SCS matrix {m}\n')
            scs_PA_values = [f'&{prefix}canon_{m}_x', f'&{prefix}canon_{m}_i',
                            f'&{prefix}canon_{m}_p', str(self.canon_constants[('n' if m == 'P' else 'm')]),
                            str(self.canon_constants['n'])]
            write_struct_def(f, scs_PA_fields, scs_PA_casts, scs_PA_values, f'{prefix}Scs_{m}', 'ScsMatrix')

        f.write(f'\n// Struct containing SCS data\n')
        scs_d_fields = ['m', 'n', 'A', 'P', 'b', 'c']
        scs_d_casts = ['', '', '', '', '(cpg_float *) ', '(cpg_float *) ']
        scs_d_values = [str(self.canon_constants['m']), str(self.canon_constants['n']),
                        f'&{prefix}Scs_A', (f'&{prefix}Scs_P' if parameter_canon.quad_obj else 'SCS_NULL'),
                        f'&{prefix}canon_b', f'&{prefix}canon_c']
        write_struct_def(f, scs_d_fields, scs_d_casts, scs_d_values, f'{prefix}Scs_D', 'ScsData')

        if self.canon_constants['qsize'] > 0:
            f.write(f'\n// SCS array of SOC dimensions\n')
            write_vec_def(f, self.canon_constants['q'], f'{prefix}scs_q', 'cpg_int')
            k_field_q_str = f'&{prefix}scs_q'
        else:
            k_field_q_str = 'SCS_NULL'

        f.write(f'\n// Struct containing SCS cone data\n')
        scs_k_fields = ['z', 'l', 'bu', 'bl', 'bsize', 'q', 'qsize', 's', 'ssize', 'ep', 'ed', 'p', 'psize']
        scs_k_casts = ['', '', '(cpg_float *) ', '(cpg_float *) ', '', '(cpg_int *) ', '', '(cpg_int *) ', '', '', '',
                    '(cpg_float *) ', '']
        scs_k_values = [str(self.canon_constants['z']), str(self.canon_constants['l']), 'SCS_NULL', 'SCS_NULL', '0',
                        k_field_q_str, str(self.canon_constants['qsize']), 'SCS_NULL', '0', '0', '0', 'SCS_NULL', '0']
        write_struct_def(f, scs_k_fields, scs_k_casts, scs_k_values, f'{prefix}Scs_K', 'ScsCone')

        f.write(f'\n// Struct containing SCS settings\n')
        scs_stgs_fields = list(self.stgs_names_to_default.keys())
        scs_stgs_casts = [''] * len(scs_stgs_fields)
        scs_stgs_values = list(self.stgs_names_to_default.values())
        write_struct_def(f, scs_stgs_fields, scs_stgs_casts, scs_stgs_values, f'{prefix}Canon_Settings', 'ScsSettings')

        f.write(f'\n// SCS solution\n')
        write_vec_def(f, np.zeros(self.canon_constants['n']), f'{prefix}scs_x', 'cpg_float')
        write_vec_def(f, np.zeros(self.canon_constants['m']), f'{prefix}scs_y', 'cpg_float')
        write_vec_def(f, np.zeros(self.canon_constants['m']), f'{prefix}scs_s', 'cpg_float')

        f.write(f'\n// Struct containing SCS solution\n')
        scs_sol_fields = ['x', 'y', 's']
        scs_sol_casts = ['(cpg_float *) ', '(cpg_float *) ', '(cpg_float *) ']
        scs_sol_values = [f'&{prefix}scs_x', f'&{prefix}scs_y', f'&{prefix}scs_s']
        write_struct_def(f, scs_sol_fields, scs_sol_casts, scs_sol_values, f'{prefix}Scs_Sol', 'ScsSolution')

        f.write(f'\n// Struct containing SCS information\n')
        scs_info_fields = ['iter', 'status', 'status_val', 'scale_updates', 'pobj', 'dobj', 'res_pri', 'res_dual',
                        'gap', 'res_infeas', 'res_unbdd_a', 'res_unbdd_p', 'comp_slack', 'setup_time', 'solve_time',
                        'scale', 'rejected_accel_steps', 'accepted_accel_steps', 'lin_sys_time', 'cone_time',
                        'accel_time']
        scs_info_casts = [''] * len(scs_info_fields)
        scs_info_values = ['0', '"unknown"', '0', '0', '0', '0', '99', '99', '99', '99', '99', '99', '99', '0', '0',
                        '1', '0', '0', '0', '0', '0']
        write_struct_def(f, scs_info_fields, scs_info_casts, scs_info_values, f'{prefix}Scs_Info', 'ScsInfo')

        f.write(f'\n// Pointer to struct containing SCS workspace\n')
        f.write(f'ScsWork* {prefix}Scs_Work = 0;\n')


class ECOSInterface(SolverInterface):
    solver_name = 'ECOS'
    solver_type = 'conic'
    canon_p_ids = ['c', 'd', 'A', 'b', 'G', 'h']
    canon_p_ids_constr_vec = ['b', 'h']
    solve_function_call = '{prefix}ecos_flag = ECOS_solve({prefix}ecos_workspace)'

    # header files
    header_files = ['"ecos.h"']
    cmake_headers = ['${ecos_headers}']
    cmake_sources = ['${ecos_sources}']

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = True

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'ecos_workspace->info->pcost',
        iterations = 'ecos_workspace->info->iter',
        status = 'ecos_flag',
        primal_residual = 'ecos_workspace->info->pres',
        dual_residual = 'ecos_workspace->info->dres',
        primal_solution = 'ecos_workspace->x',
        dual_solution = 'ecos_workspace->{dual_var_name}',
        settings = 'ecos_workspace->stgs->{setting_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = False

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'double', 'int': 'int'}

    # solver settings
    stgs_dynamically_allocated = True
    stgs_requires_extra_struct_type = True
    stgs_direct_write_ptr = None
    stgs_reset_function = None
    stgs_names = ['feastol', 'abstol', 'reltol', 'feastol_inacc', 'abstol_inacc',
                      'reltol_inacc', 'maxit']
    stgs_translation = "{}"
    stgs_types = ['cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_int']
    stgs_enabled = [True, True, True, True, True, True, True]
    stgs_defaults = ['1e-8', '1e-8', '1e-8', '1e-4', '5e-5', '5e-5', '100']

    # dual variables split into y and z vectors
    dual_var_split = True
    dual_var_names = ['y', 'z']

    # docu
    docu = 'https://github.com/embotech/ecos/wiki/Usage-from-C'

    def __init__(self, data, p_prob, enable_settings):
        n_var = p_prob.x.size
        n_eq = p_prob.cone_dims.zero
        n_ineq = data['G'].shape[0]

        indices_obj, indptr_obj, shape_obj = self.get_problem_data_index(p_prob.reduced_P)
        indices_constr, indptr_constr, shape_constr = self.get_problem_data_index(p_prob.reduced_A)

        canon_constants = {'n': n_var, 'm': n_ineq, 'p': n_eq,
                           'l': p_prob.cone_dims.nonneg,
                           'n_cones': len(p_prob.cone_dims.soc),
                           'q': np.array(p_prob.cone_dims.soc),
                           'e': p_prob.cone_dims.exp}

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition='!{prefix}ecos_workspace', functions_if_false=['AbcGh']),
                function_call=f'{{prefix}}ecos_workspace = ECOS_setup({canon_constants["n"]}, {canon_constants["m"]}, {canon_constants["p"]}, {canon_constants["l"]}, {canon_constants["n_cones"]}'
                            f', {"0" if canon_constants["n_cones"] == 0 else "(int *) &{prefix}ecos_q"}, {canon_constants["e"]}'
                            f', {{prefix}}Canon_Params_conditioning.G->x, {{prefix}}Canon_Params_conditioning.G->p, {{prefix}}Canon_Params_conditioning.G->i'
                            f', {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.A->x"}'
                            f', {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.A->p"}'
                            f', {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.A->i"}'
                            f', {{prefix}}Canon_Params_conditioning.c, {{prefix}}Canon_Params_conditioning.h'
                            f', {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.b"})'
            ),
            'AbcGh': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic(['A', 'b', 'G'], '||', ['c', 'h']),
                function_call=f'ECOS_updateData({{prefix}}ecos_workspace, {{prefix}}Canon_Params_conditioning.G->x, {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.A->x"}'
                            f', {{prefix}}Canon_Params_conditioning.c, {{prefix}}Canon_Params_conditioning.h, {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.b"})'
            ),
            'c': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic(['c']),
                function_call=f'for (i=0; i<{canon_constants["n"]}; i++) {{{{ ecos_updateDataEntry_c({{prefix}}ecos_workspace, i, {{prefix}}Canon_Params_conditioning.c[i]); }}}}'
            ),
            'h': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic(['h']),
                function_call=f'for (i=0; i<{canon_constants["m"]}; i++) {{{{ ecos_updateDataEntry_h({{prefix}}ecos_workspace, i, {{prefix}}Canon_Params_conditioning.h[i]); }}}}'
            )
        }

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0:
            raise ValueError(
                'Code generation with ECOS and exponential cones is not supported yet.')

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                  parameter_canon: ParameterCanon) -> None:

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'external', 'ecos_bb']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'ecos', dtc),
                            os.path.join(solver_code_dir, dtc))
        
        files_to_copy = ['CMakeLists.txt', 'COPYING']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', fl),
                            os.path.join(solver_code_dir, fl))
        
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', 'COPYING'),
                        os.path.join(code_dir, 'COPYING'))

        # adjust print level
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'glblopts.h'),
                        lambda x: x.replace('#define PRINTLEVEL (2)', '#define PRINTLEVEL (0)'))

        # adjust top-level CMakeLists.txt
        indent = ' ' * 6
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/'
        cmake_replacements = [
            (sdir + 'include',
            sdir + 'include\n' +
            indent + sdir + 'external/SuiteSparse_config\n' +
            indent + sdir + 'external/amd/include\n' +
            indent + sdir + 'external/ldl/include')
        ]
        read_write_file(os.path.join(code_dir, 'c', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, cmake_replacements))

        # remove library target from ECOS CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'r') as f:
            lines = [line for line in f if '# ECOS library' not in line]
        with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'w') as f:
            f.writelines(lines)

        # adjust setup.py
        indent = ' ' * 30
        setup_replacements = [
            ("os.path.join('c', 'solver_code', 'include'),",
            "os.path.join('c', 'solver_code', 'include'),\n" +
            indent + "os.path.join('c', 'solver_code', 'external', 'SuiteSparse_config'),\n" +
            indent + "os.path.join('c', 'solver_code', 'external', 'amd', 'include'),\n" +
            indent + "os.path.join('c', 'solver_code', 'external', 'ldl', 'include'),"),
            ("license='Apache 2.0'", "license='GPL 3.0'")
        ]
        read_write_file(os.path.join(code_dir, 'setup.py'),
                        lambda x: multiple_replace(x, setup_replacements))


    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['n_cones'] > 0:
            f.write('\n// ECOS array of SOC dimensions\n')
            write_vec_prot(f, self.canon_constants['q'], f'{prefix}ecos_q', 'cpg_int')
        f.write('\n// ECOS workspace\n')
        f.write(f'extern pwork* {prefix}ecos_workspace;\n')
        f.write('\n// ECOS exit flag\n')
        f.write(f'extern cpg_int {prefix}ecos_flag;\n')

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['n_cones'] > 0:
            f.write('\n// ECOS array of SOC dimensions\n')
            write_vec_def(f, self.canon_constants['q'], f'{prefix}ecos_q', 'cpg_int')
        f.write('\n// ECOS workspace\n')
        f.write(f'pwork* {prefix}ecos_workspace = 0;\n')
        f.write('\n// ECOS exit flag\n')
        f.write(f'cpg_int {prefix}ecos_flag = -99;\n')


class ClarabelInterface(SolverInterface):
    solver_name = 'Clarabel'
    solver_type = 'conic'
    canon_p_ids = ['P', 'q', 'd', 'A', 'b']
    canon_p_ids_constr_vec = ['b']

    # header and source files
    header_files = ['<Clarabel>']
    cmake_headers, cmake_sources = [], []

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = True

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'solution.obj_val',
        iterations = 'solution.iterations',
        status = 'solution.status',
        primal_residual = 'solution.r_prim',
        dual_residual = 'solution.r_dual',
        primal_solution = 'solution.x',
        dual_solution = 'solution.{dual_var_name}',
        settings = 'settings.{setting_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = False

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'ClarabelFloat', 'int': 'uintptr_t'}
    
    # solver settings
    stgs_dynamically_allocated = True
    stgs_requires_extra_struct_type = True
    stgs_direct_write_ptr = None
    stgs_reset_function = None
    # TODO: extend to all available settings
    stgs_names = ['max_iter', 'time_limit', 'verbose', 'max_step_fraction',
                  'equilibrate_enable', 'equilibrate_max_iter', 'equilibrate_min_scaling', 'equilibrate_max_scaling']
    stgs_translation = "{}"
    stgs_types = ['cpg_int', 'cpg_float', 'cpg_int', 'cpg_float',
                  'cpg_int', 'cpg_int', 'cpg_float', 'cpg_float']
    stgs_enabled = [True, True, True, True, True, True, True, True]
    stgs_defaults = ['50', '1e6', '1', '0.99',
                     '1', '10', '1e-4', '1e4']

    # dual variables split into two vectors
    dual_var_split = False
    dual_var_names = ['z']

    # docu
    docu = 'https://oxfordcontrol.github.io/ClarabelDocs/'

    def __init__(self, data, p_prob, enable_settings):
        n_var = p_prob.x.size
        n_eq = data['A'].shape[0]
        n_ineq = 0

        indices_obj, indptr_obj, shape_obj = self.get_problem_data_index(p_prob.reduced_P)
        indices_constr, indptr_constr, shape_constr = self.get_problem_data_index(p_prob.reduced_A)

        canon_constants = {'n': n_var, 'm': n_eq,
                           'cone_dims_zero': p_prob.cone_dims.zero,
                           'cone_dims_nonneg': p_prob.cone_dims.nonneg,
                           'cone_dims_exp': p_prob.cone_dims.exp,
                           'cone_dims_soc': np.array(p_prob.cone_dims.soc),
                           'cone_dims_psd': np.array(p_prob.cone_dims.psd),
                           'cone_dims_p3d': np.array(p_prob.cone_dims.p3d)}
        
        canon_constants['n_cone_types'] = int(p_prob.cone_dims.zero > 0) + \
            int(p_prob.cone_dims.nonneg > 0) + \
            int(p_prob.cone_dims.exp > 0) + \
            len(p_prob.cone_dims.soc) + \
            len(p_prob.cone_dims.psd) + \
            len(p_prob.cone_dims.p3d)
        
        canon_constants['cd_to_t'] = {
            'cone_dims_zero': 'ClarabelZeroConeT',
            'cone_dims_nonneg': 'ClarabelNonnegativeConeT',
            'cone_dims_exp': 'ClarabelExponentialConeT',
            'cone_dims_soc': 'ClarabelSecondOrderConeT',
            'cone_dims_psd': 'ClarabelPSDTriangleConeT',
            'cone_dims_p3d': 'ClarabelPowerConeT'
        }

        # catch LP case (hack, until Clarabel permits passing a zero pointer as &P)
        if indices_obj is None:
            extra_condition = '1'
            P_p = f'(cpg_int[]){{{{ {", ".join(["0"]*(n_var+1))} }}}}'
            P_i = '0'
            P_x = '0'
        else:
            extra_condition = '!{prefix}solver'
            P_p = '{prefix}Canon_Params_conditioning.P->p'
            P_i = '{prefix}Canon_Params_conditioning.P->i'
            P_x = '{prefix}Canon_Params_conditioning.P->x'

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition=extra_condition, functions_if_false=[]),
                function_call= \
                    f'clarabel_CscMatrix_init(&{{prefix}}P, {canon_constants["n"]}, {canon_constants["n"]}, {P_p}, {P_i}, {P_x});\n'
                    f'    clarabel_CscMatrix_init(&{{prefix}}A, {canon_constants["m"]}, {canon_constants["n"]}, {{prefix}}Canon_Params_conditioning.A->p, {{prefix}}Canon_Params_conditioning.A->i, {{prefix}}Canon_Params_conditioning.A->x);\n' \
                    f'    {{prefix}}settings = clarabel_DefaultSettings_default()'
            )
        }

        self.solve_function_call = \
            f'{{prefix}}solver = clarabel_DefaultSolver_new(&{{prefix}}P, {{prefix}}Canon_Params_conditioning.q, &{{prefix}}A, {{prefix}}Canon_Params_conditioning.b, {canon_constants["n_cone_types"]}, {{prefix}}cones, &{{prefix}}settings);\n' \
            f'  clarabel_DefaultSolver_solve({{prefix}}solver);\n' \
            f'  {{prefix}}solution = clarabel_DefaultSolver_solution({{prefix}}solver)'

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                    parameter_canon: ParameterCanon) -> None:

        # check if sdp cones are present
        is_sdp = len(self.canon_constants['cone_dims_psd']) > 0
        if is_sdp:
            sys.stdout.write('WARNING: You are generating code for an SDP with Clarabel, which requires BLAS and LAPACK within Rust-C wrappers - expect large compilation time and binary size.\n')

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['rust_wrapper', 'include', 'Clarabel.rs']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'Clarabel.cpp', dtc),
                            os.path.join(solver_code_dir, dtc))
        files_to_copy = ['CMakeLists.txt', 'LICENSE.md']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'Clarabel.cpp', fl),
                            os.path.join(solver_code_dir, fl))
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'LICENSE'), code_dir)

        # adjust top-level CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'a') as f:
            if is_sdp:
                f.write('\nfind_package(BLAS REQUIRED)')
                f.write('\nfind_package(LAPACK REQUIRED)')
                link_libraries = 'libclarabel_c_static ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}'
            else:
                link_libraries = 'libclarabel_c_static'
            f.write(f'\ntarget_link_libraries(cpg_example PRIVATE {link_libraries})')
            f.write(f'\ntarget_link_libraries(cpg PRIVATE {link_libraries})\n')

        # remove examples target from Clarabel.cpp/CMakeLists.txt
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                        lambda x: x.replace('add_subdirectory(examples)', '# add_subdirectory(examples)'))

        # add sdp flag
        if is_sdp:
            read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                            lambda x: x.replace('set(CLARABEL_FEATURE_SDP "none"', 'set(CLARABEL_FEATURE_SDP "sdp-openblas"'))

        # adjust paths in Clarabel.cpp/rust_wrapper/CMakeLists.txt
        replacements = [
            ('${CMAKE_SOURCE_DIR}/', '${CMAKE_SOURCE_DIR}/solver_code/'),
            ('/libclarabel_c.lib', '/clarabel_c.lib')  # until fixed on Clarabel side
        ]
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'rust_wrapper', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, replacements))

        # adjust Clarabel
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'Clarabel'),
                        lambda x: x.replace('cpp/', 'c/'))

        # adjust setup.py
        read_write_file(os.path.join(code_dir, 'setup.py'),
                        lambda x: x.replace("extra_objects=[cpg_lib])",
                                            "extra_objects=[cpg_lib, os.path.join(cpg_dir, 'solver_code', 'rust_wrapper', 'target', 'debug', 'libclarabel_c.a')])"))

    
    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        f.write('\n// Clarabel workspace\n')
        f.write(f'extern ClarabelCscMatrix {prefix}P;\n')
        f.write(f'extern ClarabelCscMatrix {prefix}A;\n')
        f.write(f'extern ClarabelSupportedConeT {prefix}cones[{self.canon_constants["n_cone_types"]}];\n')
        f.write(f'extern ClarabelDefaultSettings {prefix}settings;\n')
        f.write(f'extern ClarabelDefaultSolver *{prefix}solver;\n')
        f.write(f'extern ClarabelDefaultSolution {prefix}solution;\n')

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        f.write('\n// Clarabel workspace\n')
        f.write(f'ClarabelCscMatrix {prefix}P;\n')
        f.write(f'ClarabelCscMatrix {prefix}A;\n')
        cone_str_list = []
        for cd in ['cone_dims_zero', 'cone_dims_nonneg']:
            if self.canon_constants[cd] > 0:
                cone_str_list.append(f'{self.canon_constants["cd_to_t"][cd]}({self.canon_constants[cd]})')
        cone_str_list.extend(['ClarabelExponentialConeT()'] * self.canon_constants['cone_dims_exp'])
        for cd in ['cone_dims_soc', 'cone_dims_psd', 'cone_dims_p3d']:
            for l in self.canon_constants[cd]:
                cone_str_list.append(f'{self.canon_constants["cd_to_t"][cd]}({l})')
        f.write(f'ClarabelSupportedConeT {prefix}cones[{self.canon_constants["n_cone_types"]}] = {{ {", ".join(cone_str_list)} }};\n')
        f.write(f'ClarabelDefaultSettings {prefix}settings;\n')
        f.write(f'ClarabelDefaultSolver *{prefix}solver = 0;\n')
        f.write(f'ClarabelDefaultSolution {prefix}solution;\n')
