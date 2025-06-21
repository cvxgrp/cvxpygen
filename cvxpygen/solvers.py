import os
import sys
import shutil
import warnings
from abc import ABC, abstractmethod
import platform

import numpy as np
import scipy.sparse as sp

from cvxpygen.utils import write_file, read_write_file, write_struct_prot, write_struct_def, \
    write_vec_prot, write_vec_def, multiple_replace, cut_from_expr, \
    write_description, type_to_cast, write_mat_def, write_L_def, ones, zeros, write_canonicalize
from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, ConstraintInfo, AffineMap, \
    ParameterCanon, WorkspacePointerInfo, UpdatePendingLogic, ParameterUpdateLogic, Setting

from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL
from cvxpy.reductions.solvers.conic_solvers.qoco_conif import QOCO


UNSUPPORTED_ON_WINDOWS = ['CLARABEL']


def get_interface_class(solver_name: str) -> "SolverInterface":
    if platform.system() == 'Windows' and solver_name.upper() in UNSUPPORTED_ON_WINDOWS:
        raise ValueError(f'{solver_name} solver currently not supported on Windows.')
    mapping = {
        'OSQP': (OSQPInterface, OSQP),
        'SCS': (SCSInterface, SCS),
        'ECOS': (ECOSInterface, ECOS),
        'CLARABEL': (ClarabelInterface, CLARABEL),
        'QOCO': (QOCOInterface, QOCO),
        'QOCOGEN': (QOCOGENInterface, QOCO),
    }
    interface = mapping.get(solver_name.upper(), None)
    if interface is None:
        raise ValueError(f'Unsupported solver: {solver_name}.')
    return interface


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
    def stgs(self):
        pass

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return any(variable_info.sym) or any([s == 1 for s in variable_info.sizes])

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return any([s == 1 for s in dual_variable_info.sizes])

    def configure_settings(self) -> None:
        for s in self.stgs:
            if s in self.enable_settings:
                self.stgs[s].enabled = True
        for s in set(self.enable_settings)-set(self.stgs):
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
            affine_map.mapping = param_prob.c[[-1], :]
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
        return [n for n, s in self.stgs.items() if s.enabled]

    @property
    def stgs_names_to_type(self):
        return {n: s.type for n, s in self.stgs.items() if s.enabled}
    
    @property
    def stgs_names_to_default(self):
        return {n: s.default for n, s in self.stgs.items() if s.enabled}
    
    @property
    def stgs_translation(self):
        return {s.name_cvxpy: n for n, s in self.stgs.items() if s.enabled and s.name_cvxpy is not None}
    
    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        pass

    @abstractmethod
    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                      parameter_canon: ParameterCanon, gradient, prefix) -> None:
        pass

    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        pass

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        pass
    
    def write_gradient_def(f, configuration,
                           variable_info_first, dual_variable_info_first,
                           variable_info_second, dual_variable_info_second,
                           parameter_info, parameter_canon, solver_interface) -> None:
        pass
    
    def write_gradient_prot(f, configuration,
                            variable_info_first, dual_variable_info_first,
                            variable_info_second, dual_variable_info_second,
                            parameter_info, parameter_canon, solver_interface) -> None:
        pass
    
    def write_gradient_workspace_def(f, prefix, parameter_canon) -> None:
        pass


class OSQPInterface(SolverInterface):
    solver_name = 'OSQP'
    solver_type = 'quadratic'
    canon_p_ids = ['P', 'q', 'd', 'A', 'l', 'u']
    canon_p_ids_constr_vec = ['l', 'u']
    supports_gradient = True
    parameter_update_structure = {
        'PA': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['P', 'A'], '&&', ['P', 'A']),
            function_call = 'osqp_update_data_mat(&solver, {prefix}Canon_Params.P->x, 0, {prefix}Canon_Params.P->nnz, {prefix}Canon_Params.A->x, 0, {prefix}Canon_Params.A->nnz)',
        ),
        'P': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['P']),
            function_call = 'osqp_update_data_mat(&solver, {prefix}Canon_Params.P->x, 0, {prefix}Canon_Params.P->nnz, OSQP_NULL, 0, 0)',
        ),
        'A': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['A']),
            function_call = 'osqp_update_data_mat(&solver, OSQP_NULL, 0, 0, {prefix}Canon_Params.A->x, 0, {prefix}Canon_Params.A->nnz)'
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
    stgs = {
        'max_iter': Setting(type='cpg_int', default='4000'),
        'eps_abs': Setting(type='cpg_float', default='1e-3'),
        'eps_rel': Setting(type='cpg_float', default='1e-3'),
        'eps_prim_inf': Setting(type='cpg_float', default='1e-4'),
        'eps_dual_inf': Setting(type='cpg_float', default='1e-4'),
        'scaled_termination': Setting(type='cpg_int', default='0'),
        'check_termination': Setting(type='cpg_int', default='25'),
        'warm_starting': Setting(type='cpg_int', default='1', name_cvxpy='warm_start'),
        'verbose': Setting(type='cpg_int', default='0', enabled=False),
        'polishing': Setting(type='cpg_int', default='0', enabled=False),
        'polish_refine_iter': Setting(type='cpg_int', default='0', enabled=False),
        'delta': Setting(type='cpg_float', default='1e-6', enabled=False)
    }

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
        
    
    def write_gradient_def(self, f, configuration,
                           variable_info_first, dual_variable_info_first,
                           variable_info_second, dual_variable_info_second,
                           parameter_info, parameter_canon, solver_interface):
        """
        Write parameter initialization function to file
        """
        
        if configuration.gradient_two_stage:
            prefix = f'gradient_{configuration.prefix}'
        else:
            prefix = configuration.prefix
        
        n = self.n_var
        nl = self.n_eq
        nu = self.n_eq + self.n_ineq
        N = n + nu
        NP = len(parameter_info.flat_usp)

        write_description(f, 'c', 'Function definitions')
        f.write('#include "cpg_gradient.h"\n')
        if configuration.gradient_two_stage:
            f.write('#include "cpg_gradient_workspace.h"\n')
            f.write('#include "cpg_workspace.h"\n')
        f.write('#include "cpg_osqp_grad_workspace.h"\n')
        f.write('#include "cpg_osqp_grad_compute.h"\n')
        f.write('\n')

        f.write('static cpg_int i, j, k;\n\n')
        
        if configuration.gradient_two_stage:
            
            ret_prim_func_needed = solver_interface.ret_prim_func_exists(variable_info_second)
            
            # define sol_x and sol_y to point to the dual solution
            result_prefix = configuration.prefix if not solver_interface.ws_statically_allocated_in_solver_code else ''
            if ret_prim_func_needed:
                write_vec_def(f, np.zeros(n), f'{prefix}sol_x', 'cpg_float')
            else:
                f.write(f'cpg_float* {prefix}sol_x = (cpg_float*) &{result_prefix}{solver_interface.ws_ptrs.primal_solution} + {variable_info_second.name_to_offset["osqp_x"]};\n')
            write_vec_def(f, np.zeros(nu), f'{prefix}sol_y', 'cpg_float')
            
            # retrieve intermediate primal
            if ret_prim_func_needed:
                offset = variable_info_second.name_to_offset['osqp_x']
                x_term = f'{result_prefix}{solver_interface.ws_ptrs.primal_solution}[{offset} + i]'
                f.write(f'\nvoid {prefix}cpg_retrieve_intermediate_primal(){{\n')
                f.write(f'  for(i=0; i<{n}; i++){{\n')
                f.write(f'    {prefix}sol_x[i] = {x_term};\n')
                f.write('  }\n')
                f.write('}\n\n')
            
            # retrieve intermediate dual
            vec_l = dual_variable_info_second.name_to_vec['d0']
            vec_u = dual_variable_info_second.name_to_vec['d1']
            offset_l = dual_variable_info_second.name_to_offset['d0']
            offset_u = dual_variable_info_second.name_to_offset['d1']
            size_l = dual_variable_info_second.name_to_size['d0']
            size_u = dual_variable_info_second.name_to_size['d1']
            l_term = f'{result_prefix}{solver_interface.ws_ptrs.dual_solution.format(dual_var_name=vec_l)}[{offset_l} + i]'
            u_term = f'{result_prefix}{solver_interface.ws_ptrs.dual_solution.format(dual_var_name=vec_u)}[{offset_u} + i]'
            f.write(f'\nvoid {prefix}cpg_retrieve_intermediate_dual(){{\n')
            f.write(f'  for(i=0; i<{size_l}; i++){{\n')
            f.write(f'    {prefix}sol_y[i] = {u_term} - {l_term};\n')
            f.write('  }\n')
            f.write(f'  for(i={size_l}; i<{size_u}; i++){{\n')
            f.write(f'    {prefix}sol_y[i] = {u_term};\n')
            f.write('  }\n')
            f.write('}\n\n')
            
            for p_id, mapping in parameter_canon.p_id_to_mapping.items():
                if parameter_canon.p_id_to_changes[p_id]:
                    f.write(f'void {prefix}cpg_canonicalize_{p_id}(){{\n')
                    s = '->x' if p_id.isupper() else ''
                    write_canonicalize(f, p_id, s, mapping, prefix, configuration.prefix)
                    f.write('}\n\n')
        
        f.write('// Update user-defined variable deltas\n')
        for name, size in variable_info_first.name_to_size.items():
            if size == 1:
                f.write(f'void {configuration.prefix}cpg_update_d{name}(cpg_float val){{\n')
                f.write(f'  {prefix}CPG_OSQP_Grad.dx[{variable_info_first.name_to_offset[name]}] = val;\n')
            else:
                f.write(f'void {configuration.prefix}cpg_update_d{name}(cpg_int idx, cpg_float val){{\n')
                f.write(f'  {prefix}CPG_OSQP_Grad.dx[{variable_info_first.name_to_offset[name]}+idx] = val;\n')
            f.write('}\n')
        
        f.write('\n// End-to-end gradient\n')
        f.write(f'void {configuration.prefix}cpg_gradient(){{\n')
        if configuration.gradient_two_stage:
            for p_id, changes in parameter_canon.p_id_to_changes.items():
                if changes:
                    f.write(f'  if ({prefix}Canon_Outdated_Grad.{p_id}) {{\n')
                    f.write(f'    {prefix}cpg_canonicalize_{p_id}();\n')
                    f.write('  }\n')
        changing_matrices = []
        for p_id, changes in parameter_canon.p_id_to_changes.items():
            if p_id.isupper() and changes:
                changing_matrices.append(p_id)
                f.write(f'  if ({prefix}Canon_Outdated_Grad.{p_id}) {{\n')
                f.write(f'    cpg_{p_id}_to_K((cpg_grad_csc*) {prefix}Canon_Params.{p_id}, {prefix}CPG_OSQP_Grad.K, {prefix}CPG_OSQP_Grad.K_true);\n')
                f.write('  }\n')
        f.write(f'  if ({prefix}CPG_OSQP_Grad.init) {{\n')
        f.write('    cpg_ldl_symbolic();\n')
        f.write('    cpg_ldl_numeric();\n')
        f.write(f'    for (j=0; j<{N-1}; j++){{\n')
        f.write(f'      for (k={prefix}CPG_OSQP_Grad.L->p[j]; k<{prefix}CPG_OSQP_Grad.L->p[j+1]; k++){{\n')
        f.write(f'        i = {prefix}CPG_OSQP_Grad.L->i[k];\n')
        f.write(f'        {prefix}CPG_OSQP_Grad.Lmask[({2*N-3}-j)*j/2+i-1] = 1;\n')
        f.write('      }\n')
        f.write('    }\n')
        f.write(f'    {prefix}CPG_OSQP_Grad.init = 0;\n')
        # If any of {prefix}Canon_Outdated_Grad.{p_id} with p_id in changing_matrices is true, then call cpg_ldl_numeric()
        if len(changing_matrices) > 0:
            f.write('  } else {\n')
            f.write(f'    if ({f" || ".join([f"{prefix}Canon_Outdated_Grad.{p_id}" for p_id in changing_matrices])}) {{\n')
            f.write(f'      cpg_ldl_numeric();\n')
            # set all CPG_OSQP_Grad.a[i] to 1
            f.write(f'      for(i=0; i<{N - n}; i++){{\n')
            f.write(f'        {prefix}CPG_OSQP_Grad.a[i] = 1;\n')
            f.write('      }\n')
            f.write('    }\n')
        f.write('  }\n')
        f.write('  // Canonical gradient\n')
        f.write(f'  cpg_osqp_gradient();\n')
        f.write('  // Un-canonicalize\n')
        f.write(f'  for(j=0; j<{NP}; j++){{\n')
        f.write(f'      {configuration.prefix}cpg_dp[j] = 0.0;\n')
        f.write('  }\n')
        if parameter_canon.p_id_to_changes['q']:
            f.write(f'  for(j=0; j<{n}; j++){{\n')
            f.write(f'      for(k={prefix}canon_q_map.p[j]; k<{prefix}canon_q_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_q_map.i[k]] += {prefix}canon_q_map.x[k]*{prefix}CPG_OSQP_Grad.dq[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        if parameter_canon.p_id_to_changes['l']:
            f.write(f'  for(j=0; j<{nl}; j++){{\n')
            f.write(f'      for(k={prefix}canon_l_map.p[j]; k<{prefix}canon_l_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_l_map.i[k]] += {prefix}canon_l_map.x[k]*{prefix}CPG_OSQP_Grad.dl[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        if parameter_canon.p_id_to_changes['u']:
            f.write(f'  for(j=0; j<{nu}; j++){{\n')
            f.write(f'      for(k={prefix}canon_u_map.p[j]; k<{prefix}canon_u_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_u_map.i[k]] += {prefix}canon_u_map.x[k]*{prefix}CPG_OSQP_Grad.du[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        if parameter_canon.p_id_to_changes['P']:
            f.write(f'  for(j=0; j<{parameter_canon.p["P"].nnz}; j++){{\n')
            f.write(f'      for(k={prefix}canon_P_map.p[j]; k<{prefix}canon_P_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_P_map.i[k]] += {prefix}canon_P_map.x[k]*{prefix}CPG_OSQP_Grad.dP->x[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        if parameter_canon.p_id_to_changes['A']:
            f.write(f'  for(j=0; j<{parameter_canon.p["A"].nnz}; j++){{\n')
            f.write(f'      for(k={prefix}canon_A_map.p[j]; k<{prefix}canon_A_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_A_map.i[k]] += {prefix}canon_A_map.x[k]*{prefix}CPG_OSQP_Grad.dA->x[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        f.write('  // Reset dx\n')
        f.write(f'  for(i=0; i<{n}; i++){{\n')
        f.write(f'      {prefix}CPG_OSQP_Grad.dx[i] = 0.0;\n')
        f.write('  }\n')
        f.write('  // Reset flags for outdated canonical parameters\n')
        for p_id, changes in parameter_canon.p_id_to_changes.items():
            if changes:
                f.write(f'  {prefix}Canon_Outdated_Grad.{p_id} = 0;\n')
        f.write('}\n\n')
            
            
    def write_gradient_prot(self, f, configuration,
                            variable_info_first, dual_variable_info_first,
                            variable_info_second, dual_variable_info_second,
                            parameter_info, parameter_canon, solver_interface):
        """
        Write function declarations to file
        """
        
        if configuration.gradient_two_stage:
            prefix = f'gradient_{configuration.prefix}'
        else:
            prefix = configuration.prefix

        write_description(f, 'c', 'Function declarations')
        f.write('#include "cpg_workspace.h"\n')
        #f.write('#include "osqp_api_types.h"\n')
        
        if configuration.gradient_two_stage:
            ret_prim_func_needed = solver_interface.ret_prim_func_exists(variable_info_second)
            if ret_prim_func_needed:
                f.write(f'\nextern void {prefix}cpg_retrieve_intermediate_primal();\n')
            f.write(f'\nextern void {prefix}cpg_retrieve_intermediate_dual();\n')
            if ret_prim_func_needed:
                write_vec_prot(f, np.zeros(self.n_var), f'{prefix}sol_x', 'cpg_float')
            else:
                f.write(f'\nextern cpg_float* {prefix}sol_x;\n')
            write_vec_prot(f, np.zeros(self.n_eq + self.n_ineq), f'{prefix}sol_y', 'cpg_float')
        
        f.write('\n// Un-retrieve\n')
        for name, size in variable_info_first.name_to_size.items():
            if size == 1:
                f.write(f'extern void {configuration.prefix}cpg_update_d{name}(cpg_float val);\n')
            else:
                f.write(f'extern void {configuration.prefix}cpg_update_d{name}(cpg_int idx, cpg_float val);\n')
        
        f.write('\n// End-to-end gradient\n')
        f.write(f'extern void {configuration.prefix}cpg_gradient();\n')
        
        
    def write_gradient_workspace_def(self, f, prefix, parameter_canon):
        """
        Write canonical gradient workspace to file
        """
        
        n = parameter_canon.p['P'].shape[0]
        N = n + parameter_canon.p['A'].shape[0]
        
        K = sp.bmat([
            [parameter_canon.p['P'] + 1e-6 * sp.eye(n), parameter_canon.p['A'].T],
            [None, - 1e-6 * sp.eye(N-n)]
        ], format='csc')
        
        K_true = sp.bmat([
            [parameter_canon.p['P'], parameter_canon.p['A'].T],
            [parameter_canon.p['A'], None]
        ], format='csr')
        
        write_description(f, 'c', 'Static workspace allocation for canonical gradient computation')
        f.write('#include "cpg_osqp_grad_workspace.h"\n\n')
        
        workspace = [
            ('a',       'int',      ones(N-n, dtype=int)),
            ('etree',   'int',      zeros(N, dtype=int)),
            ('Lnz',     'int',      zeros(N, dtype=int)),
            ('iwork',   'int',      zeros(3*N, dtype=int)),
            ('bwork',   'int',      zeros(N, dtype=int)),
            ('fwork',   'float',    zeros(N)),
            ('L',       'csc_L',    N),
            ('Lmask',   'int',      zeros((N-1)*N//2, dtype=int)),
            ('D',       'float',    ones(N)),
            ('Dinv',    'float',    ones(N)),
            ('K',       'csc',      K),
            ('K_true',  'csc',      K_true),
            ('rhs',     'float',    zeros(N)),
            ('delta',   'float',    zeros(N)),
            ('c',       'float',    zeros(N)),
            ('w',       'float',    zeros(N)),
            ('wi',      'int',      np.arange(N)),
            ('l',       'float',    zeros(N)),
            ('li',      'int',      np.arange(N)),
            ('lx',      'float',    zeros(N)),
            ('dx',      'float',    zeros(n)),
            ('r',       'float',    zeros(N)),
            ('dq',      'float',    zeros(n)),
            ('dl',      'float',    zeros(N-n)),
            ('du',      'float',    zeros(N-n)),
            ('dP',      'csc',      0*parameter_canon.p['P']),
            ('dA',      'csc',      0*parameter_canon.p['A'])
        ]
        
        for name, typ, value in workspace:
            if typ == 'csc':
                write_mat_def(f, value, f'{prefix}cpg_osqp_grad_{name}', qualifier='grad')
            elif typ == 'csc_L':
                write_L_def(f, value, f'{prefix}cpg_osqp_grad_{name}', qualifier='grad')
            else:
                write_vec_def(f, value, f'{prefix}cpg_osqp_grad_{name}', 'cpg_' + typ, qualifier='grad')
                    
        OSQP_Grad_fiels = ['init'] + [w[0] for w in workspace]
        OSQP_Grad_casts = [''] + [f'{type_to_cast(w[1], qualifier="grad")}&' for w in workspace]
        OSQP_Grad_values = ['1'] + [f'{prefix}cpg_osqp_grad_{v}' for v in OSQP_Grad_fiels[1:]]
        write_struct_def(f, OSQP_Grad_fiels, OSQP_Grad_casts, OSQP_Grad_values, f'{prefix}CPG_OSQP_Grad', 'CPG_OSQP_Grad_t')
            
            
    def generate_solver_code(self, solver_code_dir, parameter_canon):
        import osqp

        # OSQP codegen
        osqp_obj = osqp.OSQP()
        osqp_obj.setup(P=parameter_canon.p['P'], q=parameter_canon.p['q'],
                    A=parameter_canon.p['A'], l=parameter_canon.p['l'],
                    u=parameter_canon.p['u'])

        osqp_obj.codegen(solver_code_dir, parameters='matrices', force_rewrite=True)
            
    
    def generate_gradient_code(self, code_dir, cvxpygen_directory, parameter_canon, prefix, two_stage):
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'grad', 'cpg_osqp_grad_compute.c'),
                    os.path.join(code_dir, 'c', 'src'))
        replacements = [
            ('$n$', str(self.n_var)),
            ('$N$', str(self.n_var + self.n_eq + self.n_ineq)),
            ('$workspace$', f'{prefix}CPG_OSQP_Grad')
        ]
        if two_stage:
            replacements.extend([
                ('#include "qdldl.h', '#include "cpg_gradient.h"\n#include "qdldl.h'),
                ('sol_x', f'{prefix}sol_x'),
                ('sol_y', f'{prefix}sol_y')
            ])
        read_write_file(os.path.join(code_dir, 'c', 'src', 'cpg_osqp_grad_compute.c'),
                        lambda x: multiple_replace(x, replacements))
        for f in ['cpg_osqp_grad_compute.h', 'cpg_osqp_grad_workspace.h']:
            shutil.copy(os.path.join(cvxpygen_directory, 'template', 'grad', f),
                        os.path.join(code_dir, 'c', 'include'))
        read_write_file(os.path.join(code_dir, 'c', 'include', 'cpg_osqp_grad_workspace.h'),
                        lambda x: x.replace('$workspace$', f'{prefix}CPG_OSQP_Grad'))
        write_file(os.path.join(code_dir, 'c', 'src', 'cpg_osqp_grad_workspace.c'), 'w', 
                    self.write_gradient_workspace_def, 
                    prefix, parameter_canon)


    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                  parameter_canon: ParameterCanon, gradient, prefix) -> None:
        
        self.generate_solver_code(solver_code_dir, parameter_canon)

        # copy / generate source files
        if gradient:
            self.generate_gradient_code(code_dir, cvxpygen_directory, parameter_canon, prefix, configuration.gradient_two_stage)
            
        # copy license files
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'osqp-python', 'LICENSE'),
                        os.path.join(solver_code_dir, 'LICENSE'))
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'LICENSE'), code_dir)

        # adjust workspace.h
        read_write_file(os.path.join(solver_code_dir, 'workspace.h'),
                        lambda x: x.replace('extern OSQPSolver solver;',
                                            'extern OSQPSolver solver;\n'
                                            + f'  extern OSQPFloat sol_x[{self.n_var}];\n'
                                            + f'  extern OSQPFloat sol_y[{self.n_eq + self.n_ineq}];'))

        # modify CMakeLists.txt
        read_write_file(os.path.join(solver_code_dir, 'CMakeLists.txt'),
                        lambda x: cut_from_expr(x, 'add_library').replace('src', '${CMAKE_CURRENT_SOURCE_DIR}/src'))
        
        main_folder = os.path.split(solver_code_dir)[-1]
        
        # adjust top-level CMakeLists.txt
        sdir = f'${{CMAKE_CURRENT_SOURCE_DIR}}/{main_folder}'
        indent = ' ' * 6
        read_write_file(os.path.join(code_dir, 'c', 'CMakeLists.txt'),
                        lambda x: x.replace('${CMAKE_CURRENT_SOURCE_DIR}/solver_code/include',
                                            '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/include\n'
                                            + indent + sdir + '\n'
                                            + indent + sdir + '/inc/public\n'
                                            + indent + sdir + '/inc/private'))
        
        # adjust setup.py
        indent = ' ' * 30
        read_write_file(os.path.join(code_dir, 'setup.py'),
                        lambda x: x.replace("os.path.join('cpp', 'include'),",
                                            "os.path.join('cpp', 'include'),\n" +
                                            indent + f"os.path.join('c', '{main_folder}'),\n" +
                                            indent + f"os.path.join('c', '{main_folder}', 'inc', 'public'),\n" + 
                                            indent + f"os.path.join('c', '{main_folder}', 'inc', 'private'),"))

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
            affine_map.mapping = param_prob.q[[-1], :]
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
    supports_gradient = False
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
    stgs = {
        'normalize': Setting(type='cpg_int', default='1'),
        'scale': Setting(type='cpg_float', default='0.1'),
        'adaptive_scale': Setting(type='cpg_int', default='1'),
        'rho_x': Setting(type='cpg_float', default='1e-6'),
        'max_iters': Setting(type='cpg_int', default='1e5'),
        'eps_abs': Setting(type='cpg_float', default='1e-4'),
        'eps_rel': Setting(type='cpg_float', default='1e-4'),
        'eps_infeas': Setting(type='cpg_float', default='1e-7'),
        'alpha': Setting(type='cpg_float', default='1.5'),
        'time_limit_secs': Setting(type='cpg_float', default='0'),
        'verbose': Setting(type='cpg_int', default='0'),
        'warm_start': Setting(type='cpg_int', default='0'),
        'acceleration_lookback': Setting(type='cpg_int', default='0'),
        'acceleration_interval': Setting(type='cpg_int', default='1'),
        'write_data_filename': Setting(type='const char*', default='SCS_NULL'),
        'log_csv_filename': Setting(type='const char*', default='SCS_NULL')
    }
    
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

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                  parameter_canon: ParameterCanon, gradient, prefix) -> None:

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
        read_write_file(os.path.join(solver_code_dir, 'CMakeLists.txt'),
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
    supports_gradient = False
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
    stgs = {
        'feastol': Setting(type='cpg_float', default='1e-8'),
        'abstol': Setting(type='cpg_float', default='1e-8'),
        'reltol': Setting(type='cpg_float', default='1e-8'),
        'feastol_inacc': Setting(type='cpg_float', default='1e-4'),
        'abstol_inacc': Setting(type='cpg_float', default='5e-5'),
        'reltol_inacc': Setting(type='cpg_float', default='5e-5'),
        'maxit': Setting(type='cpg_int', default='100', name_cvxpy='max_iters')
    }

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
                function_call=f'{{prefix}}cpg_copy_all();\n'
                            f'    {{prefix}}ecos_workspace = ECOS_setup({canon_constants["n"]}, {canon_constants["m"]}, {canon_constants["p"]}, {canon_constants["l"]}, {canon_constants["n_cones"]}'
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
                function_call=f'{{prefix}}cpg_copy_all();\n'
                            f'      ECOS_updateData({{prefix}}ecos_workspace, {{prefix}}Canon_Params_conditioning.G->x, {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.A->x"}'
                            f', {{prefix}}Canon_Params_conditioning.c, {{prefix}}Canon_Params_conditioning.h, {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.b"})'
            ),
            'c': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic(['c']),
                function_call=f'{{prefix}}cpg_copy_c();\n'
                            f'        for (i=0; i<{canon_constants["n"]}; i++) {{{{ ecos_updateDataEntry_c({{prefix}}ecos_workspace, i, {{prefix}}Canon_Params_conditioning.c[i]); }}}}'
            ),
            'h': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic(['h']),
                function_call=f'{{prefix}}cpg_copy_h();\n'
                            f'        for (i=0; i<{canon_constants["m"]}; i++) {{{{ ecos_updateDataEntry_h({{prefix}}ecos_workspace, i, {{prefix}}Canon_Params_conditioning.h[i]); }}}}'
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

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                  parameter_canon: ParameterCanon, gradient, prefix) -> None:

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
    gradient_supported = False

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
    stgs = {
        'max_iter': Setting(type='cpg_int', default='50', name_cvxpy='max_iters'),
        'time_limit': Setting(type='cpg_float', default='1e6'),
        'verbose': Setting(type='cpg_int', default='1'),
        'max_step_fraction': Setting(type='cpg_float', default='0.99'),
        'equilibrate_enable': Setting(type='cpg_int', default='1'),
        'equilibrate_max_iter': Setting(type='cpg_int', default='10'),
        'equilibrate_min_scaling': Setting(type='cpg_float', default='1e-4'),
        'equilibrate_max_scaling': Setting(type='cpg_float', default='1e4')
    }

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
            update_after_init = ['A', 'q', 'b']
        else:
            extra_condition = '!{prefix}solver'
            P_p = '{prefix}Canon_Params_conditioning.P->p'
            P_i = '{prefix}Canon_Params_conditioning.P->i'
            P_x = '{prefix}Canon_Params_conditioning.P->x'
            update_after_init = ['P', 'A', 'q', 'b']

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition=extra_condition, functions_if_false=update_after_init),
                function_call= \
                    f'{{prefix}}cpg_copy_all();\n'
                    f'    clarabel_CscMatrix_init(&{{prefix}}P, {canon_constants["n"]}, {canon_constants["n"]}, {P_p}, {P_i}, {P_x});\n'
                    f'    clarabel_CscMatrix_init(&{{prefix}}A, {canon_constants["m"]}, {canon_constants["n"]}, {{prefix}}Canon_Params_conditioning.A->p, {{prefix}}Canon_Params_conditioning.A->i, {{prefix}}Canon_Params_conditioning.A->x);\n' \
                    f'    {{prefix}}settings = clarabel_DefaultSettings_default()'
            ),
            'A': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['A']),
                function_call = f'{{prefix}}cpg_copy_A();\n      clarabel_CscMatrix_init(&{{prefix}}A, {canon_constants["m"]}, {canon_constants["n"]}, {{prefix}}Canon_Params_conditioning.A->p, {{prefix}}Canon_Params_conditioning.A->i, {{prefix}}Canon_Params_conditioning.A->x)'
            ),
            'q': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['q']),
                function_call = f'{{prefix}}cpg_copy_q()'
            ),
            'b': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['b']),
                function_call = f'{{prefix}}cpg_copy_b()'
            ),
        }
        
        if indices_obj is not None:
            self.parameter_update_structure['P'] = ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['P']),
                function_call = f'{{prefix}}cpg_copy_P();\n      clarabel_CscMatrix_init(&{{prefix}}P, {canon_constants["n"]}, {canon_constants["n"]}, {P_p}, {P_i}, {P_x})'
            )

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

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                    parameter_canon: ParameterCanon, gradient, prefix) -> None:

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

        # remove examples target from Clarabel.cpp/CMakeLists.txt and set build type to Release
        replacements = [
            ('add_subdirectory(examples)', '# add_subdirectory(examples)'),
            ('set(CMAKE_C_STANDARD_REQUIRED True)', 'set(CMAKE_C_STANDARD_REQUIRED True)\n\n# set build type to Release\nset(CMAKE_BUILD_TYPE Release)')
        ]
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, replacements))

        # add sdp flag
        if is_sdp:
            read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                            lambda x: x.replace('set(CLARABEL_FEATURE_SDP "none"', 'set(CLARABEL_FEATURE_SDP "sdp-openblas"'))

        # adjust Clarabel.cpp/rust_wrapper/CMakeLists.txt
        replacements = [
            ('${CMAKE_SOURCE_DIR}/', '${CMAKE_SOURCE_DIR}/solver_code/'),
            ('/libclarabel_c.lib', '/clarabel_c.lib'),  # until fixed on Clarabel side
            (
                'set(clarabel_c_output_directory "${CMAKE_SOURCE_DIR}/solver_code/rust_wrapper/target/release")',
                'if (ARM64)\n'
                '        message(STATUS "ARM64 detected")\n'
                '        set(clarabel_c_output_directory "${CMAKE_SOURCE_DIR}/solver_code/rust_wrapper/target/aarch64-apple-darwin/release")\n'
                '    else()\n'
                '        set(clarabel_c_output_directory "${CMAKE_SOURCE_DIR}/solver_code/rust_wrapper/target/release")\n'
                '    endif()'
            ),
            (
                '# Add the cargo project as a custom target',
                '# Add the cargo project as a custom target\n'
                'if(ARM64)\n'
                '   set(clarabel_c_build_flags "${clarabel_c_build_flags};--target;aarch64-apple-darwin")\n'
                'endif()'
            )
        ]
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'rust_wrapper', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, replacements))

        # adjust Clarabel
        read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'Clarabel'),
                        lambda x: x.replace('cpp/', 'c/'))

        # adjust setup.py
        release_dir = "'aarch64-apple-darwin/release'" if platform.system() == "Darwin" and platform.machine() == "arm64" else "'release'"
        read_write_file(os.path.join(code_dir, 'setup.py'),
                        lambda x: x.replace("extra_objects=[cpg_lib])",
                                            f"extra_objects=[cpg_lib, os.path.join(cpg_dir, 'solver_code', 'rust_wrapper', 'target', {release_dir}, 'libclarabel_c.a')])"))

    
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

class QOCOGENInterface(SolverInterface):
    solver_name = 'QOCOGEN'
    solver_type = 'conic'
    canon_p_ids = ['P', 'c', 'd', 'A', 'b', 'G', 'h']
    canon_p_ids_constr_vec = ['b', 'h']
    solve_function_call = 'qoco_custom_solve(&{prefix}qoco_custom_workspace)'

    # header files
    header_files = ['"qoco_custom.h"']
    cmake_headers = ['${qoco_custom_headers}']
    cmake_sources = ['${qoco_custom_sources}']

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'qoco_custom_workspace.sol.obj',
        iterations = 'qoco_custom_workspace.sol.iters',
        status = 'qoco_custom_workspace.sol.status',
        primal_residual = 'qoco_custom_workspace.sol.pres',
        dual_residual = 'qoco_custom_workspace.sol.dres',
        primal_solution = 'qoco_custom_workspace.sol.x',
        dual_solution = 'qoco_custom_workspace.sol.{dual_var_name}',
        settings = 'qoco_custom_workspace.settings.{setting_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'double', 'int': 'int'}

    # solver settings
    stgs_dynamically_allocated = False
    stgs_requires_extra_struct_type = False
    stgs_direct_write_ptr = '(&({prefix}qoco_custom_workspace.settings))'
    stgs_translation = "{}"
    stgs_reset_function = {'name': 'set_default_settings', 'ptr': '&{prefix}qoco_custom_workspace'}
    stgs = {
        'max_iters': Setting(type='cpg_int', default='200'),
        'bisect_iters': Setting(type='cpg_int', default='5'),
        'ruiz_iters': Setting(type='cpg_int', default='0'),
        'iter_ref_iters': Setting(type='cpg_int', default='1'),
        'kkt_static_reg': Setting(type='cpg_float', default='1e-8'),
        'kkt_dynamic_reg': Setting(type='cpg_float', default='1e-8'),
        'abstol': Setting(type='cpg_float', default='1e-7'),
        'reltol': Setting(type='cpg_float', default='1e-7'),
        'abstol_inacc': Setting(type='cpg_float', default='1e-5'),
        'reltol_inacc': Setting(type='cpg_float', default='1e-5'),
        'verbose': Setting(type='cpg_int', default='0')
    }
    
    dual_var_split = True
    dual_var_names = ['y', 'z']

    # docu
    docu = 'https://qoco-org.github.io/qoco/'

    def __init__(self, data, p_prob, enable_settings):
        n_var = p_prob.x.size
        n_eq = p_prob.cone_dims.zero
        n_ineq = data['G'].shape[0]
        q = np.array(p_prob.cone_dims.soc)
        indices_obj, indptr_obj, shape_obj = self.get_problem_data_index(p_prob.reduced_P)
        indices_constr, indptr_constr, shape_constr = self.get_problem_data_index(p_prob.reduced_A)

        canon_constants = {'n': n_var, 'm': n_ineq, 'p': n_eq,
                           'l': p_prob.cone_dims.nonneg,
                           'nsoc': len(p_prob.cone_dims.soc),
                           'q': q}

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition='{prefix}qoco_custom_workspace.n <= 0'),
                function_call=f'load_data(&{{prefix}}qoco_custom_workspace)'
            ),
            'A': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['A']),
                function_call='update_A(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.A->x)'
            ),
            'G': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['G']),
                function_call='update_G(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.G->x)'
            ),
            'c': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['c']),
                function_call = 'update_c(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.c)'
            ),
            'b': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['b']),
                function_call = 'update_b(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.b)'
            ),
            'h': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['h']),
                function_call = 'update_h(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.h)'
            ),
        }
        if indices_obj is not None:
            self.parameter_update_structure['P'] = ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['P']),
                function_call = 'update_P(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.P->x)'
            )

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0:
            raise ValueError(
                'Exponential cones is not supported for QOCOGEN.')

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                  parameter_canon: ParameterCanon, gradient, prefix) -> None:
        import qocogen

        # Need to check if problem has quadratic objective.
        P = parameter_canon.p['P'] if 'P' in parameter_canon.p.keys() else None

        # Generate qoco_custom
        qocogen.generate_solver(self.canon_constants['n'], self.canon_constants['m'], self.canon_constants['p'], P, 
                                parameter_canon.p['c'], parameter_canon.p['A'], parameter_canon.p['b'], parameter_canon.p['G'], parameter_canon.p['h'],
                                self.canon_constants['l'], self.canon_constants['nsoc'], self.canon_constants['q'],
                                os.path.join(code_dir, 'c'), "solver_code")
        
        # Copy LICENSE
        shutil.copyfile(os.path.join(code_dir, 'c', 'solver_code', 'LICENSE'),
                os.path.join(code_dir, 'LICENSE'))

        # adjust top-level CMakeLists.txt
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/'
        cmake_replacements = [
            (sdir + 'include',
            sdir)
        ]
        read_write_file(os.path.join(code_dir, 'c', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, cmake_replacements))

        # adjust setup.py
        setup_replacements = [
            ("os.path.join('c', 'solver_code', 'include'),",
            "os.path.join('c', 'solver_code'),"),
            ("license='Apache 2.0'", "license='BSD 3-Clause'")
        ]
        read_write_file(os.path.join(code_dir, 'setup.py'),
                        lambda x: multiple_replace(x, setup_replacements))


    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['nsoc'] > 0:
            f.write('\n// qoco_custom array of SOC dimensions\n')
            write_vec_prot(f, self.canon_constants['q'], f'{prefix}qoco_custom_q', 'cpg_int')
        f.write('\n// qoco_custom workspace\n')
        f.write(f'extern Workspace {prefix}qoco_custom_workspace;\n')
        f.write('\n// qoco_custom exit flag\n')
        f.write(f'extern cpg_int {prefix}qoco_custom_flag;\n')

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['nsoc'] > 0:
            f.write('\n// qoco_custom array of SOC dimensions\n')
            write_vec_def(f, self.canon_constants['q'], f'{prefix}qoco_custom_q', 'cpg_int')
        f.write('\n// qoco_custom workspace\n')
        f.write(f'Workspace {prefix}qoco_custom_workspace;\n')
        f.write('\n// qoco_custom exit flag\n')
        f.write(f'cpg_int {prefix}qoco_custom_flag = -99;\n')

class QOCOInterface(SolverInterface):
    solver_name = 'QOCO'
    solver_type = 'conic'
    canon_p_ids = ['P', 'c', 'd', 'A', 'b', 'G', 'h']
    canon_p_ids_constr_vec = ['b', 'h']
    supports_gradient = False
    solve_function_call = 'qoco_solve({prefix}qoco_solver)'

    # header files
    header_files = ['"qoco.h"']
    cmake_headers = ['${qoco_headers}']
    cmake_sources = ['${qoco_sources}']

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = True

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'qoco_solver->sol->obj',
        iterations = 'qoco_solver->sol->iters',
        status = 'qoco_solver->sol->status',
        primal_residual = 'qoco_solver->sol->pres',
        dual_residual = 'qoco_solver->sol->dres',
        primal_solution = 'qoco_solver->sol->x',
        dual_solution = 'qoco_solver->sol->{dual_var_name}',
        settings = 'qoco_solver->settings->{setting_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = False

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'QOCOFloat', 'int': 'QOCOInt'}

    # solver settings
    stgs_dynamically_allocated = True
    stgs_requires_extra_struct_type = True
    stgs_direct_write_ptr = None
    stgs_reset_function = None
    stgs = {
        'max_iters': Setting(type='cpg_int', default='200'),
        'bisect_iters': Setting(type='cpg_int', default='5'),
        'ruiz_iters': Setting(type='cpg_int', default='0'),
        'iter_ref_iters': Setting(type='cpg_int', default='1'),
        'kkt_static_reg': Setting(type='cpg_float', default='1e-8'),
        'kkt_dynamic_reg': Setting(type='cpg_float', default='1e-8'),
        'abstol': Setting(type='cpg_float', default='1e-7'),
        'reltol': Setting(type='cpg_float', default='1e-7'),
        'abstol_inacc': Setting(type='cpg_float', default='1e-5'),
        'reltol_inacc': Setting(type='cpg_float', default='1e-5'),
        'verbose': Setting(type='cpg_int', default='0')
    }

    # dual variables split into y and z vectors
    dual_var_split = True
    dual_var_names = ['y', 'z']

    # docu
    docu = 'https://qoco-org.github.io/qoco/'

    def __init__(self, data, p_prob, enable_settings):
        n_var = p_prob.x.size
        n_eq = p_prob.cone_dims.zero
        n_ineq = data['G'].shape[0]
        q = np.array(p_prob.cone_dims.soc)
        indices_obj, indptr_obj, shape_obj = self.get_problem_data_index(p_prob.reduced_P)
        indices_constr, indptr_constr, shape_constr = self.get_problem_data_index(p_prob.reduced_A)

        canon_constants = {'n': n_var, 'm': n_ineq, 'p': n_eq,
                           'l': p_prob.cone_dims.nonneg,
                           'nsoc': len(p_prob.cone_dims.soc),
                           'q': q}

        function_call = (
            f'{{prefix}}cpg_copy_all();\n'
            '   QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));\n'
            '   QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));\n'
            '   QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));\n'
        )

        if indices_obj is not None:
            function_call += (
                f'   qoco_set_csc(P, {canon_constants["n"]}, {canon_constants["n"]}, '
                f'{{prefix}}Canon_Params.P->nnz, {{prefix}}Canon_Params.P->x, '
                f'{{prefix}}Canon_Params.P->p, {{prefix}}Canon_Params.P->i);\n'
            )
        else:
            function_call += '   P = NULL;\n'

        if canon_constants["p"] > 0:
            function_call += (
                f'   qoco_set_csc(A, {canon_constants["p"]}, {canon_constants["n"]}, '
                f'{{prefix}}Canon_Params.A->nnz, {{prefix}}Canon_Params.A->x, '
                f'{{prefix}}Canon_Params.A->p, {{prefix}}Canon_Params.A->i);\n'
            )
        else:
            function_call += '   A = NULL;\n'

        if canon_constants["m"] > 0:
            function_call += (
                f'   qoco_set_csc(G, {canon_constants["m"]}, {canon_constants["n"]}, '
                f'{{prefix}}Canon_Params.G->nnz, {{prefix}}Canon_Params.G->x, '
                f'{{prefix}}Canon_Params.G->p, {{prefix}}Canon_Params.G->i);\n'
            )
        else:
            function_call += '   G = NULL;\n'

        function_call += (
            '   QOCOSettings* qoco_settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));\n'
            '   set_default_settings(qoco_settings);\n'
            f'   {{prefix}}qoco_solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));\n'
            f'   qoco_setup({{prefix}}qoco_solver, {canon_constants["n"]}, {canon_constants["m"]}, {canon_constants["p"]}, P, '
            f'{{prefix}}Canon_Params.c, A, '
            f'{"NULL" if canon_constants["p"] == 0 else f"{{prefix}}Canon_Params.b"}, '
            f'G, {"NULL" if canon_constants["m"] == 0 else f"{{prefix}}Canon_Params.h"}, '
            f'{canon_constants["l"]}, {canon_constants["nsoc"]}, '
            f'{"NULL" if canon_constants["nsoc"] == 0 else f"(int *) &{{prefix}}qoco_q"}, '
            'qoco_settings)'
        )

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition='!{prefix}qoco_solver'),
                function_call=function_call
            ),
            'A': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['A']),
                function_call='update_matrix_data({prefix}qoco_solver, NULL, {prefix}Canon_Params.A->x, NULL)'
            ),
            'G': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['G']),
                function_call='update_matrix_data({prefix}qoco_solver, NULL, NULL, {prefix}Canon_Params.G->x)'
            ),
            'c': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['c']),
                function_call = 'update_vector_data({prefix}qoco_solver, {prefix}Canon_Params.c, NULL, NULL)'
            ),
            'b': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['b']),
                function_call = 'update_vector_data({prefix}qoco_solver, NULL, {prefix}Canon_Params.b, NULL)'
            ),
            'h': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['h']),
                function_call = 'update_vector_data({prefix}qoco_solver, NULL, NULL, {prefix}Canon_Params.h)'
            ),
        }
        if indices_obj is not None:
            self.parameter_update_structure['P'] = ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['P']),
                function_call = 'update_matrix_data({prefix}qoco_solver, {prefix}Canon_Params.P->x, NULL, NULL)'
            )

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0:
            raise ValueError(
                'QOCO does not support exponential cones.')

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                  parameter_canon: ParameterCanon, gradient, prefix) -> None:

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'lib', 'configure']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'qoco', dtc),
                            os.path.join(solver_code_dir, dtc))
        
        files_to_copy = ['CMakeLists.txt', 'LICENSE']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'qoco', fl),
                            os.path.join(solver_code_dir, fl))
        
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'qoco', 'LICENSE'),
                        os.path.join(code_dir, 'LICENSE'))

        # adjust top-level CMakeLists.txt
        indent = ' ' * 6
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/'
        cmake_replacements = [
            (sdir + 'include',
            sdir + 'include\n' +
            indent + sdir + 'lib/amd\n' +
            indent + sdir + 'lib/qdldl/include')
        ]
        read_write_file(os.path.join(code_dir, 'c', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, cmake_replacements))

        cmake_replacements = [
            ('add_executable (cpg_example ${cpg_head} ${cpg_src} ${CMAKE_CURRENT_SOURCE_DIR}/src/cpg_example.c)',
            'add_executable (cpg_example ${cpg_head} ${cpg_src} ${CMAKE_CURRENT_SOURCE_DIR}/src/cpg_example.c)\n' +
            'target_link_libraries(cpg_example qocostatic)')
        ]
        read_write_file(os.path.join(code_dir, 'c', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, cmake_replacements))

        cmake_replacements = [
            ('add_library (cpg STATIC ${cpg_head} ${cpg_src})',
            'add_library (cpg STATIC ${cpg_head} ${cpg_src})\n' +
            'target_link_libraries(cpg qocostatic)')
        ]
        read_write_file(os.path.join(code_dir, 'c', 'CMakeLists.txt'),
                        lambda x: multiple_replace(x, cmake_replacements))

        # adjust setup.py
        setup_replacements = [
            ("os.path.join('c', 'solver_code', 'include'),",
            "os.path.join('c', 'solver_code', 'include'),\n" +
            5 * indent + "os.path.join('c', 'solver_code', 'lib', 'amd'),\n" +
            5 * indent + "os.path.join('c', 'solver_code', 'lib', 'qdldl', 'include'),"),
            ("license='Apache 2.0'", "license='BSD 3-Clause'"),
            ("lib_name = 'cpg.lib'", "lib_name = os.path.join('cpg.lib')\n" + "    libqoco_name = os.path.join('Release', 'qocostatic.lib')\n" + "    libqdldl_name = os.path.join('Release', 'qdldl.lib')"),
            ("lib_name = 'libcpg.a'", "lib_name = 'libcpg.a'\n" + "    libqoco_name = 'libqocostatic.a'\n" + "    libqdldl_name = 'libqdldl.a'"),
            ('extra_objects=[cpg_lib]', "extra_objects=[cpg_lib, os.path.join(cpg_dir, 'build', 'out', libqoco_name), os.path.join(cpg_dir, 'build', 'solver_code', 'lib', 'qdldl', 'out', libqdldl_name)]")
        ]
        read_write_file(os.path.join(code_dir, 'setup.py'),
                        lambda x: multiple_replace(x, setup_replacements))


    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['nsoc'] > 0:
            f.write('\n// QOCO array of SOC dimensions\n')
            write_vec_prot(f, self.canon_constants['q'], f'{prefix}qoco_q', 'cpg_int')
        f.write('\n// QOCO workspace\n')
        f.write(f'extern QOCOSolver* {prefix}qoco_solver;\n')
        f.write('\n// QOCO exit flag\n')
        f.write(f'extern cpg_int {prefix}qoco_flag;\n')

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['nsoc'] > 0:
            f.write('\n// QOCO array of SOC dimensions\n')
            write_vec_def(f, self.canon_constants['q'], f'{prefix}qoco_q', 'cpg_int')
        f.write('\n// QOCO solver\n')
        f.write(f'QOCOSolver* {prefix}qoco_solver = NULL;\n')
        f.write('\n// QOCO exit flag\n')
        f.write(f'cpg_int {prefix}qoco_flag = -99;\n')
