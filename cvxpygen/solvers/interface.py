"""
Copyright 2023-2026 Maximilian Schaller
Licensed under the Apache License, Version 2.0
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np

from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, ConstraintInfo, AffineMap, \
    ParameterCanon


UNSUPPORTED_ON_WINDOWS = ['CLARABEL']


class QPCanonMixin:
    """
    Mixin for solvers that use the OSQP QP canonical form (P, q, d, A, l, u).

    Provides the shared constructor, get_affine_map, and augment_vector_parameter
    implementations for both OSQPInterface and PDAQPInterface.
    """

    solver_type = 'quadratic'
    canon_p_ids = ['P', 'q', 'd', 'A', 'l', 'u']
    canon_p_ids_constr_vec = ['l', 'u']
    dual_var_split = False
    dual_var_names = ['y']

    def __init__(self, data, p_prob, enable_settings):
        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, p_prob, {}, enable_settings)

    def get_affine_map(self, p_id, param_prob, constraint_info: ConstraintInfo) -> AffineMap:
        affine_map = AffineMap()

        if p_id == 'P':
            if self.indices_obj is None:  # problem is an LP
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
                mapping_rows_eq >= constraint_info.n_data_constr_mat]
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
            raise ValueError(f'Unknown QP parameter name {p_id}.')

        return affine_map

    def augment_vector_parameter(self, p_id, vector_parameter):
        if p_id == 'l':
            return np.concatenate((vector_parameter, -np.inf * np.ones(self.n_ineq)), axis=0)
        return vector_parameter


class SolverInterface(ABC):

    supports_quad_obj: bool = False
    cvxpy_solver_name: str = None  # CVXPY solver string used for get_problem_data

    def __init__(self, solver_name, n_var, n_eq, n_ineq, p_prob, canon_constants, enable_settings):
        
        self.solver_name = solver_name
        self.n_var = n_var
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        
        self.indices_obj, self.indptr_obj, self.shape_obj = self.get_problem_data_index(p_prob.reduced_P)
        self.indices_constr, self.indptr_constr, self.shape_constr = self.get_problem_data_index(p_prob.reduced_A)

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
        return any(variable_info.sym) or any([s == 1 for s in variable_info.name_to_size.values()])

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return any([s == 1 for s in dual_variable_info.name_to_size.values()])

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
            affine_map.mapping = param_prob.q[:-1]
        elif p_id == 'd':
            affine_map.mapping = param_prob.q[[-1], :]
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
