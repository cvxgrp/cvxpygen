"""
Copyright 2026 Maximilian Schaller
Licensed under the Apache License, Version 2.0
"""

import copy
import platform
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
import cvxpy as cp
from cvxpy.problems.objective import Maximize
from cvxpy.cvxcore.python import canonInterface as cI
from cvxpy.atoms.affine.upper_tri import upper_tri_to_full
from cvxpy.reductions import InverseData
from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

from cvxpygen.mappings import (
    Canon, ParameterCanon, ParameterInfo, PrimalVariableInfo,
    DualVariableInfo, ConstraintInfo
)

from cvxpygen.solvers import SolverInterface, OSQPInterface, SCSInterface, ECOSInterface, \
    ClarabelInterface, QOCOInterface, QOCOGENInterface, PDAQPInterface, UNSUPPORTED_ON_WINDOWS


class Canonicalizer:
    """
    Transforms a cp.Problem into a Canon object (solver-ready canonical form).

    Owns all extraction logic: primal/dual variable info, parameter info,
    affine mappings, and the two-stage gradient canonicalization path.
    """

    def __init__(
        self,
        solver: str,
        solver_opts: Optional[dict] = None,
        enable_settings: List[str] = [],
    ) -> None:
        self.solver = solver
        self.solver_opts = solver_opts
        self.enable_settings = enable_settings

    def canonicalize(self, problem: cp.Problem) -> Tuple[Canon, SolverInterface]:
        """
        Canonicalize problem. Returns (canon, solver_interface).
        For the two-stage gradient case, returns the merged canon.
        """
        return self._extract(problem, self.solver, self.solver_opts, self.enable_settings)

    def canonicalize_two_stage(self, problem: cp.Problem) -> Tuple[Canon, Canon, Canon, SolverInterface, SolverInterface]:
        """
        Perform two-stage canonicalization for gradient computation with non-OSQP solvers.
        Returns (merged_canon, canon_gradient, canon_solver, solver_interface, gradient_interface).
        """
        canon_gradient, gradient_interface = self._extract(problem, 'OSQP', None, [])
        osqp_problem = self._get_osqp_problem(canon_gradient)
        canon_solver, solver_interface = self._extract(
            osqp_problem, self.solver, self.solver_opts, self.enable_settings
        )
        merged = self._merge(canon_gradient, canon_solver, solver_interface)
        return merged, canon_gradient, canon_solver, solver_interface, gradient_interface

    # ── private helpers ───────────────────────────────────────────────────────
    
    def _get_interface_class(self, solver_name: str) -> "SolverInterface":
        if platform.system() == 'Windows' and solver_name.upper() in UNSUPPORTED_ON_WINDOWS:
            raise ValueError(f'{solver_name} solver currently not supported on Windows.')
        mapping = {
            'OSQP': OSQPInterface,
            'SCS': SCSInterface,
            'ECOS': ECOSInterface,
            'CLARABEL': ClarabelInterface,
            'QOCO': QOCOInterface,
            'QOCOGEN': QOCOGENInterface,
            'PDAQP': PDAQPInterface,
        }
        interface = mapping.get(solver_name.upper(), None)
        if interface is None:
            raise ValueError(f'Unsupported solver: {solver_name}.')
        return interface

    def _extract(self, problem, solver, solver_opts, enable_settings) -> Tuple[Canon, object]:
        interface_class = self._get_interface_class(solver)

        data, _, inverse_data = problem.get_problem_data(
            solver=interface_class.cvxpy_solver_name,
            gp=False,
            enforce_dpp=True,
            verbose=False,
            solver_opts=solver_opts,
        )
        param_prob = data['param_prob']

        if not param_prob.parameters:
            raise ValueError('Solution does not depend on parameters. Aborting code generation.')

        if hasattr(param_prob, 'cone_dims'):
            interface_class.check_unsupported_cones(param_prob.cone_dims)

        solver_interface = interface_class(data, param_prob, enable_settings)
        solver_interface._problem = problem
        solver_interface._solver_opts = solver_opts
        prim_variable_info = self._get_primal_variable_info(problem, inverse_data)
        dual_variable_info = self._get_dual_variable_info(inverse_data, solver_interface)
        parameter_info = self._get_parameter_info(param_prob)
        constraint_info = self._get_constraint_info(solver_interface)

        adjacency, parameter_canon, canon_p_ids = self._process_canonical_parameters(
            constraint_info, param_prob, parameter_info,
            solver_interface, solver_opts, problem,
        )

        parameter_canon.user_p_name_to_canon_outdated = {
            user_p_name: [canon_p_ids[j] for j in np.nonzero(adjacency[:, i])[0]]
            for i, user_p_name in enumerate(parameter_info.names)
        }

        return Canon(prim_variable_info, dual_variable_info, parameter_info, parameter_canon), solver_interface

    def _get_primal_variable_info(self, problem, inverse_data) -> PrimalVariableInfo:
        variables = problem.variables()
        var_names = [var.name() for var in variables]
        var_ids = [var.id for var in variables]
        var_offsets = [inverse_data[-2].var_offsets[var_id] for var_id in var_ids]
        var_name_to_offset = {n: o for n, o in zip(var_names, var_offsets)}
        var_shapes = [var.shape for var in variables]
        var_sizes = [var.size for var in variables]
        var_sym = [
            var.attributes['symmetric'] or var.attributes['PSD'] or var.attributes['NSD']
            for var in variables
        ]
        var_name_to_sym = {n: s for n, s in zip(var_names, var_sym)}
        var_name_to_indices = {}
        for var_name, offset, shape, sym in zip(var_names, var_offsets, var_shapes, var_sym):
            if sym:
                fill_coefficient = upper_tri_to_full(shape[0])
                (_, col) = fill_coefficient.nonzero()
                var_name_to_indices[var_name] = offset + col
            else:
                var_name_to_indices[var_name] = np.arange(offset, offset + np.prod(shape))
        var_name_to_size = {var.name(): var.size for var in variables}
        var_name_to_shape = {var.name(): var.shape for var in variables}
        var_name_to_init = {}
        for var in variables:
            if len(var.shape) == 0:
                var_name_to_init[var.name()] = 0
            else:
                var_name_to_init[var.name()] = np.zeros(shape=var.shape)

        return PrimalVariableInfo(
            var_name_to_offset, var_name_to_indices, var_name_to_size,
            var_name_to_shape, var_name_to_init,
            var_name_to_sym, var_sym,
        )

    def _get_dual_variable_info(self, inverse_data, solver_interface) -> DualVariableInfo:
        dual_id_maps = []
        for inv in inverse_data:
            if isinstance(inv, InverseData) and not isinstance(inv, SolverInverseData):
                dual_id_maps.append(inv.cons_id_map)
            if isinstance(inv, tuple) and len(inv) == 3:
                dual_id_maps.append(inv[2])

        dual_ids = []
        for dual_id in dual_id_maps[0].keys():
            for dual_id_map in dual_id_maps[1:]:
                dual_id = dual_id_map[dual_id]
            dual_ids.append(dual_id)

        if solver_interface.solver_type == 'quadratic':
            con_canon = inverse_data[-2].constraints
        elif solver_interface.solver_type == 'conic':
            con_canon = (
                inverse_data[-1][ConicSolver.EQ_CONSTR]
                + inverse_data[-1][ConicSolver.NEQ_CONSTR]
            )
        con_canon_dict = {c.id: c for c in con_canon}
        d_canon_offsets = np.cumsum([0] + [c.size for c in con_canon[:-1]])

        if solver_interface.dual_var_split:
            n_split = len(inverse_data[-1][ConicSolver.EQ_CONSTR])
            d_canon_vectors = (
                [solver_interface.dual_var_names[0]] * n_split
                + [solver_interface.dual_var_names[1]] * (len(d_canon_offsets) - n_split)
            )
            d_canon_offsets[n_split:] -= d_canon_offsets[n_split]
        else:
            d_canon_vectors = solver_interface.dual_var_names * len(d_canon_offsets)

        d_canon_offsets_dict = {c.id: off for c, off in zip(con_canon, d_canon_offsets)}
        d_canon_vectors_dict = {c.id: v for c, v in zip(con_canon, d_canon_vectors)}

        d_offsets = [d_canon_offsets_dict[i] for i in dual_ids]
        d_vectors = [d_canon_vectors_dict[i] for i in dual_ids]
        d_sizes = [con_canon_dict[i].size for i in dual_ids]
        d_shapes = [
            con_canon_dict[i].shape
            if np.prod(con_canon_dict[i].shape) == con_canon_dict[i].size
            else None
            for i in dual_ids
        ]
        d_names = [f'd{i}' for i in range(len(dual_ids))]
        d_i_to_name = {i: f'd{i}' for i in range(len(dual_ids))}
        d_name_to_size = {n: s for n, s in zip(d_names, d_sizes)}
        d_name_to_shape = {n: d_shapes[i] for i, n in d_i_to_name.items()}
        d_name_to_indices = {
            n: (v, o + np.arange(d_name_to_size[n]))
            for n, v, o in zip(d_names, d_vectors, d_offsets)
        }
        d_name_to_vec = {n: v for n, v in zip(d_names, d_vectors)}
        d_name_to_offset = {n: o for n, o in zip(d_names, d_offsets)}

        d_name_to_init = {}
        for name, size in d_name_to_size.items():
            d_name_to_init[name] = 0 if size == 1 else np.zeros(size)

        return DualVariableInfo(
            d_name_to_offset, d_name_to_indices, d_name_to_size,
            d_name_to_shape, d_name_to_init, d_name_to_vec,
        )

    def _get_parameter_info(self, param_prob) -> ParameterInfo:
        user_p_num = len(param_prob.parameters)
        user_p_names = [par.name() for par in param_prob.parameters]
        user_p_ids = list(param_prob.param_id_to_col.keys())
        user_p_id_to_col = param_prob.param_id_to_col
        user_p_id_to_size = param_prob.param_id_to_size
        user_p_id_to_param = param_prob.id_to_param
        user_p_total_size = param_prob.total_param_size

        user_p_name_to_shape = {
            user_p_id_to_param[p_id].name(): user_p_id_to_param[p_id].shape
            for p_id in user_p_id_to_size.keys()
        }
        user_p_name_to_size_usp = {
            user_p_id_to_param[p_id].name(): size
            for p_id, size in user_p_id_to_size.items()
        }
        user_p_col_to_name_usp = {}
        cum_sum = 0
        for name, size in user_p_name_to_size_usp.items():
            user_p_col_to_name_usp[cum_sum] = name
            cum_sum += size

        user_p_writable = {}
        for p_name, p in zip(user_p_names, param_prob.parameters):
            if p.value is None:
                p.project_and_assign(np.random.randn(*p.shape))
                if type(p.value) in [sparse.dia_matrix, sparse.dia_array]:
                    p.value = p.value.toarray()
            if len(p.shape) < 2:
                user_p_writable[p_name] = p.value
            else:
                user_p_writable[p_name] = p.value.flatten(order='F')

        def user_p_value(user_p_id):
            return np.array(user_p_id_to_param[user_p_id].value)

        user_p_flat_usp = cI.get_parameter_vector(
            user_p_total_size, user_p_id_to_col, user_p_id_to_size, user_p_value
        )

        return ParameterInfo(
            user_p_col_to_name_usp, user_p_flat_usp, user_p_id_to_col,
            user_p_ids, user_p_name_to_shape, user_p_name_to_size_usp,
            user_p_names, user_p_num, user_p_writable, None, None,
        )

    def _get_constraint_info(self, solver_interface) -> ConstraintInfo:
        n_data_constr = len(solver_interface.indices_constr)
        n_data_constr_vec = (
            solver_interface.indptr_constr[-1] - solver_interface.indptr_constr[-2]
        )
        n_data_constr_mat = n_data_constr - n_data_constr_vec
        mapping_rows_eq = np.nonzero(solver_interface.indices_constr < solver_interface.n_eq)[0]
        mapping_rows_ineq = np.nonzero(solver_interface.indices_constr >= solver_interface.n_eq)[0]
        return ConstraintInfo(n_data_constr, n_data_constr_mat, mapping_rows_eq, mapping_rows_ineq)

    def _process_canonical_parameters(
        self, constraint_info, param_prob, parameter_info,
        solver_interface, solver_opts, problem,
    ):
        parameter_canon = ParameterCanon()
        parameter_canon.quad_obj = self._get_quad_obj(
            problem, solver_interface, solver_opts
        )

        if not parameter_canon.quad_obj:
            canon_p_ids = [p_id for p_id in solver_interface.canon_p_ids if p_id != 'P']
        else:
            canon_p_ids = solver_interface.canon_p_ids

        adjacency = np.zeros((len(canon_p_ids), parameter_info.num), dtype=bool)

        for i, p_id in enumerate(canon_p_ids):
            affine_map = solver_interface.get_affine_map(p_id, param_prob, constraint_info)

            if affine_map:
                if p_id in solver_interface.canon_p_ids_constr_vec:
                    affine_map = self._update_to_dense_mapping(affine_map, param_prob)

                if len(affine_map.mapping.shape) < 2:
                    affine_map.mapping = affine_map.mapping.reshape(1, -1)
                affine_map.mapping = affine_map.mapping.tocsr()

                if p_id == 'd':
                    parameter_canon.nonzero_d = affine_map.mapping.nnz > 0

                adjacency = self._update_adjacency_matrix(
                    adjacency, i, parameter_info, affine_map.mapping
                )
                affine_map.mapping = sparse.csc_matrix(
                    affine_map.mapping.toarray() * affine_map.sign
                )
                affine_map, parameter_canon = self._set_default_values(
                    affine_map, p_id, parameter_canon, parameter_info, solver_interface
                )

                parameter_canon.p_id_to_mapping[p_id] = affine_map.mapping.tocsr()
                parameter_canon.p_id_to_changes[p_id] = affine_map.mapping[:, :-1].nnz > 0
                parameter_canon.p_id_to_size[p_id] = affine_map.mapping.shape[0]
            else:
                parameter_canon.p_id_to_mapping[p_id] = None
                parameter_canon.p_id_to_changes[p_id] = False
                parameter_canon.p_id_to_size[p_id] = 0

        parameter_canon.is_maximization = isinstance(problem.objective, Maximize)
        return adjacency, parameter_canon, canon_p_ids

    def _get_osqp_problem(self, canon_osqp) -> cp.Problem:
        p = canon_osqp.parameter_canon.p
        n_eq = canon_osqp.parameter_canon.p_id_to_size['l']

        if canon_osqp.parameter_canon.p_id_to_changes['P']:
            raise ValueError(
                'Problem does not follow extended DPP rules for differentiation with general solvers '
                '(other than OSQP). Quadratics cannot be multiplied with parameters.'
            )

        row, col = p['P'].nonzero()
        assert all(row == col)

        osqp_x = cp.Variable(p['q'].shape, name='osqp_x')
        osqp_q = cp.Parameter(p['q'].shape, name='osqp_q')
        osqp_A = cp.Parameter(p['A'].shape, name='osqp_A', sparsity=p['A'].nonzero())
        osqp_l = cp.Parameter((n_eq,), name='osqp_l')
        osqp_u = cp.Parameter(p['u'].shape, name='osqp_u')
        osqp_P_diag_sqrt = np.sqrt(np.diag(p['P'].todense()))

        return cp.Problem(
            cp.Minimize(
                0.5 * cp.sum_squares(cp.multiply(osqp_x, osqp_P_diag_sqrt)) + osqp_q @ osqp_x
            ),
            [osqp_l <= osqp_A[:n_eq, :] @ osqp_x, osqp_A @ osqp_x <= osqp_u],
        )

    def _merge(self, canon_first, canon_second, solver_interface) -> Canon:
        pc_first = canon_first.parameter_canon
        pc_second = canon_second.parameter_canon
        n_user_param = canon_first.parameter_info.flat_usp.size
        ret_prim_func_needed = solver_interface.ret_prim_func_exists(
            canon_second.prim_variable_info
        )

        pc = ParameterCanon()
        pc.is_maximization = pc_first.is_maximization
        pc.nonzero_d = pc_first.nonzero_d
        pc.quad_obj = pc_second.quad_obj
        pc.p = pc_second.p
        pc.p_csc = pc_second.p_csc
        pc.p_id_to_size = pc_second.p_id_to_size
        pc.p_id_to_changes = pc_second.p_id_to_changes

        pc.p_id_to_mapping = {}
        map_first = []
        for col in sorted(canon_second.parameter_info.col_to_name_usp):
            name = canon_second.parameter_info.col_to_name_usp[col][5:]
            map_first.append(pc_first.p_id_to_mapping[name])
        map_first.append(self._create_constant_map(1, n_user_param, 1.))
        map_first = sparse.vstack(map_first)
        for p_id, map_second in pc_second.p_id_to_mapping.items():
            pc.p_id_to_mapping[p_id] = map_second @ map_first

        pc.user_p_name_to_canon_outdated = {}
        for user_p_name, canon_p_ids in pc_first.user_p_name_to_canon_outdated.items():
            canon_outdated = []
            for p_id in canon_p_ids:
                canon_outdated.extend(
                    pc_second.user_p_name_to_canon_outdated[f'osqp_{p_id}']
                )
            pc.user_p_name_to_canon_outdated[user_p_name] = list(set(canon_outdated))

        pvi = copy.deepcopy(canon_first.prim_variable_info)
        if not ret_prim_func_needed:
            offset_second = canon_second.prim_variable_info.name_to_offset['osqp_x']
            for name in pvi.name_to_offset.keys():
                pvi.name_to_offset[name] += offset_second
                pvi.name_to_indices[name] += offset_second

        return Canon(pvi, canon_first.dual_variable_info, canon_first.parameter_info, pc)

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _create_constant_map(n, m, val):
        data = np.full(n, val)
        rows = np.arange(n)
        cols = np.full(n, m - 1)
        return sparse.csc_matrix((data, (rows, cols)), shape=(n, m))

    @staticmethod
    def _get_quad_obj(problem, solver_interface, solver_opts) -> bool:
        if solver_interface.solver_type == 'quadratic':
            return True
        use_quad_obj = solver_opts.get('use_quad_obj', True) if solver_opts else True
        return (
            use_quad_obj
            and solver_interface.supports_quad_obj
            and problem.objective.expr.has_quadratic_term()
        )

    @staticmethod
    def _update_to_dense_mapping(affine_map, param_prob):
        mapping_to_sparse = param_prob.reduced_A.reduced_mat[affine_map.mapping_rows]
        dense_shape = (affine_map.shape[0], mapping_to_sparse.shape[1])
        mapping_to_dense = sparse.lil_matrix(np.zeros(dense_shape))
        for i_data, sparse_row in enumerate(mapping_to_sparse):
            mapping_to_dense[affine_map.indices[i_data], :] = sparse_row
        affine_map.mapping = sparse.csr_matrix(mapping_to_dense)
        return affine_map

    @staticmethod
    def _update_adjacency_matrix(adjacency, i, parameter_info, mapping) -> np.ndarray:
        for j in range(parameter_info.num):
            column_slice = slice(
                parameter_info.id_to_col[parameter_info.ids[j]],
                parameter_info.id_to_col[parameter_info.ids[j + 1]],
            )
            adjacency[i, j] = mapping[:, column_slice].nnz > 0
        return adjacency

    def _set_default_values(self, affine_map, p_id, parameter_canon, parameter_info, solver_interface):
        if p_id.isupper():
            rows_nonzero, _ = affine_map.mapping.nonzero()
            canon_p_data_nonzero = np.sort(np.unique(rows_nonzero))
            affine_map.mapping = affine_map.mapping[canon_p_data_nonzero, :]
            canon_p_data = affine_map.mapping @ parameter_info.flat_usp
            if solver_interface.dual_var_split:
                if p_id == 'P':
                    affine_map.indptr = solver_interface.indptr_obj
                else:
                    indptr_original = solver_interface.indptr_constr[:-1]
                    affine_map.indptr = 0 * indptr_original
                for r in affine_map.mapping_rows:
                    for c in range(affine_map.shape[1]):
                        if indptr_original[c] <= r < indptr_original[c + 1]:
                            affine_map.indptr[c + 1:] += 1
                            break
            else:
                if p_id == 'P':
                    affine_map.indptr = solver_interface.indptr_obj
                elif p_id == 'A':
                    affine_map.indptr = solver_interface.indptr_constr[:-1]
            indices_usp = affine_map.indices[canon_p_data_nonzero]
            indptr_usp = 0 * affine_map.indptr
            for r in canon_p_data_nonzero:
                for c in range(affine_map.shape[1]):
                    if affine_map.indptr[c] <= r < affine_map.indptr[c + 1]:
                        indptr_usp[c + 1:] += 1
                        break
            csc_mat = sparse.csc_matrix(
                (canon_p_data, indices_usp, indptr_usp), shape=affine_map.shape
            )
            parameter_canon.p[p_id] = csc_mat
        else:
            parameter_canon.p[p_id] = solver_interface.augment_vector_parameter(
                p_id,
                affine_map.mapping @ parameter_info.flat_usp,
            )
        return affine_map, parameter_canon
