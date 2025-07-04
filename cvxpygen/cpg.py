"""
Copyright 2022 Maximilian Schaller
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
import sys
import shutil
import pickle
import warnings
import copy
import importlib

from cvxpygen import utils, mpqp
from cvxpygen.utils import write_file, read_write_file, write_example_def, write_module_prot, write_module_def, \
    write_canon_cmake, write_method, replace_cmake_data, replace_setup_data, replace_html_data
from cvxpygen.mappings import Configuration, PrimalVariableInfo, DualVariableInfo, ConstraintInfo, \
    ParameterCanon, ParameterInfo, Canon
from cvxpygen.solvers import get_interface_class
import cvxpy as cp
import numpy as np
from scipy import sparse
from subprocess import call
from cvxpy.problems.objective import Maximize
from cvxpy.cvxcore.python import canonInterface as cI
from cvxpy.atoms.affine.upper_tri import upper_tri_to_full


def generate_code(problem, code_dir='cpg_code', solver=None, solver_opts=None,
                  enable_settings=[], unroll=False, prefix='', wrapper=True, gradient=False):
    """
    Generate C code to solve a CVXPY problem
    """
    sys.stdout.write('Generating code with CVXPYgen ...\n')
    
    create_folder_structure(code_dir)
    
    # get solver and explicit flag
    solver, explicit = get_solver_and_explicit_flag(solver, solver_opts)
    
    # in explicit mode, check that gradient computation is not requested
    if explicit and gradient:
        raise ValueError('Explicit mode: Gradient computation is not supported!')
    
    gradient_two_stage = gradient and solver != 'OSQP'

    # extract canonicalization
    if gradient_two_stage:
        # first-stage canonicalization from user problem to OSQP problem
        canon_gradient, gradient_interface = extract_canonicalization(problem, 'OSQP', None, [])
        # create parametrized OSQP problem
        osqp_problem = get_osqp_problem(canon_gradient)
        # second-stage canonicalization from OSQP problem to solver problem
        canon_solver, solver_interface = extract_canonicalization(osqp_problem, solver, solver_opts, enable_settings)
        # chain two stages of canonicalization into one for solving
        canon = merge_canonicalizations(canon_gradient, canon_solver, solver_interface)
    else:
        # default behavior, single canonicalization
        canon_gradient, canon_solver = None, None
        canon, solver_interface = extract_canonicalization(problem, solver, solver_opts, enable_settings)
        gradient_interface = solver_interface

    configuration = get_configuration(code_dir, solver, unroll, prefix, gradient, gradient_two_stage, explicit)
    cvxpygen_directory = os.path.dirname(os.path.realpath(__file__))
    solver_code_dir = os.path.join(code_dir, 'c', 'solver_code')
    osqp_code_dir = os.path.join(code_dir, 'c', 'osqp_code')
    if gradient and solver != 'OSQP':
        solver_interface.generate_code(configuration, code_dir, solver_code_dir, cvxpygen_directory, canon.parameter_canon, False, configuration.prefix)
        gradient_interface.generate_code(configuration, code_dir, osqp_code_dir, cvxpygen_directory, canon_gradient.parameter_canon, True, f'gradient_{configuration.prefix}')
    else:
        cvxpygen_directory = os.path.dirname(os.path.realpath(__file__))
        solver_code_dir = os.path.join(code_dir, 'c', 'solver_code')
        if explicit:
            mpqp.offline_solve_and_codegen_explicit(problem, canon, solver_code_dir, solver_opts, explicit)
        else:
            solver_interface.generate_code(configuration, code_dir, solver_code_dir, cvxpygen_directory, canon.parameter_canon, gradient, configuration.prefix)
    
    write_c_code(problem, configuration, canon, canon_gradient, canon_solver, solver_interface, gradient_interface)

    sys.stdout.write('CVXPYgen finished generating code.\n')
    
    if wrapper:
        compile_python_module(code_dir)
        module = importlib.import_module(f'{code_dir}.cpg_solver')
        cpg_solve = getattr(module, 'cpg_solve')
        problem.register_solve('CPG', cpg_solve)
        
        
def get_solver_and_explicit_flag(solver, solver_opts):
    if solver is None:
        return None, 0
    elif solver.lower() == 'explicit':
        if solver_opts and solver_opts.get('dual', False):
            return 'OSQP', 2
        else:
            return 'OSQP', 1
    else:
        return solver, 0
        
        
def extract_canonicalization(problem, solver, solver_opts, enable_settings) -> Canon:
    
    # interface class
    interface_class, cvxpy_interface_class = get_interface_class(solver)
    
    # problem data
    data, _, inverse_data = problem.get_problem_data(
        solver=cvxpy_interface_class.__name__,
        gp=False,
        enforce_dpp=True,
        verbose=False,
        solver_opts=solver_opts
    )
    param_prob = data['param_prob']
    
    if not param_prob.parameters:
        raise ValueError('Solution does not depend on parameters. Aborting code generation.')

    # cone problems check
    if hasattr(param_prob, 'cone_dims'):
        cone_dims = param_prob.cone_dims
        interface_class.check_unsupported_cones(cone_dims)

    handle_sparsity(param_prob)

    solver_interface = interface_class(data, param_prob, enable_settings)  # noqa
    prim_variable_info = get_primal_variable_info(problem, inverse_data)
    dual_variable_info = get_dual_variable_info(inverse_data, solver_interface, cvxpy_interface_class)
    parameter_info = get_parameter_info(param_prob)
    constraint_info = get_constraint_info(solver_interface)

    adjacency, parameter_canon, canon_p_ids = process_canonical_parameters(
        constraint_info, param_prob, parameter_info, solver_interface, solver_opts, problem, cvxpy_interface_class
    )

    parameter_canon.user_p_name_to_canon_outdated = {
        user_p_name: [canon_p_ids[j] for j in np.nonzero(adjacency[:, i])[0]]
        for i, user_p_name in enumerate(parameter_info.names)
    }

    return Canon(prim_variable_info, dual_variable_info, parameter_info, parameter_canon), solver_interface
    

def get_osqp_problem(canon_osqp) -> cp.Problem:
    
    p = canon_osqp.parameter_canon.p
    n_eq = canon_osqp.parameter_canon.p_id_to_size['l']
    
    # assert that quadratic form is not parametric
    if canon_osqp.parameter_canon.p_id_to_changes['P']:
        raise ValueError('Problem does not follow extended DPP rules for differentiation with general solvers (other than OSQP). '
                         'Quadratics cannot be multiplied with parameters.')
    
    # assert that P is diagonal
    row, col = p['P'].nonzero()
    assert all(row == col)
    
    osqp_x = cp.Variable(p['q'].shape, name='osqp_x')
    
    osqp_q = cp.Parameter(p['q'].shape, name='osqp_q')
    osqp_A_rows, osqp_A_cols = p['A'].nonzero()
    osqp_A = cp.Parameter(p['A'].shape, name='osqp_A', sparsity=list(zip(osqp_A_rows, osqp_A_cols)))
    osqp_l = cp.Parameter((n_eq,), name='osqp_l')
    osqp_u = cp.Parameter(p['u'].shape, name='osqp_u')
    
    osqp_P_diag_sqrt = np.sqrt(np.diag(p['P'].todense()))
    
    osqp_problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(cp.multiply(osqp_x, osqp_P_diag_sqrt)) + osqp_q @ osqp_x),
        [osqp_l <= osqp_A[:n_eq, :] @ osqp_x, osqp_A @ osqp_x <= osqp_u]
    )
    
    return osqp_problem


def merge_canonicalizations(canon_first, canon_second, solver_interface) -> Canon:
    
    pc_first = canon_first.parameter_canon
    pc_second = canon_second.parameter_canon
    
    n_user_param = canon_first.parameter_info.flat_usp.size    
    ret_prim_func_needed = solver_interface.ret_prim_func_exists(canon_second.prim_variable_info)
    
    pc = ParameterCanon()
    
    pc.is_maximization = pc_first.is_maximization
    pc.nonzero_d = pc_first.nonzero_d
    pc.quad_obj = pc_second.quad_obj
    
    pc.p = pc_second.p
    pc.p_csc = pc_second.p_csc
    pc.p_id_to_size = pc_second.p_id_to_size
    pc.p_id_to_changes = pc_second.p_id_to_changes # TODO: consider cases where q, A, l, or u do not change
    
    # figure out in which order q, A, l, u are flattened and concatenated, for example (flat(A), q, l, u)
    # for very pc_second.p_id_to_mapping[p_id], multiply with [pc_first.p_id_to_mapping['A']; ...]
    pc.p_id_to_mapping = {}
    map_first = []
    for col in sorted(canon_second.parameter_info.col_to_name_usp):
        name = canon_second.parameter_info.col_to_name_usp[col][5:]
        map_first.append(pc_first.p_id_to_mapping[name])
    map_first.append(create_constant_map(1, n_user_param, 1.))
    map_first = sparse.vstack(map_first)
    for p_id, map_second in pc_second.p_id_to_mapping.items():
        pc.p_id_to_mapping[p_id] = map_second @ map_first
    
    pc.user_p_name_to_canon_outdated = {}
    for user_p_name, canon_p_ids in pc_first.user_p_name_to_canon_outdated.items():
        canon_outdated = []
        for p_id in canon_p_ids:
            canon_outdated.extend(pc_second.user_p_name_to_canon_outdated[f'osqp_{p_id}'])
        pc.user_p_name_to_canon_outdated[user_p_name] = list(set(canon_outdated))
        
    pvi = copy.deepcopy(canon_first.prim_variable_info)
    if not ret_prim_func_needed:
        offset_second = canon_second.prim_variable_info.name_to_offset['osqp_x']
        for name in pvi.name_to_offset.keys():
            pvi.name_to_offset[name] += offset_second
            pvi.name_to_indices[name] += offset_second
    
    return Canon(pvi, canon_first.dual_variable_info, canon_first.parameter_info, pc)
    

def create_constant_map(n, m, val):
    data = np.full(n, val)
    rows = np.arange(n)
    cols = np.full(n, m-1)
    return sparse.csc_matrix((data, (rows, cols)), shape=(n, m))


def get_quad_obj(problem, solver_type, solver_opts, solver_class) -> bool:
    
    if solver_type == 'quadratic':
        return True
    
    use_quad_obj = solver_opts.get('use_quad_obj', True) if solver_opts else True

    return use_quad_obj and solver_class().supports_quad_obj() and problem.objective.expr.has_quadratic_term()


def process_canonical_parameters(
        constraint_info, param_prob, parameter_info, 
        solver_interface, solver_opts, problem, cvxpy_interface_class):
    
    parameter_canon = ParameterCanon()
    parameter_canon.quad_obj = get_quad_obj(
        problem, solver_interface.solver_type, solver_opts, cvxpy_interface_class
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
                affine_map = update_to_dense_mapping(affine_map, param_prob)
            
            if len(affine_map.mapping.shape) < 2:
                affine_map.mapping = affine_map.mapping.reshape(1, -1)
            affine_map.mapping = affine_map.mapping.tocsr()
            
            if p_id == 'd':
                parameter_canon.nonzero_d = affine_map.mapping.nnz > 0

            adjacency = update_adjacency_matrix(adjacency, i, parameter_info, affine_map.mapping)
            affine_map.mapping = sparse.csc_matrix(affine_map.mapping.toarray() * affine_map.sign)
            affine_map.mapping = affine_map.mapping[:, parameter_info.sparsity_mask]
            affine_map, parameter_canon = set_default_values(affine_map, p_id, parameter_canon, parameter_info, solver_interface)
            
            parameter_canon.p_id_to_mapping[p_id] = affine_map.mapping.tocsr()
            parameter_canon.p_id_to_changes[p_id] = affine_map.mapping[:, :-1].nnz > 0
            parameter_canon.p_id_to_size[p_id] = affine_map.mapping.shape[0]
        else:
            parameter_canon.p_id_to_mapping[p_id] = None
            parameter_canon.p_id_to_changes[p_id] = False
            parameter_canon.p_id_to_size[p_id] = 0
    
    parameter_canon.is_maximization = isinstance(problem.objective, Maximize)

    return adjacency, parameter_canon, canon_p_ids


def update_to_dense_mapping(affine_map, param_prob):

    # Extract the sparse matrix and prepare a zero-initialized dense matrix
    mapping_to_sparse = param_prob.reduced_A.reduced_mat[affine_map.mapping_rows]
    dense_shape = (affine_map.shape[0], mapping_to_sparse.shape[1])
    mapping_to_dense = sparse.lil_matrix(np.zeros(dense_shape))

    # Update dense mapping with data from sparse mapping
    for i_data, sparse_row in enumerate(mapping_to_sparse):
        mapping_to_dense[affine_map.indices[i_data], :] = sparse_row

    # Convert to Compressed Sparse Column format and update mapping
    affine_map.mapping = sparse.csr_matrix(mapping_to_dense)
    
    return affine_map


def set_default_values(affine_map, p_id, parameter_canon, parameter_info, solver_interface):
    if p_id.isupper():
        rows_nonzero, _ = affine_map.mapping.nonzero()
        canon_p_data_nonzero = np.sort(np.unique(rows_nonzero))
        affine_map.mapping = affine_map.mapping[canon_p_data_nonzero, :]
        canon_p_data = affine_map.mapping @ parameter_info.flat_usp
        # compute 'indptr' to construct sparse matrix from 'canon_p_data' and 'indices'
        if solver_interface.dual_var_split: # by equality and inequality
            if p_id == 'P':
                affine_map.indptr = solver_interface.indptr_obj
            else:
                indptr_original = solver_interface.indptr_constr[:-1]
                affine_map.indptr = 0 * indptr_original # rebuild 'indptr' by considering only 'mapping_rows' (corresponds to either equality or inequality)
            for r in affine_map.mapping_rows:
                for c in range(affine_map.shape[1]): # shape of matrix re-shaped from flat param vector resulting from mapping
                    if indptr_original[c] <= r < indptr_original[c + 1]:
                        affine_map.indptr[c + 1:] += 1
                        break
        else:
            if p_id == 'P':
                affine_map.indptr = solver_interface.indptr_obj
            elif p_id == 'A':
                affine_map.indptr = solver_interface.indptr_constr[:-1] # leave out part for rhs
        # compute 'indices_usp' and 'indptr_usp' (usp = user-defined sparsity)
        indices_usp = affine_map.indices[canon_p_data_nonzero]
        indptr_usp = 0 * affine_map.indptr
        for r in canon_p_data_nonzero:
            for c in range(affine_map.shape[1]):
                if affine_map.indptr[c] <= r < affine_map.indptr[c + 1]:
                    indptr_usp[c + 1:] += 1
                    break
        csc_mat = sparse.csc_matrix((canon_p_data, indices_usp, indptr_usp),
                                    shape=affine_map.shape)
        parameter_canon.p[p_id] = csc_mat
    else:
        parameter_canon.p[p_id] = solver_interface.augment_vector_parameter(
            p_id,
            affine_map.mapping @ parameter_info.flat_usp
            )

    return affine_map, parameter_canon


def get_primal_variable_info(problem, inverse_data) -> PrimalVariableInfo:
    variables = problem.variables()
    var_names = [var.name() for var in variables]
    var_ids = [var.id for var in variables]
    var_offsets = [inverse_data[-2].var_offsets[var_id] for var_id in var_ids]
    var_name_to_offset = {n: o for n, o in zip(var_names, var_offsets)}
    var_shapes = [var.shape for var in variables]
    var_sizes = [var.size for var in variables]
    var_sym = [var.attributes['symmetric'] or var.attributes['PSD'] or var.attributes['NSD'] for var
               in variables]
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
    var_name_to_init = dict()
    for var in variables:
        if len(var.shape) == 0:
            var_name_to_init[var.name()] = 0
        else:
            var_name_to_init[var.name()] = np.zeros(shape=var.shape)

    prim_variable_info = PrimalVariableInfo(var_name_to_offset, var_name_to_indices, var_name_to_size,
                                            var_sizes, var_name_to_shape, var_name_to_init,
                                            var_name_to_sym, var_sym)
    return prim_variable_info


def get_dual_variable_info(inverse_data, solver_interface, cvxpy_interface_class) -> DualVariableInfo:
    
    # get chain of constraint id maps for 'CvxAttr2Constr' and 'Canonicalization' objects
    dual_id_maps = []
    if solver_interface.solver_type == 'quadratic':
        if inverse_data[-4]:
            dual_id_maps.append(inverse_data[-4][2])
        dual_id_maps.append(inverse_data[-3].cons_id_map)
    elif solver_interface.solver_type == 'conic':
        dual_id_maps.append(inverse_data[-4].cons_id_map)
        if inverse_data[-3]:
            dual_id_maps.append(inverse_data[-3][2])
        dual_id_maps.append(inverse_data[-2].cons_id_map)
    
    # recurse chain of constraint ids to get ordered list of constraint ids
    dual_ids = []
    for dual_id in dual_id_maps[0].keys():
        for dual_id_map in dual_id_maps[1:]:
            dual_id = dual_id_map[dual_id]
        dual_ids.append(dual_id)
    
    # get canonical constraint information
    if solver_interface.solver_type == 'quadratic':
        con_canon = inverse_data[-2].constraints  # same order as in canonical dual vector
    elif solver_interface.solver_type == 'conic':
        con_canon = inverse_data[-1][cvxpy_interface_class.EQ_CONSTR] + inverse_data[-1][cvxpy_interface_class.NEQ_CONSTR]
    con_canon_dict = {c.id: c for c in con_canon}
    d_canon_offsets = np.cumsum([0] + [c.args[0].size for c in con_canon[:-1]])
    if solver_interface.dual_var_split:
        n_split = len(inverse_data[-1][cvxpy_interface_class.EQ_CONSTR])
        d_canon_vectors = [solver_interface.dual_var_names[0]] * n_split + [solver_interface.dual_var_names[1]] * (len(d_canon_offsets) - n_split)
        d_canon_offsets[n_split:] -= d_canon_offsets[n_split]
    else:
        d_canon_vectors = solver_interface.dual_var_names * len(d_canon_offsets)
    d_canon_offsets_dict = {c.id: off for c, off in zip(con_canon, d_canon_offsets)}
    d_canon_vectors_dict = {c.id: v for c, v in zip(con_canon, d_canon_vectors)}
    
    # select for user-defined constraints
    d_offsets = [d_canon_offsets_dict[i] for i in dual_ids]
    d_vectors = [d_canon_vectors_dict[i] for i in dual_ids]
    d_sizes = [con_canon_dict[i].size for i in dual_ids]
    d_shapes = [con_canon_dict[i].shape for i in dual_ids]
    d_names = [f'd{i}' for i in range(len(dual_ids))]
    d_i_to_name = {i: f'd{i}' for i in range(len(dual_ids))}
    d_name_to_shape = {n: d_shapes[i] for i, n in d_i_to_name.items()}
    d_name_to_indices = {n: (v, o + np.arange(np.prod(d_name_to_shape[n])))
                         for n, v, o in zip(d_names, d_vectors, d_offsets)}
    d_name_to_vec = {n: v for n, v in zip(d_names, d_vectors)}
    d_name_to_offset = {n: o for n, o in zip(d_names, d_offsets)}
    d_name_to_size = {n: s for n, s in zip(d_names, d_sizes)}

    # initialize values to zero
    d_name_to_init = dict()
    for name, shape in d_name_to_shape.items():
        if len(shape) == 0:
            d_name_to_init[name] = 0
        else:
            d_name_to_init[name] = np.zeros(shape=shape)

    dual_variable_info = DualVariableInfo(d_name_to_offset, d_name_to_indices, d_name_to_size,
                                          d_sizes, d_name_to_shape, d_name_to_init, d_name_to_vec)
    return dual_variable_info


def get_constraint_info(solver_interface) -> ConstraintInfo:

    n_data_constr = len(solver_interface.indices_constr)
    n_data_constr_vec = (solver_interface.indptr_constr[-1] 
                         - solver_interface.indptr_constr[-2])
    n_data_constr_mat = n_data_constr - n_data_constr_vec

    # Obtain rows related to equalities and inequalities
    mapping_rows_eq = np.nonzero(solver_interface.indices_constr 
                                 < solver_interface.n_eq)[0]
    mapping_rows_ineq = np.nonzero(solver_interface.indices_constr 
                                   >= solver_interface.n_eq)[0]

    return ConstraintInfo(n_data_constr, n_data_constr_mat, 
                          mapping_rows_eq, mapping_rows_ineq)


def update_adjacency_matrix(adjacency, i, parameter_info, mapping) -> np.ndarray:
    
    # Iterate through parameters and update adjacency if there are non-zero entries in mapping
    for j in range(parameter_info.num):
        column_slice = slice(parameter_info.id_to_col[parameter_info.ids[j]],
                             parameter_info.id_to_col[parameter_info.ids[j + 1]])
        # Update adjacency matrix if there are non-zero entries in the mapped slice
        adjacency[i, j] = mapping[:, column_slice].nnz > 0
    
    return adjacency


def write_c_code(problem: cp.Problem, configuration: Configuration,
                 canon: Canon, canon_first: Canon, canon_second: Canon,
                 solver_interface, gradient_interface) -> None:

    prim_variable_info = canon.prim_variable_info
    dual_variable_info = canon.dual_variable_info
    parameter_info = canon.parameter_info
    parameter_canon = canon.parameter_canon

    # Simplified directory and file access
    c_dir = os.path.join(configuration.code_dir, 'c')
    cpp_dir = os.path.join(configuration.code_dir, 'cpp')
    include_dir = os.path.join(c_dir, 'include')
    src_dir = os.path.join(c_dir, 'src')
    solver_code_dir = os.path.join(c_dir, 'solver_code')
    osqp_code_dir = os.path.join(c_dir, 'osqp_code')
    
    # write files
    # if two-stage gradient is used, write main workspace and solve files without gradient stuff
    if configuration.gradient_two_stage:
        primal_solution_ptr = solver_interface.ws_ptrs.primal_solution
        dual_solution_ptr = solver_interface.ws_ptrs.dual_solution
        # poin to intermediate primal solution if it needs to be computed in second stage
        if solver_interface.ret_prim_func_exists(canon_second.prim_variable_info):
            solver_interface.ws_ptrs.primal_solution = f'gradient_{configuration.prefix}sol_x'
        # always point to intermediate dual solution because it is always computed (summed to osqp's y) in second stage
        solver_interface.ws_ptrs.dual_solution = f'gradient_{configuration.prefix}sol_y'
        parameter_canon_gradient = canon_first.parameter_canon
    else:
        parameter_canon_gradient = None
    write_file(os.path.join(include_dir, f'cpg_workspace.h'), 'w', 
                getattr(utils, f'write_workspace_prot'),
                configuration, prim_variable_info, dual_variable_info, 
                parameter_info, parameter_canon, solver_interface, True)
    
    write_file(os.path.join(src_dir, f'cpg_workspace.c'), 'w', 
                getattr(utils, f'write_workspace_def'),
                configuration, prim_variable_info, dual_variable_info, 
                parameter_info, parameter_canon, solver_interface, True)
    write_file(os.path.join(include_dir, f'cpg_solve.h'), 'w', 
                getattr(utils, f'write_solve_prot'),
                configuration, prim_variable_info, dual_variable_info, 
                parameter_info, parameter_canon, solver_interface, parameter_canon_gradient)
    
    write_file(os.path.join(src_dir, f'cpg_solve.c'), 'w', 
                getattr(utils, f'write_solve_def'),
                configuration, prim_variable_info, dual_variable_info, 
                parameter_info, parameter_canon, solver_interface, parameter_canon_gradient)
    if configuration.gradient_two_stage:
        # switch back to second-stage pointer for remainder of code generation
        solver_interface.ws_ptrs.primal_solution = primal_solution_ptr
        solver_interface.ws_ptrs.dual_solution = dual_solution_ptr
    
    if configuration.gradient:
        if configuration.gradient_two_stage:
            # write extra workspace files for gradient
            write_file(os.path.join(include_dir, 'cpg_gradient_workspace.h'), 'w', 
                    getattr(utils, 'write_workspace_prot'),
                    configuration, canon_first.prim_variable_info, canon_first.dual_variable_info, 
                    canon_first.parameter_info, canon_first.parameter_canon, gradient_interface, False)
            
            write_file(os.path.join(src_dir, 'cpg_gradient_workspace.c'), 'w', 
                    getattr(utils, 'write_workspace_def'),
                    configuration, canon_first.prim_variable_info, canon_first.dual_variable_info, 
                    canon_first.parameter_info, canon_first.parameter_canon, gradient_interface, False)
            
            # write gradient files
            write_file(os.path.join(include_dir, 'cpg_gradient.h'), 'w', 
                    getattr(gradient_interface, 'write_gradient_prot'),
                    configuration, canon_first.prim_variable_info, canon_first.dual_variable_info,
                    canon_second.prim_variable_info, canon_second.dual_variable_info,
                    canon_first.parameter_info, canon_first.parameter_canon, solver_interface)
            
            write_file(os.path.join(src_dir, 'cpg_gradient.c'), 'w', 
                    getattr(gradient_interface, 'write_gradient_def'),
                    configuration, canon_first.prim_variable_info, canon_first.dual_variable_info,
                    canon_second.prim_variable_info, canon_second.dual_variable_info,
                    canon_first.parameter_info, canon_first.parameter_canon, solver_interface)
        else:
            write_file(os.path.join(include_dir, 'cpg_gradient.h'), 'w', 
                    getattr(solver_interface, 'write_gradient_prot'),
                    configuration, prim_variable_info, dual_variable_info,
                    None, None,
                    parameter_info, parameter_canon, solver_interface)
            
            write_file(os.path.join(src_dir, 'cpg_gradient.c'), 'w', 
                    getattr(solver_interface, 'write_gradient_def'),
                    configuration, prim_variable_info, dual_variable_info,
                    None, None,
                    parameter_info, parameter_canon, solver_interface)
    
    write_file(os.path.join(src_dir, 'cpg_example.c'), 'w', 
               write_example_def, 
               configuration, prim_variable_info, dual_variable_info, parameter_info)
    
    write_file(os.path.join(cpp_dir, 'include', 'cpg_module.hpp'), 'w',
               write_module_prot,
               configuration, parameter_info, prim_variable_info, 
               dual_variable_info, solver_interface, gradient_interface)

    write_file(os.path.join(cpp_dir, 'src', 'cpg_module.cpp'), 'w',
               write_module_def,
               configuration, prim_variable_info, dual_variable_info, 
               parameter_info, solver_interface, gradient_interface)

    if not configuration.explicit:
        write_file(os.path.join(solver_code_dir, 'CMakeLists.txt'), 'a',
                write_canon_cmake,
                'solver', solver_interface)
    
    if configuration.gradient_two_stage:
        read_write_file(os.path.join(osqp_code_dir, 'CMakeLists.txt'),
                        lambda x: x.replace(
                            '${CMAKE_CURRENT_SOURCE_DIR}/src/*.c',
                            '${CMAKE_CURRENT_SOURCE_DIR}/src/qdldl*.c\n'
                            ), 
                        )
        write_file(os.path.join(osqp_code_dir, 'CMakeLists.txt'), 'a',
                   write_canon_cmake,
                   'osqp', gradient_interface)

    write_file(os.path.join(configuration.code_dir, 'cpg_solver.py'), 'w',
               write_method,
               configuration, prim_variable_info, dual_variable_info, 
               parameter_info, solver_interface, gradient_interface)

    write_file(os.path.join(configuration.code_dir, 'problem.pickle'), 'wb',
               lambda x, y: pickle.dump(y, x),
               cp.Problem(problem.objective, problem.constraints))
    
    # replace file contents
    read_write_file(os.path.join(c_dir, 'CMakeLists.txt'),
                    replace_cmake_data, 
                    configuration)
    
    read_write_file(os.path.join(configuration.code_dir, 'setup.py'),
                    replace_setup_data)

    read_write_file(os.path.join(configuration.code_dir, 'README.html'),
                    replace_html_data,
                    configuration, prim_variable_info, dual_variable_info, 
                    parameter_info, solver_interface)
    

def adjust_prefix(prefix):
    if prefix and not prefix[0].isalpha():
        prefix = '_' + prefix
    return prefix + '_' if prefix else prefix


def get_configuration(code_dir, solver_name, unroll, prefix, gradient, gradient_two_stage, explicit) -> Configuration:
    return Configuration(code_dir, solver_name, unroll, adjust_prefix(prefix), gradient, gradient_two_stage, explicit)


def get_parameter_info(p_prob) -> ParameterInfo:
    user_p_num = len(p_prob.parameters)
    user_p_names = [par.name() for par in p_prob.parameters]
    user_p_ids = list(p_prob.param_id_to_col.keys())
    user_p_id_to_col = p_prob.param_id_to_col
    user_p_id_to_size = p_prob.param_id_to_size
    user_p_id_to_param = p_prob.id_to_param
    user_p_total_size = p_prob.total_param_size
    user_p_name_to_shape = {user_p_id_to_param[p_id].name(): user_p_id_to_param[p_id].shape
                            for p_id in user_p_id_to_size.keys()}
    user_p_name_to_size_usp = {user_p_id_to_param[p_id].name(): size for p_id, size in
                               user_p_id_to_size.items()}
    user_p_name_to_sparsity = {}
    user_p_name_to_sparsity_type = {}
    user_p_sparsity_mask = np.ones(user_p_total_size + 1, dtype=bool)
    for p in p_prob.parameters:
        if p.attributes['sparsity']:
            user_p_name_to_size_usp[p.name()] = len(p.attributes['sparsity'][0])
            user_p_name_to_sparsity[p.name()] = np.sort([r + p.shape[0] * c
                                                         for r, c in zip(*p.attributes['sparsity'])])
            if p.attributes['diag']:
                user_p_name_to_sparsity_type[p.name()] = 'diag'
            else:
                user_p_name_to_sparsity_type[p.name()] = 'general'
            user_p_sparsity_mask[
            user_p_id_to_col[p.id]:user_p_id_to_col[p.id] + user_p_id_to_size[p.id]] = False
            user_p_sparsity_mask[user_p_id_to_col[p.id] + user_p_name_to_sparsity[p.name()]] = True
    user_p_col_to_name_usp = {}
    cum_sum = 0
    for name, size in user_p_name_to_size_usp.items():
        user_p_col_to_name_usp[cum_sum] = name
        cum_sum += size
    user_p_writable = dict()
    for p_name, p in zip(user_p_names, p_prob.parameters):
        if p.value is None:
            p.project_and_assign(np.random.randn(*p.shape))
            if type(p.value) in [sparse.dia_matrix, sparse.dia_array]:
                p.value = p.value.toarray()
        if len(p.shape) < 2:
            # dealing with scalar or vector
            user_p_writable[p_name] = p.value
        else:
            # dealing with matrix
            if p_name in user_p_name_to_sparsity.keys():
                dense_value = p.value.flatten(order='F')
                sparse_value = np.zeros(len(user_p_name_to_sparsity[p_name]))
                for i, idx in enumerate(user_p_name_to_sparsity[p_name]):
                    sparse_value[i] = dense_value[idx]
                user_p_writable[p_name] = sparse_value
            else:
                user_p_writable[p_name] = p.value.flatten(order='F')

    def user_p_value(user_p_id):
        """
        Returns the value of the parameter with the given ID.
        """
        return np.array(user_p_id_to_param[user_p_id].value)

    user_p_flat = cI.get_parameter_vector(user_p_total_size, user_p_id_to_col, user_p_id_to_size,
                                          user_p_value)
    user_p_flat_usp = user_p_flat[user_p_sparsity_mask]
    parameter_info = ParameterInfo(user_p_col_to_name_usp, user_p_flat_usp, user_p_id_to_col,
                                   user_p_ids, user_p_name_to_shape, user_p_name_to_size_usp,
                                   user_p_name_to_sparsity, user_p_name_to_sparsity_type,
                                   user_p_names, user_p_num, user_p_sparsity_mask, user_p_writable, None, None)
    return parameter_info


def handle_sparsity(p_prob: cp.Problem) -> None:
    for param in p_prob.parameters:
        # Check and warn about inappropriate sparsity for scalar and vector
        if param.attributes['sparsity']:
            if param.size == 1 or max(param.shape) == param.size:
                param_type = 'scalar' if param.size == 1 else 'vector'
                warnings.warn(f'Ignoring sparsity pattern for {param_type} parameter {param.name()}!')
                param.attributes['sparsity'] = None
            else:
                invalid_sparsity = False
                for r, c in zip(*param.attributes['sparsity']):
                    if r < 0 or c < 0 or r >= param.shape[0] or c >= param.shape[1]:
                        warnings.warn(f'Invalid sparsity pattern for parameter {param.name()} - out of range! Ignoring sparsity pattern.')
                        param.attributes['sparsity'] = None
                        invalid_sparsity = True
                        break
                if not invalid_sparsity:
                    coo_unique = set(zip(*param.attributes['sparsity']))
                    coo_unique_sorted = sorted(coo_unique, key=lambda x: (x[1], x[0]))
                    param.attributes['sparsity'] = tuple(zip(*coo_unique_sorted))
        elif param.attributes['diag']:
            size = param.shape[0]
            param.attributes['sparsity'] = (np.arange(size), np.arange(size))

        # Zero out non-sparse values
        if param.attributes['sparsity'] and param.value is not None:
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    if (i, j) not in zip(*param.attributes['sparsity']) and param.value[i, j] != 0:
                        warnings.warn(f'Ignoring nonzero value outside of sparsity pattern for parameter {param.name()}!')
                        param.value[i, j] = 0



def compile_python_module(code_dir: str):
    sys.stdout.write('Compiling python wrapper with CVXPYgen ... \n')
    p_dir = os.getcwd()
    os.chdir(code_dir)
    call([sys.executable, 'setup.py', '--quiet', 'build_ext', '--inplace'])
    os.chdir(p_dir)
    sys.stdout.write("CVXPYgen finished compiling python wrapper.\n")


def create_folder_structure(code_dir: str):
    cvxpygen_directory = os.path.dirname(os.path.realpath(__file__))

    # Re-create code directory
    shutil.rmtree(code_dir, ignore_errors=True)
    os.mkdir(code_dir)
    
    # Create directory structures
    os.makedirs(os.path.join(code_dir, 'c', 'src'))
    os.makedirs(os.path.join(code_dir, 'c', 'include'))
    os.makedirs(os.path.join(code_dir, 'c', 'build'))
    os.makedirs(os.path.join(code_dir, 'cpp', 'src'))
    os.makedirs(os.path.join(code_dir, 'cpp', 'include'))

    # Copy template files
    shutil.copy(os.path.join(cvxpygen_directory, 'template', 'CMakeLists.txt'),
                os.path.join(code_dir, 'c'))
    for file in ['setup.py', 'README.html', '__init__.py']:
        shutil.copy(os.path.join(cvxpygen_directory, 'template', file), code_dir)

    return cvxpygen_directory
