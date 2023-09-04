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

from cvxpygen import utils
from cvxpygen.mappings import Configuration, PrimalVariableInfo, DualVariableInfo, ConstraintInfo, \
    ParameterCanon, ParameterInfo
from cvxpygen.solvers import get_interface_class
import cvxpy as cp
import numpy as np
from scipy import sparse
from subprocess import call
from cvxpy.problems.objective import Maximize
from cvxpy.cvxcore.python import canonInterface as cI
from cvxpy.expressions.variable import upper_tri_to_full
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS


def generate_code(problem, code_dir='CPG_code', solver=None, enable_settings=[], unroll=False, prefix='', wrapper=True):
    """
    Generate C code for CVXPY problem and (optionally) python wrapper
    """

    sys.stdout.write('Generating code with CVXPYgen ...\n')

    create_folder_structure(code_dir)

    # problem data
    solver_opts = {}
    # TODO support quadratic objective for SCS.
    if solver == 'SCS':
        solver_opts['use_quad_obj'] = False
    data, solving_chain, inverse_data = problem.get_problem_data(
        solver=solver,
        gp=False,
        enforce_dpp=True,
        verbose=False,
        solver_opts=solver_opts,
    )
    param_prob = data['param_prob']

    solver_name = solving_chain.solver.name()
    interface_class = get_interface_class(solver_name)

    # for cone problems, check if all cones are supported
    if hasattr(param_prob, 'cone_dims'):
        cone_dims = param_prob.cone_dims
        interface_class.check_unsupported_cones(cone_dims)

    # checks in sparsity
    handle_sparsity(param_prob)

    # variable information
    variable_info = get_variable_info(problem, inverse_data)

    # dual variable information
    dual_variable_info = get_dual_variable_info(inverse_data, solver_name)

    # user parameters
    parameter_info = get_parameter_info(param_prob)

    # dimensions and information specific to solver
    solver_interface = interface_class(data, param_prob, enable_settings)  # noqa

    constraint_info = get_constraint_info(solver_interface)

    adjacency, parameter_canon = process_canonical_parameters(constraint_info, param_prob,
                                                              parameter_info, solver_interface,
                                                              solver_name, problem)

    cvxpygen_directory = os.path.dirname(os.path.realpath(__file__))
    solver_code_dir = os.path.join(code_dir, 'c', 'solver_code')
    solver_interface.generate_code(code_dir, solver_code_dir, cvxpygen_directory, parameter_canon)

    parameter_canon.user_p_name_to_canon_outdated = {
        user_p_name: [solver_interface.canon_p_ids[j] for j in np.nonzero(adjacency[:, i])[0]]
        for i, user_p_name in enumerate(parameter_info.names)}

    # configuration
    configuration = get_configuration(code_dir, solver, unroll, prefix)

    write_c_code(problem, configuration, variable_info, dual_variable_info, parameter_info,
                 parameter_canon, solver_interface)

    sys.stdout.write('CVXPYgen finished generating code.\n')

    if wrapper:
        compile_python_module(code_dir)

def process_canonical_parameters(constraint_info, param_prob, parameter_info, solver_interface, solver_name, problem):
    adjacency = np.zeros(shape=(len(solver_interface.canon_p_ids), parameter_info.num), dtype=bool)
    parameter_canon = ParameterCanon()
    # compute affine mapping for each canonical parameter
    for i, p_id in enumerate(solver_interface.canon_p_ids):

        affine_map = solver_interface.get_affine_map(p_id, param_prob, constraint_info)

        if p_id in solver_interface.canon_p_ids_constr_vec:
            affine_map = update_to_dense_mapping(affine_map, param_prob, solver_interface)

        if p_id == 'd':
            parameter_canon.nonzero_d = affine_map.mapping.nnz > 0

        adjacency = update_adjacency_matrix(adjacency, i, parameter_info, affine_map.mapping)

        # take sparsity into account
        affine_map.mapping = affine_map.mapping[:, parameter_info.sparsity_mask]

        # compute default values of canonical parameters
        affine_map, parameter_canon = set_default_values(affine_map, p_id, parameter_canon, parameter_info, solver_interface, solver_name)

        parameter_canon.p_id_to_mapping[p_id] = affine_map.mapping.tocsr()
        parameter_canon.p_id_to_changes[p_id] = affine_map.mapping[:, :-1].nnz > 0
        parameter_canon.p_id_to_size[p_id] = affine_map.mapping.shape[0]
    parameter_canon.is_maximization = type(problem.objective) == Maximize
    return adjacency, parameter_canon


def update_to_dense_mapping(affine_map, param_prob, solver_interface):
    mapping_to_sparse = solver_interface.sign_constr_vec * param_prob.reduced_A.reduced_mat[
        affine_map.mapping_rows]
    mapping_to_dense = sparse.lil_matrix(
        np.zeros((affine_map.shape[0], mapping_to_sparse.shape[1])))
    for i_data in range(mapping_to_sparse.shape[0]):
        mapping_to_dense[affine_map.indices[i_data], :] = mapping_to_sparse[i_data, :]
    affine_map.mapping = sparse.csc_matrix(mapping_to_dense)
    return affine_map


def set_default_values(affine_map, p_id, parameter_canon, parameter_info, solver_interface,
                       solver_name):
    if p_id.isupper():
        rows_nonzero, _ = affine_map.mapping.nonzero()
        canon_p_data_nonzero = np.sort(np.unique(rows_nonzero))
        affine_map.mapping = affine_map.mapping[canon_p_data_nonzero, :]
        canon_p_data = affine_map.mapping @ parameter_info.flat_usp
        # compute 'indptr' to construct sparse matrix from 'canon_p_data' and 'indices'
        if solver_name in ['OSQP', 'SCS']:
            if p_id == 'P':
                affine_map.indptr = solver_interface.indptr_obj
            elif p_id == 'A':
                affine_map.indptr = solver_interface.indptr_constr[:-1]
        elif solver_name == 'ECOS':
            indptr_original = solver_interface.indptr_constr[:-1]
            affine_map.indptr = 0 * indptr_original
            for r in affine_map.mapping_rows:
                for c in range(affine_map.shape[1]):
                    if indptr_original[c] <= r < indptr_original[c + 1]:
                        affine_map.indptr[c + 1:] += 1
                        break
        # compute 'indices_usp' and 'indptr_usp'
        indices_usp = affine_map.indices[canon_p_data_nonzero]
        indptr_usp = 0 * affine_map.indptr
        for r in canon_p_data_nonzero:
            for c in range(affine_map.shape[1]):
                if affine_map.indptr[c] <= r < affine_map.indptr[c + 1]:
                    indptr_usp[c + 1:] += 1
                    break
        csc_mat = sparse.csc_matrix((canon_p_data, indices_usp, indptr_usp),
                                    shape=affine_map.shape)
        if solver_name == 'OSQP':
            parameter_canon.p_csc[p_id] = csc_mat
        parameter_canon.p[p_id] = utils.csc_to_dict(csc_mat)
    else:
        canon_p_data = affine_map.mapping @ parameter_info.flat_usp
        if solver_name == 'OSQP' and p_id == 'l':
            parameter_canon.p[p_id] = np.concatenate(
                (canon_p_data, -np.inf * np.ones(solver_interface.n_ineq)), axis=0)
        else:
            parameter_canon.p[p_id] = canon_p_data

    return affine_map, parameter_canon


def get_variable_info(problem, inverse_data) -> PrimalVariableInfo:
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

    variable_info = PrimalVariableInfo(var_name_to_offset, var_name_to_indices, var_name_to_size,
                                       var_sizes, var_name_to_shape, var_name_to_init,
                                       var_name_to_sym, var_sym)
    return variable_info


def get_dual_variable_info(inverse_data, solver_name) -> DualVariableInfo:
    # get chain of constraint id maps for 'CvxAttr2Constr' and 'Canonicalization' objects
    dual_id_maps = []
    if solver_name == 'OSQP':
        if inverse_data[-4]:
            dual_id_maps.append(inverse_data[-4][2])
        dual_id_maps.append(inverse_data[-3].cons_id_map)
    elif solver_name in ['SCS', 'ECOS']:
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
    if solver_name == 'OSQP':
        con_canon = inverse_data[-2].constraints  # same order as in canonical dual vector
    elif solver_name == 'SCS':
        con_canon = inverse_data[-1][SCS.EQ_CONSTR] + inverse_data[-1][SCS.NEQ_CONSTR]
    else:
        con_canon = inverse_data[-1][ECOS.EQ_CONSTR] + inverse_data[-1][ECOS.NEQ_CONSTR]
    con_canon_dict = {c.id: c for c in con_canon}
    d_canon_offsets = np.cumsum([0] + [c.args[0].size for c in con_canon[:-1]])
    if solver_name in ['OSQP', 'SCS']:
        d_vectors = ['y'] * len(d_canon_offsets)
    else:
        n_split_yz = len(inverse_data[-1][ECOS.EQ_CONSTR])
        d_vectors = ['y'] * n_split_yz + ['z'] * (len(d_canon_offsets) - n_split_yz)
        d_canon_offsets[n_split_yz:] -= d_canon_offsets[n_split_yz]
    d_canon_offsets_dict = {c.id: off for c, off in zip(con_canon, d_canon_offsets)}
    # select for user-defined constraints
    d_offsets = [d_canon_offsets_dict[i] for i in dual_ids]
    d_sizes = [con_canon_dict[i].size for i in dual_ids]
    d_shapes = [con_canon_dict[i].shape for i in dual_ids]
    d_names = ['d%d' % i for i in range(len(dual_ids))]
    d_i_to_name = {i: 'd%d' % i for i in range(len(dual_ids))}
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
    n_data_constr_vec = solver_interface.indptr_constr[-1] - solver_interface.indptr_constr[-2]
    n_data_constr_mat = n_data_constr - n_data_constr_vec

    mapping_rows_eq = np.nonzero(solver_interface.indices_constr < solver_interface.n_eq)[0]
    mapping_rows_ineq = np.nonzero(solver_interface.indices_constr >= solver_interface.n_eq)[0]

    return ConstraintInfo(n_data_constr, n_data_constr_mat, mapping_rows_eq, mapping_rows_ineq)


def update_adjacency_matrix(adjacency, i, parameter_info, mapping) -> np.ndarray:
    # compute adjacency matrix
    for j in range(parameter_info.num):
        column_slice = slice(parameter_info.id_to_col[parameter_info.ids[j]],
                             parameter_info.id_to_col[parameter_info.ids[j + 1]])
        if mapping[:, column_slice].nnz > 0:
            adjacency[i, j] = True
    return adjacency


def write_c_code(problem: cp.Problem, configuration: dict, variable_info: dict, dual_variable_info: dict,
                 parameter_info: dict, parameter_canon: dict, solver_interface: dict) -> None:
    # 'workspace' prototypes
    with open(os.path.join(configuration.code_dir, 'c', 'include', 'cpg_workspace.h'), 'w') as f:
        utils.write_workspace_prot(f, configuration, variable_info, dual_variable_info, parameter_info, parameter_canon, solver_interface)
    # 'workspace' definitions
    with open(os.path.join(configuration.code_dir, 'c', 'src', 'cpg_workspace.c'), 'w') as f:
        utils.write_workspace_def(f, configuration, variable_info, dual_variable_info, parameter_info, parameter_canon, solver_interface)
    # 'solve' prototypes
    with open(os.path.join(configuration.code_dir, 'c', 'include', 'cpg_solve.h'), 'w') as f:
        utils.write_solve_prot(f, configuration, variable_info, dual_variable_info, parameter_info, parameter_canon, solver_interface)
    # 'solve' definitions
    with open(os.path.join(configuration.code_dir, 'c', 'src', 'cpg_solve.c'), 'w') as f:
        utils.write_solve_def(f, configuration, variable_info, dual_variable_info, parameter_info, parameter_canon, solver_interface)
    # 'example' definitions
    with open(os.path.join(configuration.code_dir, 'c', 'src', 'cpg_example.c'), 'w') as f:
        utils.write_example_def(f, configuration, variable_info, dual_variable_info, parameter_info)
    # adapt top-level CMakeLists.txt
    with open(os.path.join(configuration.code_dir, 'c', 'CMakeLists.txt'), 'r') as f:
        cmake_data = f.read()
    cmake_data = utils.replace_cmake_data(cmake_data, configuration)
    with open(os.path.join(configuration.code_dir, 'c', 'CMakeLists.txt'), 'w') as f:
        f.write(cmake_data)
    # adapt solver CMakeLists.txt
    with open(os.path.join(configuration.code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'a') as f:
        utils.write_canon_cmake(f, configuration)
    # binding module prototypes
    with open(os.path.join(configuration.code_dir, 'cpp', 'include', 'cpg_module.hpp'), 'w') as f:
        utils.write_module_prot(f, configuration, parameter_info, variable_info, dual_variable_info)
    # binding module definition
    with open(os.path.join(configuration.code_dir, 'cpp', 'src', 'cpg_module.cpp'), 'w') as f:
        utils.write_module_def(f, configuration, variable_info, dual_variable_info, parameter_info, solver_interface)
    # adapt setup.py
    with open(os.path.join(configuration.code_dir, 'setup.py'), 'r') as f:
        setup_data = f.read()
    setup_data = utils.replace_setup_data(setup_data)
    with open(os.path.join(configuration.code_dir, 'setup.py'), 'w') as f:
        f.write(setup_data)
    # custom CVXPY solve method
    with open(os.path.join(configuration.code_dir, 'cpg_solver.py'), 'w') as f:
        utils.write_method(f, configuration, variable_info, dual_variable_info, parameter_info)
    # serialize problem formulation
    with open(os.path.join(configuration.code_dir, 'problem.pickle'), 'wb') as f:
        pickle.dump(cp.Problem(problem.objective, problem.constraints), f)
    # html documentation file
    with open(os.path.join(configuration.code_dir, 'README.html'), 'r') as f:
        html_data = f.read()
    html_data = utils.replace_html_data(html_data, configuration, variable_info, dual_variable_info, parameter_info, solver_interface)
    with open(os.path.join(configuration.code_dir, 'README.html'), 'w') as f:
        f.write(html_data)


def adjust_prefix(prefix):
    if prefix != '':
        if not prefix[0].isalpha():
            prefix = '_' + prefix
        prefix = prefix + '_'
    return prefix


def get_configuration(code_dir, solver_name, unroll, prefix) -> Configuration:
    return Configuration(code_dir, solver_name, unroll, adjust_prefix(prefix))


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
        if p.attributes['sparsity'] is not None:
            user_p_name_to_size_usp[p.name()] = len(p.attributes['sparsity'])
            user_p_name_to_sparsity[p.name()] = np.sort([coord[0] + p.shape[0] * coord[1]
                                                         for coord in p.attributes['sparsity']])
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
            if type(p.value) is sparse.dia_matrix:
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
                                   user_p_names, user_p_num, user_p_sparsity_mask, user_p_writable)
    return parameter_info


def handle_sparsity(p_prob: cp.Problem) -> None:
    for p in p_prob.parameters:
        if p.attributes['sparsity'] is not None:
            if p.size == 1:
                warnings.warn('Ignoring sparsity pattern for scalar parameter %s!' % p.name())
                p.attributes['sparsity'] = None
            elif max(p.shape) == p.size:
                warnings.warn('Ignoring sparsity pattern for vector parameter %s!' % p.name())
                p.attributes['sparsity'] = None
            else:
                for coord in p.attributes['sparsity']:
                    if coord[0] < 0 or coord[1] < 0 or coord[0] >= p.shape[0] or coord[1] >= \
                            p.shape[1]:
                        warnings.warn('Invalid sparsity pattern for parameter %s - out of range! '
                                      'Ignoring sparsity pattern.' % p.name())
                        p.attributes['sparsity'] = None
                        break
                p.attributes['sparsity'] = list(set(p.attributes['sparsity']))
        elif p.attributes['diag']:
            p.attributes['sparsity'] = [(i, i) for i in range(p.shape[0])]
        if p.attributes['sparsity'] is not None and p.value is not None:
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    if (i, j) not in p.attributes['sparsity'] and p.value[i, j] != 0:
                        warnings.warn(
                            'Ignoring nonzero value outside of sparsity pattern for parameter %s!' % p.name())
                        p.value[i, j] = 0


def compile_python_module(code_dir: str):
    sys.stdout.write('Compiling python wrapper with CVXPYgen ... \n')
    p_dir = os.getcwd()
    os.chdir(code_dir)
    call([sys.executable, 'setup.py', '--quiet', 'build_ext', '--inplace'])
    os.chdir(p_dir)
    sys.stdout.write("CVXPYgen finished compiling python wrapper.\n")


def create_folder_structure(code_dir: str):
    cvxpygen_directory = os.path.dirname(os.path.realpath(__file__))

    # create code directory and copy template files
    if os.path.isdir(code_dir):
        shutil.rmtree(code_dir)
    os.mkdir(code_dir)
    os.mkdir(os.path.join(code_dir, 'c'))
    for d in ['src', 'include', 'build']:
        os.mkdir(os.path.join(code_dir, 'c', d))
    os.mkdir(os.path.join(code_dir, 'cpp'))
    for d in ['src', 'include']:
        os.mkdir(os.path.join(code_dir, 'cpp', d))
    shutil.copy(os.path.join(cvxpygen_directory, 'template', 'CMakeLists.txt'),
                os.path.join(code_dir, 'c'))
    for file in ['setup.py', 'README.html', '__init__.py']:
        shutil.copy(os.path.join(cvxpygen_directory, 'template', file), code_dir)
    return cvxpygen_directory
