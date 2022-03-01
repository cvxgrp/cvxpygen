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
import osqp
from cvxpygen import utils
from cvxpygen.utils import C
import cvxpy as cp
import numpy as np
from scipy import sparse
from subprocess import call
from platform import system
from cvxpy.problems.objective import Maximize
from cvxpy.cvxcore.python import canonInterface as cI
from cvxpy.expressions.variable import upper_tri_to_full
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS


def generate_code(problem, code_dir='CPG_code', solver=None, unroll=False, prefix='', wrapper=True):
    """
    Generate C code for CVXPY problem and (optionally) python wrapper
    """

    sys.stdout.write('Generating code with CVXPYgen ...\n')

    cvxpygen_directory = os.path.dirname(os.path.realpath(__file__))
    solver_code_dir = os.path.join(code_dir, 'c', 'solver_code')

    # adjust problem_name
    if prefix != '':
        if not prefix[0].isalpha():
            prefix = '_' + prefix
        prefix = prefix + '_'

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
    shutil.copy(os.path.join(cvxpygen_directory, 'template', 'CMakeLists.txt'), os.path.join(code_dir, 'c'))
    for file in ['setup.py', 'README.html']:
        shutil.copy(os.path.join(cvxpygen_directory, 'template', file), code_dir)

    # problem data
    data, solving_chain, inverse_data = problem.get_problem_data(solver=solver, gp=False, enforce_dpp=True,
                                                                 verbose=False)

    # catch non-supported cone types
    solver_name = solving_chain.solver.name()
    p_prob = data['param_prob']
    if solver_name == 'ECOS':
        if p_prob.cone_dims.exp > 0:
            raise ValueError('Code generation with ECOS and exponential cones is not supported yet.')
    elif solver_name == 'SCS':
        if p_prob.cone_dims.exp > 0 or len(p_prob.cone_dims.psd) > 0 or len(p_prob.cone_dims.p3d) > 0:
            raise ValueError('Code generation with SCS and exponential, positive semidefinite, or power cones '
                             'is not supported yet.')

    # checks in sparsity
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
                    if coord[0] < 0 or coord[1] < 0 or coord[0] >= p.shape[0] or coord[1] >= p.shape[1]:
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
                        warnings.warn('Ignoring nonzero value outside of sparsity pattern for parameter %s!' % p.name())
                        p.value[i, j] = 0

    # variable information
    variables = problem.variables()
    var_names = [var.name() for var in variables]
    var_ids = [var.id for var in variables]
    var_offsets = [inverse_data[-2].var_offsets[var_id] for var_id in var_ids]
    var_name_to_offset = {n: o for n, o in zip(var_names, var_offsets)}
    var_shapes = [var.shape for var in variables]
    var_sizes = [var.size for var in variables]
    var_sym = [var.attributes['symmetric'] or var.attributes['PSD'] or var.attributes['NSD'] for var in variables]
    var_name_to_sym = {n: s for n, s in zip(var_names, var_sym)}
    var_name_to_indices = {}
    for var_name, offset, shape, sym in zip(var_names, var_offsets, var_shapes, var_sym):
        if sym:
            fill_coefficient = upper_tri_to_full(shape[0])
            (_, col) = fill_coefficient.nonzero()
            var_name_to_indices[var_name] = offset + col
        else:
            var_name_to_indices[var_name] = np.arange(offset, offset+np.prod(shape))
    var_name_to_size = {var.name(): var.size for var in variables}
    var_name_to_shape = {var.name(): var.shape for var in variables}
    var_name_to_init = dict()
    for var in variables:
        if len(var.shape) == 0:
            var_name_to_init[var.name()] = 0
        else:
            var_name_to_init[var.name()] = np.zeros(shape=var.shape)

    # dual variable information
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
        d_vectors = ['y']*len(d_canon_offsets)
    else:
        n_split_yz = len(inverse_data[-1][ECOS.EQ_CONSTR])
        d_vectors = ['y']*n_split_yz + ['z']*(len(d_canon_offsets)-n_split_yz)
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

    # user parameters
    user_p_num = len(p_prob.parameters)
    user_p_names = [par.name() for par in p_prob.parameters]
    user_p_ids = list(p_prob.param_id_to_col.keys())
    user_p_id_to_col = p_prob.param_id_to_col
    user_p_id_to_size = p_prob.param_id_to_size
    user_p_id_to_param = p_prob.id_to_param
    user_p_total_size = p_prob.total_param_size
    user_p_name_to_shape = {user_p_id_to_param[p_id].name(): user_p_id_to_param[p_id].shape
                            for p_id in user_p_id_to_size.keys()}
    user_p_name_to_size_usp = {user_p_id_to_param[p_id].name(): size for p_id, size in user_p_id_to_size.items()}
    user_p_name_to_sparsity = {}
    user_p_name_to_sparsity_type = {}
    user_p_sparsity_mask = np.ones(user_p_total_size + 1, dtype=bool)
    for p in p_prob.parameters:
        if p.attributes['sparsity'] is not None:
            user_p_name_to_size_usp[p.name()] = len(p.attributes['sparsity'])
            user_p_name_to_sparsity[p.name()] = np.sort([coord[0]+p.shape[0]*coord[1]
                                                         for coord in p.attributes['sparsity']])
            if p.attributes['diag']:
                user_p_name_to_sparsity_type[p.name()] = 'diag'
            else:
                user_p_name_to_sparsity_type[p.name()] = 'general'
            user_p_sparsity_mask[user_p_id_to_col[p.id]:user_p_id_to_col[p.id]+user_p_id_to_size[p.id]] = False
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
        return np.array(user_p_id_to_param[user_p_id].value)
    user_p_flat = cI.get_parameter_vector(user_p_total_size, user_p_id_to_col, user_p_id_to_size, user_p_value)
    user_p_flat_usp = user_p_flat[user_p_sparsity_mask]

    canon_p = {}
    canon_p_csc = {}
    canon_p_id_to_mapping = {}
    canon_p_id_to_changes = {}
    canon_p_id_to_size = {}
    nonzero_d = True

    # dimensions and information specific to solver

    if solver_name == 'OSQP':

        canon_p_ids = ['P', 'q', 'd', 'A', 'l', 'u']
        canon_p_ids_constr_vec = ['l', 'u']
        sign_constr_vec = -1
        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        indices_obj, indptr_obj, shape_obj = p_prob.problem_data_index_P
        indices_constr, indptr_constr, shape_constr = p_prob.problem_data_index_A

        canon_constants = {}

    elif solver_name == 'SCS':

        canon_p_ids = ['c', 'd', 'A', 'b']
        canon_p_ids_constr_vec = ['b']
        sign_constr_vec = 1
        n_var = p_prob.x.size
        n_eq = data['A'].shape[0]
        n_ineq = 0

        indices_obj, indptr_obj, shape_obj = None, None, None
        indices_constr, indptr_constr, shape_constr = p_prob.problem_data_index

        canon_constants = {'n': n_var, 'm': n_eq, 'z': p_prob.cone_dims.zero, 'l': p_prob.cone_dims.nonneg,
                           'q': np.array(p_prob.cone_dims.soc), 'qsize': len(p_prob.cone_dims.soc)}

    elif solver_name == 'ECOS':

        canon_p_ids = ['c', 'd', 'A', 'b', 'G', 'h']
        canon_p_ids_constr_vec = ['b', 'h']
        sign_constr_vec = 1
        n_var = p_prob.x.size
        n_eq = p_prob.cone_dims.zero
        n_ineq = data['G'].shape[0]

        indices_obj, indptr_obj, shape_obj = None, None, None
        indices_constr, indptr_constr, shape_constr = p_prob.problem_data_index

        canon_constants = {'n': n_var, 'm': n_ineq, 'p': n_eq, 'l': p_prob.cone_dims.nonneg,
                           'n_cones': len(p_prob.cone_dims.soc), 'q': np.array(p_prob.cone_dims.soc),
                           'e': p_prob.cone_dims.exp}

    else:
        raise ValueError("Problem class cannot be addressed by the OSQP, SCS, or ECOS solver!")

    n_data_constr = len(indices_constr)
    n_data_constr_vec = indptr_constr[-1] - indptr_constr[-2]
    n_data_constr_mat = n_data_constr - n_data_constr_vec

    mapping_rows_eq = np.nonzero(indices_constr < n_eq)[0]
    mapping_rows_ineq = np.nonzero(indices_constr >= n_eq)[0]

    adjacency = np.zeros(shape=(len(canon_p_ids), user_p_num), dtype=bool)

    for i, p_id in enumerate(canon_p_ids):

        # compute affine mapping for each canonical parameter
        mapping_rows = []
        mapping = []
        indices = []
        indptr = []
        shape = ()

        if solver_name == 'OSQP':

            if p_id == 'P':
                mapping = p_prob.reduced_P
                indices = indices_obj
                shape = (n_var, n_var)
            elif p_id == 'q':
                mapping = p_prob.q[:-1]
            elif p_id == 'd':
                mapping = p_prob.q[-1]
            elif p_id == 'A':
                mapping = p_prob.reduced_A[:n_data_constr_mat]
                indices = indices_constr[:n_data_constr_mat]
                shape = (n_eq+n_ineq, n_var)
            elif p_id == 'l':
                mapping_rows_eq = np.nonzero(indices_constr < n_eq)[0]
                mapping_rows = mapping_rows_eq[mapping_rows_eq >= n_data_constr_mat]  # mapping to the finite part of l
                indices = indices_constr[mapping_rows]
                shape = (n_eq, 1)
            elif p_id == 'u':
                mapping_rows = np.arange(n_data_constr_mat, n_data_constr)
                indices = indices_constr[mapping_rows]
                shape = (n_eq+n_ineq, 1)
            else:
                raise ValueError('Unknown OSQP parameter name: "%s"' % p_id)

        elif solver_name in ['SCS', 'ECOS']:

            if p_id == 'c':
                mapping = p_prob.c[:-1]
            elif p_id == 'd':
                mapping = p_prob.c[-1]
            elif p_id == 'A':
                mapping_rows = mapping_rows_eq[mapping_rows_eq < n_data_constr_mat]
                shape = (n_eq, n_var)
            elif p_id == 'G':
                mapping_rows = mapping_rows_ineq[mapping_rows_ineq < n_data_constr_mat]
                shape = (n_ineq, n_var)
            elif p_id == 'b':
                mapping_rows = mapping_rows_eq[mapping_rows_eq >= n_data_constr_mat]
                shape = (n_eq, 1)
            elif p_id == 'h':
                mapping_rows = mapping_rows_ineq[mapping_rows_ineq >= n_data_constr_mat]
                shape = (n_ineq, 1)
            else:
                raise ValueError('Unknown %s parameter name: "%s"' % (solver_name, p_id))

            if p_id in ['A', 'b']:
                indices = indices_constr[mapping_rows]
            elif p_id in ['G', 'h']:
                indices = indices_constr[mapping_rows] - n_eq

            if p_id.isupper():
                mapping = -p_prob.reduced_A[mapping_rows]

        if p_id in canon_p_ids_constr_vec:
            mapping_to_sparse = sign_constr_vec*p_prob.reduced_A[mapping_rows]
            mapping_to_dense = sparse.lil_matrix(np.zeros((shape[0], mapping_to_sparse.shape[1])))
            for i_data in range(mapping_to_sparse.shape[0]):
                mapping_to_dense[indices[i_data], :] = mapping_to_sparse[i_data, :]
            mapping = sparse.csc_matrix(mapping_to_dense)

        if p_id == 'd':
            nonzero_d = mapping.nnz > 0

        # compute adjacency matrix
        for j in range(user_p_num):
            column_slice = slice(user_p_id_to_col[user_p_ids[j]], user_p_id_to_col[user_p_ids[j + 1]])
            if mapping[:, column_slice].nnz > 0:
                adjacency[i, j] = True

        # take sparsity into account
        mapping = mapping[:, user_p_sparsity_mask]

        # compute default values of canonical parameters
        if p_id.isupper():
            rows_nonzero, _ = mapping.nonzero()
            canon_p_data_nonzero = np.sort(np.unique(rows_nonzero))
            mapping = mapping[canon_p_data_nonzero, :]
            canon_p_data = mapping @ user_p_flat_usp
            # compute 'indptr' to construct sparse matrix from 'canon_p_data' and 'indices'
            if solver_name in ['OSQP', 'SCS']:
                if p_id == 'P':
                    indptr = indptr_obj
                elif p_id == 'A':
                    indptr = indptr_constr[:-1]
            elif solver_name == 'ECOS':
                indptr_original = indptr_constr[:-1]
                indptr = 0 * indptr_original
                for r in mapping_rows:
                    for c in range(shape[1]):
                        if indptr_original[c] <= r < indptr_original[c + 1]:
                            indptr[c + 1:] += 1
                            break
            # compute 'indices_usp' and 'indptr_usp'
            indices_usp = indices[canon_p_data_nonzero]
            indptr_usp = 0 * indptr
            for r in canon_p_data_nonzero:
                for c in range(shape[1]):
                    if indptr[c] <= r < indptr[c + 1]:
                        indptr_usp[c + 1:] += 1
                        break
            csc_mat = sparse.csc_matrix((canon_p_data, indices_usp, indptr_usp), shape=shape)
            if solver_name == 'OSQP':
                canon_p_csc[p_id] = csc_mat
            canon_p[p_id] = utils.csc_to_dict(csc_mat)
        else:
            canon_p_data = mapping @ user_p_flat_usp
            if solver_name == 'OSQP' and p_id == 'l':
                canon_p[p_id] = np.concatenate((canon_p_data, -np.inf * np.ones(n_ineq)), axis=0)
            else:
                canon_p[p_id] = canon_p_data

        canon_p_id_to_mapping[p_id] = mapping.tocsr()
        canon_p_id_to_changes[p_id] = mapping[:, :-1].nnz > 0
        canon_p_id_to_size[p_id] = mapping.shape[0]

    if solver_name == 'OSQP':

        # solver settings
        settings_names = ['rho', 'max_iter', 'eps_abs', 'eps_rel', 'eps_prim_inf', 'eps_dual_inf', 'alpha',
                          'scaled_termination', 'check_termination', 'warm_start']
        settings_types = ['c_float', 'c_int', 'c_float', 'c_float', 'c_float', 'c_float', 'c_float', 'c_int', 'c_int',
                          'c_int']
        settings_defaults = []

        # OSQP codegen
        osqp_obj = osqp.OSQP()
        osqp_obj.setup(P=canon_p_csc['P'], q=canon_p['q'], A=canon_p_csc['A'], l=canon_p['l'], u=canon_p['u'])
        if system() == 'Windows':
            cmake_generator = 'MinGW Makefiles'
        elif system() == 'Linux' or system() == 'Darwin':
            cmake_generator = 'Unix Makefiles'
        else:
            raise OSError('Unknown operating system!')
        osqp_obj.codegen(os.path.join(code_dir, 'c', 'solver_code'), project_type=cmake_generator,
                         parameters='matrices', force_rewrite=True)

        # copy license files
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'osqp-python', 'LICENSE'),
                        os.path.join(solver_code_dir, 'LICENSE'))
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'LICENSE'), code_dir)

    elif solver_name == 'SCS':

        # solver settings
        settings_names = ['normalize', 'scale', 'adaptive_scale', 'rho_x', 'max_iters', 'eps_abs',  'eps_rel',
                          'eps_infeas', 'alpha', 'time_limit_secs', 'verbose', 'warm_start', 'acceleration_lookback',
                          'acceleration_interval', 'write_data_filename', 'log_csv_filename']
        settings_types = ['c_int', 'c_float', 'c_int', 'c_float', 'c_int', 'c_float', 'c_float', 'c_float', 'c_float',
                          'c_float', 'c_int', 'c_int', 'c_int', 'c_int', 'const char*', 'const char*']
        settings_defaults = ['1', '0.1', '1', '1e-6', '1e5', '1e-4', '1e-4', '1e-7', '1.5', '0', '0', '0', '0', '1',
                             'SCS_NULL', 'SCS_NULL']

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'linsys', 'cmake']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'scs', dtc), os.path.join(solver_code_dir, dtc))
        files_to_copy = ['scs.mk', 'CMakeLists.txt', 'LICENSE.txt']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'scs', fl),
                            os.path.join(solver_code_dir, fl))
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'LICENSE'), code_dir)

        # disable BLAS and LAPACK
        with open(os.path.join(code_dir, 'c', 'solver_code', 'scs.mk'), 'r') as f:
            scs_mk_data = f.read()
        scs_mk_data = scs_mk_data.replace('USE_LAPACK = 1', 'USE_LAPACK = 0')
        with open(os.path.join(code_dir, 'c', 'solver_code', 'scs.mk'), 'w') as f:
            f.write(scs_mk_data)

        # modify CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'r') as f:
            cmake_data = f.read()
        cmake_data = cmake_data.replace(' include/', ' ${CMAKE_CURRENT_SOURCE_DIR}/include/')
        cmake_data = cmake_data.replace(' src/', ' ${CMAKE_CURRENT_SOURCE_DIR}/src/')
        cmake_data = cmake_data.replace(' ${LINSYS}/', ' ${CMAKE_CURRENT_SOURCE_DIR}/${LINSYS}/')
        with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'w') as f:
            f.write(cmake_data)

        # adjust top-level CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'r') as f:
            cmake_data = f.read()
        indent = ' ' * 6
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/'
        cmake_data = cmake_data.replace(sdir + 'include',
                                        sdir + 'include\n' +
                                        indent + sdir + 'linsys')
        with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'w') as f:
            f.write(cmake_data)

        # adjust setup.py
        with open(os.path.join(code_dir, 'setup.py'), 'r') as f:
            setup_text = f.read()
        indent = ' ' * 30
        setup_text = setup_text.replace("os.path.join('c', 'solver_code', 'include'),",
                                        "os.path.join('c', 'solver_code', 'include'),\n" +
                                        indent + "os.path.join('c', 'solver_code', 'linsys'),")
        with open(os.path.join(code_dir, 'setup.py'), 'w') as f:
            f.write(setup_text)

    elif solver_name == 'ECOS':

        # solver settings
        settings_names = ['feastol', 'abstol', 'reltol', 'feastol_inacc', 'abstol_inacc', 'reltol_inacc', 'maxit']
        settings_types = ['c_float', 'c_float', 'c_float', 'c_float', 'c_float', 'c_float', 'c_int']
        settings_defaults = ['1e-8', '1e-8', '1e-8', '1e-4', '5e-5', '5e-5', '100']

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'external', 'ecos_bb']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'ecos', dtc), os.path.join(solver_code_dir, dtc))
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', 'CMakeLists.txt'),
                        os.path.join(solver_code_dir, 'CMakeLists.txt'))
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', 'COPYING'),
                        os.path.join(solver_code_dir, 'COPYING'))
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', 'COPYING'),
                        os.path.join(code_dir, 'COPYING'))

        # adjust print level
        with open(os.path.join(code_dir, 'c', 'solver_code', 'include', 'glblopts.h'), 'r') as f:
            glbl_opts_data = f.read()
        glbl_opts_data = glbl_opts_data.replace('#define PRINTLEVEL (2)', '#define PRINTLEVEL (0)')
        with open(os.path.join(code_dir, 'c', 'solver_code', 'include', 'glblopts.h'), 'w') as f:
            f.write(glbl_opts_data)

        # adjust top-level CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'r') as f:
            cmake_data = f.read()
        indent = ' ' * 6
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/'
        cmake_data = cmake_data.replace(sdir + 'include',
                                        sdir + 'include\n' +
                                        indent + sdir + 'external/SuiteSparse_config\n' +
                                        indent + sdir + 'external/amd/include\n' +
                                        indent + sdir + 'external/ldl/include')
        with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'w') as f:
            f.write(cmake_data)

        # remove library target from ECOS CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'w') as f:
            for line in lines:
                if '# ECOS library' in line:
                    break
                f.write(line)

        # adjust setup.py
        with open(os.path.join(code_dir, 'setup.py'), 'r') as f:
            setup_text = f.read()
        indent = ' ' * 30
        setup_text = setup_text.replace("os.path.join('c', 'solver_code', 'include'),",
                                        "os.path.join('c', 'solver_code', 'include'),\n" +
                                        indent+"os.path.join('c', 'solver_code', 'external', 'SuiteSparse_config'),\n" +
                                        indent+"os.path.join('c', 'solver_code', 'external', 'amd', 'include'),\n" +
                                        indent+"os.path.join('c', 'solver_code', 'external', 'ldl', 'include'),")
        setup_text = setup_text.replace("license='Apache 2.0'", "license='GPL 3.0'")
        with open(os.path.join(code_dir, 'setup.py'), 'w') as f:
            f.write(setup_text)

    else:
        raise ValueError("Problem class cannot be addressed by the OSQP, SCS, or ECOS solver!")

    user_p_name_to_canon_outdated = {user_p_name: [canon_p_ids[j] for j in np.nonzero(adjacency[:, i])[0]]
                                     for i, user_p_name in enumerate(user_p_names)}
    settings_names_to_type = {name: typ for name, typ in zip(settings_names, settings_types)}
    settings_names_to_default = {name: typ for name, typ in zip(settings_names, settings_defaults)}

    ret_prim_func_exists = any(var_sym) or any([s == 1 for s in var_sizes]) or solver_name == 'ECOS'
    ret_dual_func_exists = any([s == 1 for s in d_sizes]) or solver_name == 'ECOS'

    # summarize information on options, codegen, user parameters / variables, canonicalization in dictionaries

    info_opt = {C.CODE_DIR: code_dir,
                C.SOLVER_NAME: solver_name,
                C.UNROLL: unroll,
                C.PREFIX: prefix}

    info_cg = {C.RET_PRIM_FUNC_EXISTS: ret_prim_func_exists,
               C.RET_DUAL_FUNC_EXISTS: ret_dual_func_exists,
               C.NONZERO_D: nonzero_d,
               C.IS_MAXIMIZATION: type(problem.objective) == Maximize}

    info_usr = {C.P_WRITABLE: user_p_writable,
                C.P_FLAT_USP: user_p_flat_usp,
                C.P_COL_TO_NAME_USP: user_p_col_to_name_usp,
                C.P_NAME_TO_SHAPE: user_p_name_to_shape,
                C.P_NAME_TO_SIZE: user_p_name_to_size_usp,
                C.P_NAME_TO_CANON_OUTDATED: user_p_name_to_canon_outdated,
                C.P_NAME_TO_SPARSITY: user_p_name_to_sparsity,
                C.P_NAME_TO_SPARSITY_TYPE: user_p_name_to_sparsity_type,
                C.V_NAME_TO_INDICES: var_name_to_indices,
                C.V_NAME_TO_SIZE: var_name_to_size,
                C.V_NAME_TO_SHAPE: var_name_to_shape,
                C.V_NAME_TO_INIT: var_name_to_init,
                C.V_NAME_TO_SYM: var_name_to_sym,
                C.V_NAME_TO_OFFSET: var_name_to_offset,
                C.D_NAME_TO_INIT: d_name_to_init,
                C.D_NAME_TO_VEC: d_name_to_vec,
                C.D_NAME_TO_OFFSET: d_name_to_offset,
                C.D_NAME_TO_SIZE: d_name_to_size,
                C.D_NAME_TO_SHAPE: d_name_to_shape,
                C.D_NAME_TO_INDICES: d_name_to_indices}

    info_can = {C.P: canon_p,
                C.P_ID_TO_SIZE: canon_p_id_to_size,
                C.P_ID_TO_CHANGES: canon_p_id_to_changes,
                C.P_ID_TO_MAPPING: canon_p_id_to_mapping,
                C.CONSTANTS: canon_constants,
                C.SETTINGS_NAMES_TO_TYPE: settings_names_to_type,
                C.SETTINGS_NAMES_TO_DEFAULT: settings_names_to_default}

    # 'workspace' prototypes
    with open(os.path.join(code_dir, 'c', 'include', 'cpg_workspace.h'), 'w') as f:
        utils.write_workspace_prot(f, info_opt, info_usr, info_can)

    # 'workspace' definitions
    with open(os.path.join(code_dir, 'c', 'src', 'cpg_workspace.c'), 'w') as f:
        utils.write_workspace_def(f, info_opt, info_usr, info_can)

    # 'solve' prototypes
    with open(os.path.join(code_dir, 'c', 'include', 'cpg_solve.h'), 'w') as f:
        utils.write_solve_prot(f, info_opt, info_cg, info_usr, info_can)

    # 'solve' definitions
    with open(os.path.join(code_dir, 'c', 'src', 'cpg_solve.c'), 'w') as f:
        utils.write_solve_def(f, info_opt, info_cg, info_usr, info_can)

    # 'example' definitions
    with open(os.path.join(code_dir, 'c', 'src', 'cpg_example.c'), 'w') as f:
        utils.write_example_def(f, info_opt, info_usr)

    # adapt top-level CMakeLists.txt
    with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'r') as f:
        cmake_data = f.read()
    cmake_data = utils.replace_cmake_data(cmake_data, info_opt)
    with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'w') as f:
        f.write(cmake_data)

    # adapt solver CMakeLists.txt
    with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'a') as f:
        utils.write_canon_cmake(f, info_opt)

    # binding module prototypes
    with open(os.path.join(code_dir, 'cpp', 'include', 'cpg_module.hpp'), 'w') as f:
        utils.write_module_prot(f, info_opt, info_usr)

    # binding module definition
    with open(os.path.join(code_dir, 'cpp', 'src', 'cpg_module.cpp'), 'w') as f:
        utils.write_module_def(f, info_opt, info_usr, info_can)

    # adapt setup.py
    with open(os.path.join(code_dir, 'setup.py'), 'r') as f:
        setup_data = f.read()
    setup_data = utils.replace_setup_data(setup_data)
    with open(os.path.join(code_dir, 'setup.py'), 'w') as f:
        f.write(setup_data)

    # custom CVXPY solve method
    with open(os.path.join(code_dir, 'cpg_solver.py'), 'w') as f:
        utils.write_method(f, info_opt, info_usr)

    # serialize problem formulation
    with open(os.path.join(code_dir, 'problem.pickle'), 'wb') as f:
        pickle.dump(cp.Problem(problem.objective, problem.constraints), f)

    # html documentation file
    with open(os.path.join(code_dir, 'README.html'), 'r') as f:
        html_data = f.read()
    html_data = utils.replace_html_data(html_data, info_opt, info_usr)
    with open(os.path.join(code_dir, 'README.html'), 'w') as f:
        f.write(html_data)

    sys.stdout.write('CVXPYgen finished generating code.\n')

    # compile python module
    if wrapper:
        sys.stdout.write('Compiling python wrapper with CVXPYgen ... \n')
        p_dir = os.getcwd()
        os.chdir(code_dir)
        call([sys.executable, 'setup.py', '--quiet', 'build_ext', '--inplace'])
        os.chdir(p_dir)
        sys.stdout.write("CVXPYgen finished compiling python wrapper.\n")
