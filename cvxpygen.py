
import os
import sys
import shutil
import pickle
import warnings
import osqp
import utils
import cvxpy as cp
import numpy as np
from scipy import sparse
from subprocess import call
from platform import system
from cvxpy.problems.objective import Maximize
from cvxpy.cvxcore.python import canonInterface as cI
from cvxpy.expressions.variable import upper_tri_to_full


def generate_code(problem, code_dir='CPG_code', solver=None, compile_module=True, explicit=False, problem_name=''):
    """
    Generate C code for CVXPY problem and (optionally) python wrapper
    """

    sys.stdout.write('Generating code with CVXPYGEN ...\n')

    current_directory = os.path.dirname(os.path.realpath(__file__))
    solver_code_dir = os.path.join(code_dir, 'c', 'solver_code')

    # adjust problem_name
    if problem_name != '':
        if not problem_name[0].isalpha():
            problem_name = '_' + problem_name
        problem_name = problem_name + '_'

    # copy TEMPLATE
    if os.path.isdir(code_dir):
        shutil.rmtree(code_dir)
    shutil.copytree(os.path.join(current_directory, 'TEMPLATE'), code_dir)

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
        if p.attributes['diag']:
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
    inverse_data_idx = 0
    for inverse_data_idx in range(len(inverse_data)-1, -1, -1):
        if type(inverse_data[inverse_data_idx]) == cp.reductions.inverse_data.InverseData:
            break
    var_offsets = [inverse_data[inverse_data_idx].var_offsets[var_id] for var_id in var_ids]
    var_shapes = [var.shape for var in variables]
    var_sizes = [var.size for var in variables]
    var_symmetric = [var.attributes['symmetric'] or var.attributes['PSD'] or var.attributes['NSD'] for var in variables]
    var_name_to_indices = {}
    for var_name, offset, shape, symm in zip(var_names, var_offsets, var_shapes, var_symmetric):
        if symm:
            fill_coefficient = upper_tri_to_full(shape[0])
            (_, col) = fill_coefficient.nonzero()
            var_name_to_indices[var_name] = offset + col
        else:
            var_name_to_indices[var_name] = np.arange(offset, offset+np.prod(shape))

    var_name_to_size = {name: size for name, size in zip(var_names, var_sizes)}
    var_name_to_shape = {var.name(): var.shape for var in variables}
    var_init = dict()
    for var in variables:
        if len(var.shape) == 0:
            var_init[var.name()] = 0
        else:
            var_init[var.name()] = np.zeros(shape=var.shape)

    # user parameters
    user_p_num = len(p_prob.parameters)
    user_p_names = [par.name() for par in p_prob.parameters]
    user_p_ids = list(p_prob.param_id_to_col.keys())
    user_p_id_to_col = p_prob.param_id_to_col
    user_p_id_to_size = p_prob.param_id_to_size
    user_p_id_to_param = p_prob.id_to_param
    user_p_total_size = p_prob.total_param_size
    user_p_name_to_size_usp = {name: size for name, size in zip(user_p_names, user_p_id_to_size.values())}
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
    user_p_sizes_usp = list(user_p_name_to_size_usp.values())
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

    canon_mappings = []
    canon_p = {}
    canon_p_to_changes = {}
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
        raise ValueError("Problem class cannot be addressed by the OSQP or ECOS solver!")

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
                canon_p[p_id + '_osqp'] = csc_mat
            canon_p[p_id] = utils.csc_to_dict(csc_mat)
        else:
            canon_p_data = mapping @ user_p_flat_usp
            if solver_name == 'OSQP' and p_id == 'l':
                canon_p[p_id] = np.concatenate((canon_p_data, -np.inf * np.ones(n_ineq)), axis=0)
            else:
                canon_p[p_id] = canon_p_data

        canon_mappings.append(mapping.tocsr())
        canon_p_to_changes[p_id] = mapping[:, :-1].nnz > 0
        canon_p_id_to_size[p_id] = mapping.shape[0]

    if solver_name == 'OSQP':

        # solver settings
        canon_settings_names = ['rho', 'max_iter', 'eps_abs', 'eps_rel', 'eps_prim_inf', 'eps_dual_inf',
                                'alpha', 'scaled_termination', 'check_termination', 'warm_start']
        canon_settings_types = ['c_float', 'c_int', 'c_float', 'c_float', 'c_float', 'c_float', 'c_float',
                                'c_int', 'c_int', 'c_int']
        canon_settings_defaults = []

        # OSQP codegen
        osqp_obj = osqp.OSQP()
        osqp_obj.setup(P=canon_p['P_osqp'], q=canon_p['q'], A=canon_p['A_osqp'], l=canon_p['l'], u=canon_p['u'])
        if system() == 'Windows':
            cmake_generator = 'MinGW Makefiles'
        elif system() == 'Linux' or system() == 'Darwin':
            cmake_generator = 'Unix Makefiles'
        else:
            raise OSError('Unknown operating system!')
        osqp_obj.codegen(os.path.join(code_dir, 'c', 'solver_code'), project_type=cmake_generator,
                         parameters='matrices', force_rewrite=True)

        # copy license files
        shutil.copyfile(os.path.join(current_directory, 'solver', 'osqp-python', 'LICENSE'),
                        os.path.join(solver_code_dir, 'LICENSE'))
        shutil.copyfile(os.path.join(current_directory, 'LICENSE'),
                        os.path.join(code_dir, 'LICENSE'))

    elif solver_name == 'SCS':

        # solver settings
        canon_settings_names = ['normalize', 'scale', 'adaptive_scale', 'rho_x', 'max_iters', 'eps_abs',
                                'eps_rel', 'eps_infeas', 'alpha', 'time_limit_secs', 'verbose', 'warm_start',
                                'acceleration_lookback', 'acceleration_interval', 'write_data_filename',
                                'log_csv_filename']
        canon_settings_types = ['c_int', 'c_float', 'c_int', 'c_float', 'c_int', 'c_float', 'c_float',
                                'c_float', 'c_float', 'c_float', 'c_int', 'c_int', 'c_int', 'c_int', 'const char*',
                                'const char*']
        canon_settings_defaults = ['1', '0.1', '1', '1e-6', '1e5', '1e-4', '1e-4', '1e-7', '1.5', '0', '0', '0', '0',
                                   '1', 'SCS_NULL', 'SCS_NULL']

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'linsys', 'cmake']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(current_directory, 'solver', 'scs', dtc), os.path.join(solver_code_dir, dtc))
        files_to_copy = ['scs.mk', 'CMakeLists.txt', 'LICENSE.txt']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(current_directory, 'solver', 'scs', fl),
                            os.path.join(solver_code_dir, fl))

        # disable BLAS and LAPACK
        with open(os.path.join(code_dir, 'c', 'solver_code', 'scs.mk'), 'r') as f:
            scs_mk_data = f.read()
        scs_mk_data = scs_mk_data.replace('USE_LAPACK = 1', 'USE_LAPACK = 0')
        with open(os.path.join(code_dir, 'c', 'solver_code', 'scs.mk'), 'w') as f:
            f.write(scs_mk_data)

        # modify CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'r') as f:
            cmake_data = f.read()
        cmake_data = cmake_data.replace('include/', '${CMAKE_CURRENT_SOURCE_DIR}/include/')
        cmake_data = cmake_data.replace('src/', '${CMAKE_CURRENT_SOURCE_DIR}/src/')
        cmake_data = cmake_data.replace('${LINSYS}/', '${CMAKE_CURRENT_SOURCE_DIR}/${LINSYS}/')
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
        canon_settings_names = ['feastol', 'abstol', 'reltol', 'feastol_inacc', 'abstol_inacc', 'reltol_inacc', 'maxit']
        canon_settings_types = ['c_float', 'c_float', 'c_float', 'c_float', 'c_float', 'c_float', 'c_int']
        canon_settings_defaults = ['1e-8', '1e-8', '1e-8', '1e-4', '5e-5', '5e-5', '100']

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'external', 'ecos_bb']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(current_directory, 'solver', 'ecos', dtc), os.path.join(solver_code_dir, dtc))
        shutil.copyfile(os.path.join(current_directory, 'solver', 'ecos', 'CMakeLists.txt'),
                        os.path.join(solver_code_dir, 'CMakeLists.txt'))
        shutil.copyfile(os.path.join(current_directory, 'solver', 'ecos', 'COPYING'),
                        os.path.join(solver_code_dir, 'COPYING'))
        shutil.copyfile(os.path.join(current_directory, 'solver', 'ecos', 'COPYING'),
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

    user_p_to_canon_outdated = {user_p_name: [canon_p_ids[j] for j in np.nonzero(adjacency[:, i])[0]]
                                for i, user_p_name in enumerate(user_p_names)}
    canon_settings_names_to_types = {name: typ for name, typ in zip(canon_settings_names, canon_settings_types)}
    canon_settings_names_to_default = {name: typ for name, typ in zip(canon_settings_names, canon_settings_defaults)}

    ret_sol_func_exists = any(var_symmetric) or any([s == 1 for s in var_sizes]) or solver_name == 'ECOS'

    # 'workspace' prototypes
    with open(os.path.join(code_dir, 'c', 'include', 'cpg_workspace.h'), 'a') as f:
        utils.write_workspace_prot(f, solver_name, explicit, user_p_name_to_size_usp, user_p_writable, user_p_flat_usp,
                                   var_init, canon_p_ids, canon_p, canon_mappings, var_symmetric, canon_constants,
                                   canon_settings_names_to_types, problem_name)

    # 'workspace' definitions
    with open(os.path.join(code_dir, 'c', 'src', 'cpg_workspace.c'), 'a') as f:
        utils.write_workspace_def(f, solver_name, explicit, user_p_names, user_p_writable, user_p_flat_usp, var_init,
                                  canon_p_ids, canon_p, canon_mappings, var_symmetric, var_offsets, canon_constants,
                                  canon_settings_names_to_default, problem_name)

    # 'solve' prototypes
    with open(os.path.join(code_dir, 'c', 'include', 'cpg_solve.h'), 'a') as f:
        utils.write_solve_prot(f, solver_name, canon_p_ids, user_p_name_to_size_usp, canon_settings_names_to_types,
                               ret_sol_func_exists, problem_name)

    # 'solve' definitions
    with open(os.path.join(code_dir, 'c', 'src', 'cpg_solve.c'), 'a') as f:
        utils.write_solve_def(f, solver_name, explicit, canon_p_ids, canon_mappings, user_p_col_to_name_usp,
                              user_p_sizes_usp, user_p_name_to_size_usp, var_name_to_indices, canon_p_id_to_size,
                              type(problem.objective) == Maximize, user_p_to_canon_outdated,
                              canon_settings_names_to_types, canon_settings_names_to_default, var_symmetric,
                              canon_p_to_changes, canon_constants, nonzero_d, ret_sol_func_exists, problem_name)

    # 'example' definitions
    with open(os.path.join(code_dir, 'c', 'src', 'cpg_example.c'), 'a') as f:
        utils.write_example_def(f, solver_name, user_p_writable, var_name_to_size, problem_name)

    # adapt top-level CMakeLists.txt
    with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'r') as f:
        cmake_data = f.read()
    cmake_data = utils.replace_cmake_data(cmake_data, problem_name)
    with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'w') as f:
        f.write(cmake_data)

    # adapt solver CMakeLists.txt
    with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'a') as f:
        utils.write_canon_cmake(f, solver_name)

    # binding module prototypes
    with open(os.path.join(code_dir, 'cpp', 'include', 'cpg_module.hpp'), 'a') as f:
        utils.write_module_prot(f, solver_name, user_p_name_to_size_usp, var_name_to_size, problem_name)

    # binding module definition
    with open(os.path.join(code_dir, 'cpp', 'src', 'cpg_module.cpp'), 'a') as f:
        utils.write_module_def(f, user_p_name_to_size_usp, var_name_to_size, canon_settings_names, problem_name)

    # custom CVXPY solve method
    with open(os.path.join(code_dir, 'cpg_solver.py'), 'a') as f:
        utils.write_method(f, solver_name, code_dir, user_p_name_to_size_usp, user_p_name_to_sparsity,
                           user_p_name_to_sparsity_type, var_name_to_shape)

    # serialize problem formulation
    with open(os.path.join(code_dir, 'problem.pickle'), 'wb') as f:
        pickle.dump(cp.Problem(problem.objective, problem.constraints), f)

    # html documentation file
    with open(os.path.join(code_dir, 'README.html'), 'r') as f:
        html_data = f.read()
    html_data = utils.replace_html_data(code_dir, solver_name, explicit, html_data, user_p_name_to_size_usp,
                                        user_p_writable, var_name_to_size, user_p_total_size, canon_p_ids,
                                        canon_p_id_to_size, canon_settings_names_to_types, canon_constants,
                                        canon_mappings, ret_sol_func_exists, problem_name)
    with open(os.path.join(code_dir, 'README.html'), 'w') as f:
        f.write(html_data)

    sys.stdout.write('CVXPYGEN finished generating code.\n')

    # compile python module
    if compile_module:
        sys.stdout.write('Compiling python wrapper with CVXPYGEN ... \n')
        p_dir = os.getcwd()
        os.chdir(code_dir)
        call([sys.executable, 'setup.py', '--quiet', 'build_ext', '--inplace'])
        os.chdir(p_dir)
        sys.stdout.write("CVXPYGEN finished compiling python wrapper.\n")
