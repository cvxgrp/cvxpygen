
import os
import shutil
import numpy as np
from scipy import sparse
from cvxpy.cvxcore.python import canonInterface as cI
from cvxpy.expressions.variable import upper_tri_to_full
import cvxpy as cp
import osqp
import utils
import pickle
import sys
from subprocess import call
from platform import system


def generate_code(problem, code_dir='CPG_code', solver=None, compile_module=True, explicit=False, problem_name=''):
    """
    Generate C code for CVXPY problem and (optionally) python wrapper
    """

    sys.stdout.write('Generating code with CVXPYGEN ...\n')

    current_directory = os.path.dirname(os.path.realpath(__file__))

    # copy TEMPLATE
    if os.path.isdir(code_dir):
        shutil.rmtree(code_dir)
    shutil.copytree(os.path.join(current_directory, 'TEMPLATE'), code_dir)

    # get problem data
    data, solving_chain, inverse_data = problem.get_problem_data(solver=solver, gp=False, enforce_dpp=True,
                                                                 verbose=False)

    solver_name = solving_chain.solver.name()
    p_prob = data['param_prob']

    # get variable information
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
    user_p_col_to_name = {k: v for k, v in zip(user_p_id_to_col.values(), user_p_names)}
    user_p_id_to_size = p_prob.param_id_to_size
    user_p_id_to_param = p_prob.id_to_param
    user_p_total_size = p_prob.total_param_size
    user_p_name_to_size = {name: size for name, size in zip(user_p_names, user_p_id_to_size.values())}
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
            user_p_writable[p_name] = p.value.flatten(order='F')

    def user_p_value(user_p_id):
        return np.array(user_p_id_to_param[user_p_id].value)
    user_p_flat = cI.get_parameter_vector(user_p_total_size, user_p_id_to_col, user_p_id_to_size, user_p_value)

    canon_p = {}
    canon_mappings = []
    canon_p_to_changes = {}
    canon_constants = {}
    canon_p_id_to_size = {}
    canon_settings_names_to_default = {}

    if solver_name == 'OSQP':

        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        # extract csc values for individual osqp parameters (Alu -> A, lu)
        (indices_P, indptr_P, shape_P) = p_prob.problem_data_index_P
        (indices_Alu, indptr_Alu, shape_Alu) = p_prob.problem_data_index_A
        indices_A, indptr_A, shape_A = indices_Alu[:indptr_Alu[-2]], indptr_Alu[:-1], (shape_Alu[0], shape_Alu[1] - 1)
        indices_lu, indptr_lu, shape_lu = indices_Alu[indptr_Alu[-2]:indptr_Alu[-1]], indptr_Alu[-2:] - indptr_Alu[-2], \
                                                    (shape_Alu[0], 1)
        P_data_length = p_prob.reduced_P.shape[0]
        A_data_length = len(indices_A)
        lu_data_length = len(indices_lu)
        l_data_length = np.count_nonzero(indices_lu < n_eq)

        # osqp parameters
        canon_p_ids = ['P', 'q', 'd', 'A', 'l', 'u']
        osqp_p_ids_lu = canon_p_ids[:-2] + ['lu']
        osqp_p_id_to_i = {k: v for v, k in enumerate(canon_p_ids)}
        osqp_p_num = len(canon_p_ids)
        osqp_p_sizes_lu = [P_data_length, n_var, 1, A_data_length, lu_data_length]
        osqp_p_id_to_size_lu = {k: v for k, v in zip(osqp_p_ids_lu, osqp_p_sizes_lu)}
        osqp_p_id_to_col_lu = {k: v for k, v in zip(osqp_p_ids_lu, np.cumsum([0] + osqp_p_sizes_lu[:-1] + [0]))}

        # OSQP settings
        canon_settings_names = ['rho', 'max_iter', 'eps_abs', 'eps_rel', 'eps_prim_inf', 'eps_dual_inf', 'alpha',
                                'scaled_termination', 'check_termination', 'warm_start']
        canon_settings_types = ['c_float', 'c_int', 'c_float', 'c_float', 'c_float', 'c_float', 'c_float',
                                'c_int', 'c_int', 'c_int']
        canon_settings_names_to_types = {name: typ for name, typ in zip(canon_settings_names, canon_settings_types)}

        # adjacency matrix describing OSQP_params - user_params dependencies
        adjacency = np.zeros(shape=(osqp_p_num, user_p_num), dtype=bool)

        # adjacency P, q, d
        for j in range(user_p_num):
            column_slice = slice(user_p_id_to_col[user_p_ids[j]], user_p_id_to_col[user_p_ids[j + 1]])
            if p_prob.reduced_P[:, column_slice].data.size > 0:
                adjacency[0, j] = True
            if p_prob.q[:-1, column_slice].data.size > 0:
                adjacency[1, j] = True
            if p_prob.q[-1, column_slice].data.size > 0:
                adjacency[2, j] = True

        # adjacency A, l, u
        Alu_dummy = sparse.csc_matrix((np.zeros((len(indices_Alu),)), indices_Alu, indptr_Alu), shape=shape_Alu).tocoo()
        for j in range(user_p_num):
            column_slice = slice(user_p_id_to_col[user_p_ids[j]], user_p_id_to_col[user_p_ids[j + 1]])
            flat_adjacent_idx = np.unique(p_prob.reduced_A[:, column_slice].tocoo().row)
            Alu_rows = Alu_dummy.row[flat_adjacent_idx]
            Alu_columns = Alu_dummy.col[flat_adjacent_idx]
            if Alu_rows.size > 0:
                if np.min(Alu_columns) < n_var:
                    adjacency[osqp_p_id_to_i['A'], j] = True
                if np.max(Alu_columns) >= n_var:
                    adjacency[osqp_p_id_to_i['u'], j] = True
                    if np.min(Alu_rows) < n_eq:
                        adjacency[osqp_p_id_to_i['l'], j] = True

        # default values of OSQP parameters via one big affine mapping
        MAP = sparse.vstack([p_prob.reduced_P, p_prob.q, p_prob.reduced_A])
        OSQP_p_flat = MAP @ user_p_flat
        data_P = OSQP_p_flat[osqp_p_id_to_col_lu['P']:osqp_p_id_to_col_lu['P'] + osqp_p_id_to_size_lu['P']]
        canon_p['P_csc'] = sparse.csc_matrix((data_P, indices_P, indptr_P), shape=shape_P)
        canon_p['P'] = utils.csc_to_dict(canon_p['P_csc'])
        canon_p['q'] = OSQP_p_flat[osqp_p_id_to_col_lu['q']:osqp_p_id_to_col_lu['q'] + osqp_p_id_to_size_lu['q']]
        canon_p['d'] = OSQP_p_flat[osqp_p_id_to_col_lu['d']:osqp_p_id_to_col_lu['d'] + osqp_p_id_to_size_lu['d']]
        data_A = OSQP_p_flat[osqp_p_id_to_col_lu['A']:osqp_p_id_to_col_lu['A'] + osqp_p_id_to_size_lu['A']]
        canon_p['A_csc'] = sparse.csc_matrix((data_A, indices_A, indptr_A), shape=shape_A)
        canon_p['A'] = utils.csc_to_dict(canon_p['A_csc'])
        data_lu = OSQP_p_flat[osqp_p_id_to_col_lu['lu']:osqp_p_id_to_col_lu['lu'] + osqp_p_id_to_size_lu['lu']]
        lu = sparse.csc_matrix((data_lu, indices_lu, indptr_lu), shape=shape_lu).toarray().squeeze()
        canon_p['l'] = np.concatenate((-lu[:n_eq], -np.inf * np.ones(n_ineq)))
        canon_p['u'] = -lu

        # affine mapping for each OSQP parameter
        for osqp_name in canon_p_ids:
            if osqp_name in ['l', 'u']:
                if osqp_name == 'l':
                    data_length = l_data_length
                    vec_length = n_eq
                else:
                    data_length = lu_data_length
                    vec_length = n_eq+n_ineq
                row_slice = slice(osqp_p_id_to_col_lu['lu'], osqp_p_id_to_col_lu['lu'] + data_length)
                mapping_to_sparse_vec = -MAP[row_slice, :]
                mapping_to_dense_vec = sparse.lil_matrix(np.zeros((vec_length, mapping_to_sparse_vec.shape[1])))
                for i_data in range(data_length):
                    mapping_to_dense_vec[indices_lu[i_data], :] = mapping_to_sparse_vec[i_data, :]
                mapping = sparse.csr_matrix(mapping_to_dense_vec)
            else:
                row_slice = slice(osqp_p_id_to_col_lu[osqp_name],
                                  osqp_p_id_to_col_lu[osqp_name] + osqp_p_id_to_size_lu[osqp_name])
                mapping = MAP[row_slice, :]
            canon_mappings.append(mapping)
            canon_p_to_changes[osqp_name] = mapping[:, :-1].nnz > 0

        # OSQP codegen
        myOSQP = osqp.OSQP()
        myOSQP.setup(P=canon_p['P_csc'], q=canon_p['q'], A=canon_p['A_csc'], l=canon_p['l'], u=canon_p['u'])
        if system() == 'Windows':
            cmake_generator = 'MinGW Makefiles'
        elif system() == 'Linux' or system() == 'Darwin':
            cmake_generator = 'Unix Makefiles'
        else:
            raise OSError('Unknown operating system!')
        myOSQP.codegen(os.path.join(code_dir, 'c', 'solver_code'), project_type=cmake_generator, parameters='matrices',
                       force_rewrite=True)

    elif solver_name == 'ECOS':

        n_var = p_prob.x.size
        n_eq = p_prob.cone_dims.zero
        n_ineq = data['G'].shape[0]

        canon_constants['n'] = n_var
        canon_constants['m'] = n_ineq
        canon_constants['p'] = n_eq
        canon_constants['l'] = p_prob.cone_dims.nonneg
        canon_constants['n_cones'] = len(p_prob.cone_dims.soc)
        canon_constants['q'] = np.array(p_prob.cone_dims.soc)
        canon_constants['e'] = p_prob.cone_dims.exp

        # affine mapping for each OSQP parameter
        canon_p_ids = ['c', 'd', 'A', 'b', 'G', 'h']
        adjacency = np.zeros(shape=(len(canon_p_ids), user_p_num), dtype=bool)

        indices_AGbh, indptr_AGbh, shape_AGbh = p_prob.problem_data_index
        n_data_AGbh = len(indices_AGbh)
        n_data_bh = indptr_AGbh[-1] - indptr_AGbh[-2]
        n_data_AG = n_data_AGbh - n_data_bh

        mapping_rows_eq = np.nonzero(indices_AGbh < n_eq)[0]
        mapping_rows_ineq = np.nonzero(indices_AGbh >= n_eq)[0]

        for i, p_id in enumerate(canon_p_ids):

            mapping_rows = []
            indices = []
            indptr_original = []
            shape = ()

            if p_id == 'c':
                mapping = p_prob.c[:-1]
            elif p_id == 'd':
                mapping = p_prob.c[-1]
            elif p_id == 'A':
                mapping_rows = mapping_rows_eq[mapping_rows_eq < n_data_AG]
                shape = (n_eq, n_var)
            elif p_id == 'G':
                mapping_rows = mapping_rows_ineq[mapping_rows_ineq < n_data_AG]
                shape = (n_ineq, n_var)
            elif p_id == 'b':
                mapping_rows = mapping_rows_eq[mapping_rows_eq >= n_data_AG]
                shape = (n_eq, 1)
            elif p_id == 'h':
                mapping_rows = mapping_rows_ineq[mapping_rows_ineq >= n_data_AG]
                shape = (n_ineq, 1)
            else:
                raise ValueError('Unknown ECOS parameter name: "%s"' % p_id)

            if p_id in ['A', 'b']:
                indices = indices_AGbh[mapping_rows]
            elif p_id in ['G', 'h']:
                indices = indices_AGbh[mapping_rows] - n_eq

            if p_id in ['A', 'G']:
                mapping = -p_prob.reduced_A[mapping_rows]
                indptr_original = indptr_AGbh[:-1]
            elif p_id in ['b', 'h']:
                mapping_to_sparse = p_prob.reduced_A[mapping_rows]
                mapping_to_dense = sparse.lil_matrix(np.zeros((shape[0], mapping_to_sparse.shape[1])))
                for i_data in range(mapping_to_sparse.shape[0]):
                    mapping_to_dense[indices[i_data], :] = mapping_to_sparse[i_data, :]
                mapping = sparse.csc_matrix(mapping_to_dense)

            canon_mappings.append(mapping.tocsr())
            canon_p_to_changes[p_id] = mapping[:, :-1].nnz > 0
            canon_p_id_to_size[p_id] = mapping.shape[0]

            for j in range(user_p_num):
                column_slice = slice(user_p_id_to_col[user_p_ids[j]], user_p_id_to_col[user_p_ids[j + 1]])
                if mapping[:, column_slice].nnz > 0:
                    adjacency[i, j] = True

            canon_p_data = mapping @ user_p_flat

            if p_id.isupper():
                indptr = 0 * indptr_original
                for r in mapping_rows:
                    for c in range(shape[1]):
                        if indptr_original[c] <= r < indptr_original[c + 1]:
                            indptr[c + 1:] += 1
                            break
                csc_mat = sparse.csc_matrix((canon_p_data, indices, indptr), shape=shape)
                if p_id in ['A', 'G']:
                    canon_p[p_id] = utils.csc_to_dict(csc_mat)
            else:
                canon_p[p_id] = canon_p_data

        canon_settings_names = ['feastol', 'abstol', 'reltol', 'feastol_inacc', 'abstol_inacc', 'reltol_inacc', 'maxit']
        canon_settings_types = ['c_float', 'c_float', 'c_float', 'c_float', 'c_float', 'c_float', 'c_int']
        canon_settins_defaults = ['1e-8', '1e-8', '1e-8', '1e-4', '5e-5', '5e-5', '100']
        canon_settings_names_to_types = {name: typ for name, typ in zip(canon_settings_names, canon_settings_types)}
        canon_settings_names_to_default = {name: typ for name, typ in zip(canon_settings_names, canon_settins_defaults)}

        # copy sources
        solver_code_dir = os.path.join(code_dir, 'c', 'solver_code')
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'external', 'ecos_bb']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(current_directory, 'solver', 'ecos', dtc), os.path.join(solver_code_dir, dtc))
        shutil.copyfile(os.path.join(current_directory, 'solver', 'ecos', 'CMakeLists.txt'),
                        os.path.join(solver_code_dir, 'CMakeLists.txt'))

        # adjust print level
        with open(os.path.join(code_dir, 'c', 'solver_code', 'include', 'glblopts.h'), 'r') as f:
            glbl_opts_data = f.read()
        glbl_opts_data = glbl_opts_data.replace('#define PRINTLEVEL (2)', '#define PRINTLEVEL (0)')
        with open(os.path.join(code_dir, 'c', 'solver_code', 'include', 'glblopts.h'), 'w') as f:
            f.write(glbl_opts_data)

        # adjust top-level CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'r') as f:
            CMakeLists_data = f.read()
        indent = ' ' * 6
        CMakeLists_data = CMakeLists_data.replace('${CMAKE_CURRENT_SOURCE_DIR}/solver_code/include',
                                                  '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/include\n' +
                                                  indent + '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/external/SuiteSparse_config\n' +
                                                  indent + '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/external/amd/include\n' +
                                                  indent + '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/external/ldl/include')
        with open(os.path.join(code_dir, 'c', 'CMakeLists.txt'), 'w') as f:
            f.write(CMakeLists_data)

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
        with open(os.path.join(code_dir, 'setup.py'), 'w') as f:
            f.write(setup_text)

    else:
        raise ValueError("Problem class cannot be addressed by the OSQP or ECOS solver!")

    user_p_to_canon_outdated = {user_p_name: [canon_p_ids[j] for j in np.nonzero(adjacency[:, i])[0]]
                                for i, user_p_name in enumerate(user_p_names)}

    # 'workspace' prototypes
    with open(os.path.join(code_dir, 'c', 'include', 'cpg_workspace.h'), 'a') as f:
        utils.write_workspace_prot(f, solver_name, explicit, user_p_names, user_p_writable, user_p_flat, var_init,
                                   canon_p_ids, canon_p, canon_mappings, var_symmetric, canon_constants,
                                   canon_settings_names_to_types)

    # 'workspace' definitions
    with open(os.path.join(code_dir, 'c', 'src', 'cpg_workspace.c'), 'a') as f:
        utils.write_workspace_def(f, solver_name, explicit, user_p_names, user_p_writable, user_p_flat, var_init,
                                  canon_p_ids, canon_p, canon_mappings, var_symmetric, var_offsets, canon_constants,
                                  canon_settings_names_to_default)

    # 'solve' prototypes
    with open(os.path.join(code_dir, 'c', 'include', 'cpg_solve.h'), 'a') as f:
        utils.write_solve_prot(f, solver_name, canon_p_ids, user_p_name_to_size, canon_settings_names_to_types)

    # 'solve' definitions
    with open(os.path.join(code_dir, 'c', 'src', 'cpg_solve.c'), 'a') as f:
        utils.write_solve_def(f, solver_name, explicit, canon_p_ids, canon_mappings, user_p_col_to_name,
                              list(user_p_id_to_size.values()), var_name_to_indices, canon_p_id_to_size,
                              type(problem.objective) == cp.problems.objective.Maximize, user_p_to_canon_outdated,
                              canon_settings_names_to_types, canon_settings_names_to_default, var_symmetric,
                              canon_p_to_changes, canon_constants)

    # 'example' definitions
    with open(os.path.join(code_dir, 'c', 'src', 'cpg_example.c'), 'a') as f:
        utils.write_example_def(f, solver_name, user_p_writable, var_name_to_size)

    # adapt solver CMakeLists.txt
    with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'a') as f:
        utils.write_canon_CMakeLists(f, solver_name)

    # binding module prototypes
    with open(os.path.join(code_dir, 'cpp', 'include', 'cpg_module.hpp'), 'a') as f:
        utils.write_module_prot(f, solver_name, user_p_name_to_size, var_name_to_size, problem_name)

    # binding module definition
    with open(os.path.join(code_dir, 'cpp', 'src', 'cpg_module.cpp'), 'a') as f:
        utils.write_module_def(f, user_p_name_to_size, var_name_to_size, canon_settings_names, problem_name)

    # custom CVXPY solve method
    with open(os.path.join(code_dir, 'cpg_solver.py'), 'a') as f:
        utils.write_method(f, solver_name, code_dir, user_p_name_to_size, var_name_to_shape)

    # serialize problem formulation
    with open(os.path.join(code_dir, 'problem.pickle'), 'wb') as f:
        pickle.dump(cp.Problem(problem.objective, problem.constraints), f)

    # compile python module
    if compile_module:
        sys.stdout.write('Compiling python wrapper with CVXPYGEN ... \n')
        p_dir = os.getcwd()
        os.chdir(code_dir)
        call([sys.executable, 'setup.py', '--quiet', 'build_ext', '--inplace'])
        os.chdir(p_dir)
        sys.stdout.write("CVXPYGEN finished compiling python wrapper.\n")

    # html documentation file
    with open(os.path.join(code_dir, 'README.html'), 'r') as f:
        html_data = f.read()
    html_data = utils.replace_html_data(code_dir, solver_name, explicit, html_data, user_p_name_to_size,
                                        user_p_writable, var_name_to_size, user_p_total_size, canon_p_ids,
                                        canon_p_id_to_size, canon_settings_names_to_types, canon_constants)
    with open(os.path.join(code_dir, 'README.html'), 'w') as f:
        f.write(html_data)

    sys.stdout.write('CVXPYGEN finished generating code.\n')
