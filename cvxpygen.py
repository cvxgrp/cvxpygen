
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


def generate_code(problem, code_dir='CPG_code', explicit=True):
    """
    Generate C code for CVXPY problem and optionally compile example program
    """

    print('Generating code ...')

    # copy TEMPLATE
    if os.path.isdir(code_dir):
        shutil.rmtree(code_dir)
    shutil.copytree(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TEMPLATE'), code_dir)

    # get problem data
    data, solving_chain, inverse_data = problem.get_problem_data(solver='OSQP', gp=False, enforce_dpp=True, verbose=False)
    n_var = data['n_var']
    n_eq = data['n_eq']
    n_ineq = data['n_ineq']
    p_prob = data['param_prob']

    # get variable information
    vars = problem.variables()
    var_names = [var.name() for var in vars]
    var_ids = [var.id for var in vars]
    for inverse_data_idx in range(len(inverse_data)-1, -1, -1):
        if type(inverse_data[inverse_data_idx]) == cp.reductions.inverse_data.InverseData:
            break
    var_offsets = [inverse_data[inverse_data_idx].var_offsets[var_id] for var_id in var_ids]
    var_shapes = [var.shape for var in vars]
    var_sizes = [var.size for var in vars]
    var_symm = [var.attributes['symmetric'] or var.attributes['PSD'] or var.attributes['NSD'] for var in vars]
    var_name_to_indices = {}
    for var_name, offset, shape, symm in zip(var_names, var_offsets, var_shapes, var_symm):
        if symm:
            fill_coeff = upper_tri_to_full(shape[0])
            (_, col) = fill_coeff.nonzero()
            var_name_to_indices[var_name] = offset + col
        else:
            var_name_to_indices[var_name] = np.arange(offset, offset+np.prod(shape))

    var_name_to_size = {name: size for name, size in zip(var_names, var_sizes)}
    var_name_to_shape = {var.name(): var.shape for var in vars}
    var_init = dict()
    for var in vars:
        if len(var.shape) == 0:
            var_init[var.name()] = 0
        else:
            var_init[var.name()] = np.zeros(shape=var.shape)

    # extract csc values for individual OSQP parameters (Alu -> A, lu)
    (indices_P, indptr_P, shape_P) = p_prob.problem_data_index_P
    (indices_Alu, indptr_Alu, shape_Alu) = p_prob.problem_data_index_A
    indices_A, indptr_A, shape_A = indices_Alu[:indptr_Alu[-2]], indptr_Alu[:-1], (shape_Alu[0], shape_Alu[1] - 1)
    indices_lu, indptr_lu, shape_lu = indices_Alu[indptr_Alu[-2]:indptr_Alu[-1]], indptr_Alu[-2:] - indptr_Alu[-2], \
                                      (shape_Alu[0], 1)
    P_data_length = p_prob.reduced_P.shape[0]
    A_data_length = len(indices_A)
    lu_data_length = len(indices_lu)
    l_data_length = np.count_nonzero(indices_lu < n_eq)

    # OSQP parameters
    OSQP_p_ids = ['P', 'q', 'd', 'A', 'l', 'u']
    OSQP_p_ids_lu = OSQP_p_ids[:-2] + ['lu']
    OSQP_p_id_to_i = {k: v for v, k in enumerate(OSQP_p_ids)}
    OSQP_p_num = len(OSQP_p_ids)
    OSQP_p_sizes = [P_data_length, n_var, 1, A_data_length, n_eq, n_eq+n_ineq]
    OSQP_p_id_to_size = {k: v for k, v in zip(OSQP_p_ids, OSQP_p_sizes)}
    OSQP_p_id_to_col = {k: v for k, v in zip(OSQP_p_ids, np.cumsum([0] + OSQP_p_sizes[:-2] + [0]))}
    OSQP_p_sizes_lu = [P_data_length, n_var, 1, A_data_length, lu_data_length]
    OSQP_p_id_to_size_lu = {k: v for k, v in zip(OSQP_p_ids_lu, OSQP_p_sizes_lu)}
    OSQP_p_id_to_col_lu = {k: v for k, v in zip(OSQP_p_ids_lu, np.cumsum([0] + OSQP_p_sizes_lu[:-1] + [0]))}

    # OSQP settings
    OSQP_settings_names = ['rho', 'max_iter', 'eps_abs', 'eps_rel', 'eps_prim_inf', 'eps_dual_inf', 'alpha',
                           'scaled_termination', 'check_termination', 'warm_start']
    OSQP_settings_types = ['c_float', 'c_int', 'c_float', 'c_float', 'c_float', 'c_float', 'c_float',
                           'c_int', 'c_int', 'c_int']
    OSQP_settings_names_to_types = {name: typ for name, typ in zip(OSQP_settings_names, OSQP_settings_types)}

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

    # adjacency matrix describing OSQP_params - user_params dependencies
    adjacency = np.zeros(shape=(OSQP_p_num, user_p_num), dtype=np.bool)

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
                adjacency[OSQP_p_id_to_i['A'], j] = True
            if np.max(Alu_columns) >= n_var:
                adjacency[OSQP_p_id_to_i['u'], j] = True
                if np.min(Alu_rows) < n_eq:
                    adjacency[OSQP_p_id_to_i['l'], j] = True

    # default values of user parameters
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

    # default values of OSQP parameters via one big affine mapping
    OSQP_p = dict()

    def user_p_value(user_p_id):
        return np.array(user_p_id_to_param[user_p_id].value)

    user_p_flat = cI.get_parameter_vector(user_p_total_size, user_p_id_to_col, user_p_id_to_size, user_p_value)
    MAP = sparse.vstack([p_prob.reduced_P, p_prob.q, p_prob.reduced_A])
    OSQP_p_flat = MAP @ user_p_flat
    data_P = OSQP_p_flat[OSQP_p_id_to_col_lu['P']:OSQP_p_id_to_col_lu['P'] + OSQP_p_id_to_size_lu['P']]
    OSQP_p['P_csc'] = sparse.csc_matrix((data_P, indices_P, indptr_P), shape=shape_P)
    OSQP_p['P'] = utils.csc_to_dict(OSQP_p['P_csc'])
    OSQP_p['q'] = OSQP_p_flat[OSQP_p_id_to_col_lu['q']:OSQP_p_id_to_col_lu['q'] + OSQP_p_id_to_size_lu['q']]
    OSQP_p['d'] = OSQP_p_flat[OSQP_p_id_to_col_lu['d']:OSQP_p_id_to_col_lu['d'] + OSQP_p_id_to_size_lu['d']]
    data_A = OSQP_p_flat[OSQP_p_id_to_col_lu['A']:OSQP_p_id_to_col_lu['A'] + OSQP_p_id_to_size_lu['A']]
    OSQP_p['A_csc'] = sparse.csc_matrix((data_A, indices_A, indptr_A), shape=shape_A)
    OSQP_p['A'] = utils.csc_to_dict(OSQP_p['A_csc'])
    data_lu = OSQP_p_flat[OSQP_p_id_to_col_lu['lu']:OSQP_p_id_to_col_lu['lu'] + OSQP_p_id_to_size_lu['lu']]
    lu = sparse.csc_matrix((data_lu, indices_lu, indptr_lu), shape=shape_lu).toarray().squeeze()
    OSQP_p['l'] = np.concatenate((-lu[:n_eq], -np.inf * np.ones((n_ineq))))
    OSQP_p['u'] = -lu

    # default values of decomposed OSQP parameters
    OSQP_p_decomposed = dict()
    for i, OSQP_p_id in enumerate(OSQP_p_ids):
        matrix = np.zeros((OSQP_p_id_to_size[OSQP_p_id], np.sum(adjacency[i, :]) + 1))
        if OSQP_p_id in ['l', 'u']:
            OSQP_id_lu = 'lu'
        else:
            OSQP_id_lu = OSQP_p_id
        mapping = MAP[OSQP_p_id_to_col_lu[OSQP_id_lu]:OSQP_p_id_to_col_lu[OSQP_id_lu]+OSQP_p_id_to_size_lu[OSQP_id_lu], -1]
        if OSQP_p_id in ['l', 'u']:
            OSQP_rows = indices_lu
            if OSQP_p_id == 'l':
                n_rows = np.count_nonzero(OSQP_rows < n_eq)
            else:
                n_rows = len(mapping.indptr)-1
            for k in range(n_rows):
                matrix[OSQP_rows[k], -1] = mapping[k].toarray()
        else:
            matrix[:, -1] = mapping.toarray().squeeze()
        OSQP_p_decomposed[OSQP_p_id+'_decomposed'] = matrix

    # affine mapping for each OSQP parameter
    OSQP_mappings = []
    OSQP_p_to_changes = {}
    for OSQP_name in OSQP_p_ids:
        if OSQP_name in ['l', 'u']:
            if OSQP_name == 'l':
                data_length = l_data_length
                vec_length = n_eq
            else:
                data_length = lu_data_length
                vec_length = n_eq+n_ineq
            row_slice = slice(OSQP_p_id_to_col_lu['lu'], OSQP_p_id_to_col_lu['lu'] + data_length)
            mapping_to_sparse_vec = -MAP[row_slice, :]
            mapping_to_dense_vec = sparse.lil_matrix(np.zeros((vec_length, mapping_to_sparse_vec.shape[1])))
            for i_data in range(data_length):
                mapping_to_dense_vec[indices_lu[i_data], :] = mapping_to_sparse_vec[i_data, :]
            mapping = sparse.csr_matrix(mapping_to_dense_vec)
        else:
            row_slice = slice(OSQP_p_id_to_col_lu[OSQP_name], OSQP_p_id_to_col_lu[OSQP_name] + OSQP_p_id_to_size_lu[OSQP_name])
            mapping = MAP[row_slice, :]
        OSQP_mappings.append(mapping)
        OSQP_p_to_changes[OSQP_name] = mapping.nnz > 0

    user_p_to_OSQP_outdated = {user_p_name: [OSQP_p_ids[j] for j in np.nonzero(adjacency[:, i])[0]]
                               for i, user_p_name in enumerate(user_p_names)}

    # 'workspace' prototypes
    with open(os.path.join(code_dir, 'c/include/cpg_workspace.h'), 'a') as f:
        utils.write_workspace_extern(f, explicit, user_p_names, user_p_writable, user_p_flat, var_init, OSQP_p_ids, OSQP_p, OSQP_mappings, var_symm)

    # 'workspace' definitions
    with open(os.path.join(code_dir, 'c/src/cpg_workspace.c'), 'a') as f:
        utils.write_workspace(f, explicit, user_p_names, user_p_writable, user_p_flat, var_init, OSQP_p_ids, OSQP_p, OSQP_mappings, var_symm, var_offsets)

    # 'solve' prototypes
    with open(os.path.join(code_dir, 'c/include/cpg_solve.h'), 'a') as f:
        utils.write_solve_extern(f, user_p_name_to_size, OSQP_settings_names_to_types)

    # 'solve' definitions
    with open(os.path.join(code_dir, 'c/src/cpg_solve.c'), 'a') as f:
        utils.write_solve(f, explicit, OSQP_p_ids, OSQP_mappings, user_p_col_to_name,
                          list(user_p_id_to_size.values()), var_name_to_indices,
                          type(problem.objective) == cp.problems.objective.Maximize, user_p_to_OSQP_outdated,
                          OSQP_settings_names_to_types, var_symm, OSQP_p_to_changes)

    # 'example' definitions
    with open(os.path.join(code_dir, 'c/src/cpg_example.c'), 'a') as f:
        utils.write_main(f, user_p_writable, var_name_to_size)

    # OSQP codegen
    if os.path.isfile('emosqp.cpython-38-darwin.so'):
        os.remove('emosqp.cpython-38-darwin.so')

    myOSQP = osqp.OSQP()
    myOSQP.setup(P=OSQP_p['P_csc'], q=OSQP_p['q'], A=OSQP_p['A_csc'], l=OSQP_p['l'], u=OSQP_p['u'])
    myOSQP.codegen(os.path.join(code_dir, 'c/OSQP_code'), parameters='matrices', force_rewrite=True)

    # adapt OSQP CMakeLists.txt
    with open(os.path.join(code_dir, 'c/OSQP_code/CMakeLists.txt'), 'a') as f:
        utils.write_OSQP_CMakeLists(f)

    # binding module
    with open(os.path.join(code_dir, 'cpp/cpg_module.cpp'), 'a') as f:
        utils.write_module(f, user_p_name_to_size, var_name_to_size, OSQP_settings_names)

    # custom CVXPY solve method
    with open(os.path.join(code_dir, 'cpg_solver.py'), 'a') as f:
        utils.write_method(f, code_dir, user_p_name_to_size, var_name_to_shape)

    # serialize problem formulation
    with open(os.path.join(code_dir, 'problem.pickle'), 'wb') as f:
        pickle.dump(cp.Problem(problem.objective, problem.constraints), f)

    # create python module
    sys.stdout.write("Compiling CPG Python wrapper... \t\t\t\t\t")
    pdir = os.getcwd()
    os.chdir(code_dir)
    call([sys.executable, 'setup.py', '--quiet', 'build_ext', '--inplace'])
    os.chdir(pdir)

    # html documentation file
    with open(os.path.join(code_dir, 'README.html'), 'r') as f:
        html_data = f.read()
    html_data = utils.replace_html(code_dir, explicit, html_data, user_p_name_to_size, user_p_writable, var_name_to_size, user_p_total_size)
    with open(os.path.join(code_dir, 'README.html'), 'w') as f:
        f.write(html_data)

    print('CPG Code Generation Done.')
