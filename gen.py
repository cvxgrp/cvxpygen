
import os
import shutil
import numpy as np
from scipy import sparse
from cvxpy.cvxcore.python import canonInterface as cI
from osqp.codegen import utils as osqp_utils
import osqp
import utils


def csc_to_dict(m):
    """
    Convert scipy csc matrix to dict that can be passed to osqp_utils.write_mat()
    """

    d = dict()
    d['i'] = m.indices
    d['p'] = m.indptr
    d['x'] = m.data
    d['nzmax'] = m.nnz
    (d['m'], d['n']) = m.shape
    d['nz'] = -1

    return d


def replace_inf(v):
    """
    Replace infinity by large number
    """

    # check if dealing with csc dict or numpy array
    if type(v) == dict:
        sign = np.sign(v['x'])
        idx = np.isinf(v['x'])
        v['x'][idx] = 1e30 * sign[idx]
    else:
        sign = np.sign(v)
        idx = np.isinf(v)
        v[idx] = 1e30 * sign[idx]

    return v


def generate_code(problem, code_dir='cpg_code', compile=True):
    """
    Generate C code for CVXPY problem and optionally compile example program
    """

    print('Generating code ...')

    # copy TEMPLATE
    if os.path.isdir(code_dir):
        shutil.rmtree(code_dir)
    shutil.copytree('TEMPLATE', code_dir)

    # get problem data
    data, solving_chain, inverse_data = problem.get_problem_data(solver='OSQP', gp=False, enforce_dpp=True, verbose=False)
    n_var = data['n_var']
    n_eq = data['n_eq']
    n_ineq = data['n_ineq']
    p_prob = data['param_prob']

    # get variable information
    var_names = [var.name() for var in problem.variables()]
    var_ids = list(inverse_data[1].id_map.keys())
    var_offsets = [inverse_data[2].var_offsets[var_id] for var_id in var_ids]
    var_sizes = [np.prod(inverse_data[2].var_shapes[var_id]) for var_id in var_ids]
    var_name_to_indices = {var_name: np.arange(offset, offset+size)
                           for var_name, offset, size in zip(var_names, var_offsets, var_sizes)}
    var_name_to_size = {name: size for name, size in zip(var_names, var_sizes)}
    var_init = {var.name(): np.zeros(shape=var.shape) for var in problem.variables()}

    # extract csc values for individual OSQP parameters (Alu -> A, lu)
    (indices_P, indptr_P, shape_P) = p_prob.problem_data_index_P
    (indices_Alu, indptr_Alu, shape_Alu) = p_prob.problem_data_index_A
    indices_A, indptr_A, shape_A = indices_Alu[:indptr_Alu[-2]], indptr_Alu[:-1], (shape_Alu[0], shape_Alu[1] - 1)
    indices_lu, indptr_lu, shape_lu = indices_Alu[indptr_Alu[-2]:indptr_Alu[-1]], indptr_Alu[-2:] - indptr_Alu[-2], \
                                      (shape_Alu[0], 1)
    P_data_length = p_prob.reduced_P.shape[0]
    A_data_length = len(indices_A)
    lu_data_length = len(indices_lu)

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

    # user parameters
    user_p_num = len(p_prob.parameters)
    user_p_names = [par.name() for par in p_prob.parameters]
    user_p_ids = list(p_prob.param_id_to_col.keys())
    user_p_id_to_col = p_prob.param_id_to_col
    user_p_col_to_name = {k: v for k, v in zip(user_p_id_to_col.values(), user_p_names)}
    user_p_id_to_size = p_prob.param_id_to_size
    user_p_id_to_param = p_prob.id_to_param
    user_p_total_size = p_prob.total_param_size

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
    np.random.seed(26)
    user_p_writable = dict()
    for p_name, p in zip(user_p_names, p_prob.parameters):
        if p.size == 1:
            # dealing with scalar, treating as vector
            p.value = np.array(np.random.rand())
            user_p_writable[p_name] = p.value.reshape((1,))
        elif np.max(p.shape) == p.size:
            # dealing with vector
            p.value = np.random.rand(p.shape[0], 1)
            user_p_writable[p_name] = p.value.squeeze()
        else:
            # dealing with matrix
            p.value = np.random.rand(p.shape[0], p.shape[1])
            user_p_writable[p_name] = p.value.flatten(order='F')

    # default values of OSQP parameters via one big affine mapping
    OSQP_p = dict()

    def user_p_value(user_p_id):
        return user_p_id_to_param[user_p_id].value

    user_p_flat = cI.get_parameter_vector(user_p_total_size, user_p_id_to_col, user_p_id_to_size, user_p_value)
    MAP = sparse.vstack([p_prob.reduced_P, p_prob.q, p_prob.reduced_A])
    OSQP_p_flat = MAP @ user_p_flat
    data_P = OSQP_p_flat[OSQP_p_id_to_col_lu['P']:OSQP_p_id_to_col_lu['P'] + OSQP_p_id_to_size_lu['P']]
    OSQP_p['P_csc'] = sparse.csc_matrix((data_P, indices_P, indptr_P), shape=shape_P)
    OSQP_p['P'] = csc_to_dict(OSQP_p['P_csc'])
    OSQP_p['q'] = OSQP_p_flat[OSQP_p_id_to_col_lu['q']:OSQP_p_id_to_col_lu['q'] + OSQP_p_id_to_size_lu['q']]
    OSQP_p['d'] = OSQP_p_flat[OSQP_p_id_to_col_lu['d']:OSQP_p_id_to_col_lu['d'] + OSQP_p_id_to_size_lu['d']]
    data_A = OSQP_p_flat[OSQP_p_id_to_col_lu['A']:OSQP_p_id_to_col_lu['A'] + OSQP_p_id_to_size_lu['A']]
    OSQP_p['A_csc'] = sparse.csc_matrix((data_A, indices_A, indptr_A), shape=shape_A)
    OSQP_p['A'] = csc_to_dict(OSQP_p['A_csc'])
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
            OSQP_rows = indices_Alu[indptr_Alu[-2]:indptr_Alu[-1]]
            if OSQP_p_id == 'l':
                n_rows = np.count_nonzero(OSQP_rows < n_eq)
            else:
                n_rows = len(mapping.indptr)-1
            for k in range(n_rows):
                matrix[OSQP_rows[k], -1] = mapping[k].toarray()
        else:
            matrix[:, -1] = mapping.toarray().squeeze()
        OSQP_p_decomposed[OSQP_p_id+'_decomposed'] = matrix

    # 'types' prototypes
    with open(os.path.join(code_dir, 'include/cpg_types.h'), 'a') as f:
        utils.write_types(f, user_p_names)

    # 'work' prototypes
    with open(os.path.join(code_dir, 'include/cpg_workspace.h'), 'a') as f:
        for name, value in user_p_writable.items():
            osqp_utils.write_vec_extern(f, value, name, 'c_float')
        utils.write_struct_extern(f, 'CPG_Workspace', 'CPG_Workspace_t')
        for name, value in var_init.items():
            osqp_utils.write_vec_extern(f, value, name, 'c_float')

        for OSQP_p_id in OSQP_p_ids:
            utils.write_osqp_extern(f, OSQP_p[OSQP_p_id], OSQP_p_id)
        utils.write_struct_extern(f, 'OSQP_Workspace', 'OSQP_Workspace_t')

    # 'work' definitions
    with open(os.path.join(code_dir, 'src/cpg_workspace.c'), 'a') as f:
        for name, value in user_p_writable.items():
            osqp_utils.write_vec(f, value, name, 'c_float')
        utils.write_struct(f, user_p_names, user_p_names, 'CPG_Workspace', 'CPG_Workspace_t')
        for name, value in var_init.items():
            osqp_utils.write_vec(f, value, name, 'c_float')

        for OSQP_p_id in OSQP_p_ids:
            utils.write_osqp(f, replace_inf(OSQP_p[OSQP_p_id]), OSQP_p_id)
        utils.write_struct(f, OSQP_p_ids, OSQP_p_ids, 'OSQP_Workspace', 'OSQP_Workspace_t')

    # 'update' prototypes
    with open(os.path.join(code_dir, 'include/cpg_update.h'), 'a') as f:
        utils.write_update_extern(f)

    # 'update' definitions
    with open(os.path.join(code_dir, 'src/cpg_update.c'), 'a') as f:
        mappings = []
        for i in range(OSQP_p_num):
            if i >= 4:
                OSQP_p_id_lu = 'lu'
            else:
                OSQP_p_id_lu = OSQP_p_ids_lu[i]
            row_slice = slice(OSQP_p_id_to_col_lu[OSQP_p_id_lu],
                              OSQP_p_id_to_col_lu[OSQP_p_id_lu] + OSQP_p_id_to_size_lu[OSQP_p_id_lu])
            mappings.append(MAP[row_slice, :])
        nonconstant_OSQP_names = [n for (n, b) in zip(OSQP_p_ids, np.sum(adjacency, axis=1) > 0) if b]
        utils.write_update(f, OSQP_p_ids, nonconstant_OSQP_names, mappings, user_p_col_to_name,
                           list(user_p_id_to_size.values()), n_eq, p_prob.problem_data_index_A, var_name_to_indices)

    # 'solve' prototypes
    with open(os.path.join(code_dir, 'include/cpg_solve.h'), 'a') as f:
        utils.write_solve_extern(f)

    # 'solve' definitions
    with open(os.path.join(code_dir, 'src/cpg_solve.c'), 'a') as f:
        utils.write_solve(f)

    # 'example' definitions
    with open(os.path.join(code_dir, 'src/cpg_example.c'), 'a') as f:
        utils.write_main(f, user_p_writable, var_name_to_size)

    # OSQP codegen
    if os.path.isfile('emosqp.cpython-38-darwin.so'):
        os.remove('emosqp.cpython-38-darwin.so')

    myOSQP = osqp.OSQP()
    myOSQP.setup(P=OSQP_p['P_csc'], q=OSQP_p['q'], A=OSQP_p['A_csc'], l=OSQP_p['l'], u=OSQP_p['u'])
    myOSQP.codegen(os.path.join(code_dir, 'OSQP_code'), parameters='matrices', force_rewrite=True)

    # adapt OSQP CMakeLists.txt
    with open(os.path.join(code_dir, 'OSQP_code/CMakeLists.txt'), 'a') as f:
        utils.write_OSQP_CMakeLists(f)

    print('Done.')

    # compile code if wished
    if compile:
        print('Compiling code ...')
        os.system('cd ' + os.path.join(code_dir, 'build') + ' && cmake .. && make')
        print('Done.')
