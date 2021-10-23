
import numpy as np
from osqp.codegen import utils as osqp_utils


sign_to_str = {1: '', -1: '-'}


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


def param_is_empty(param):

    if type(param) == dict:
        return param['x'].size == 0
    else:
        return param.size == 0


def write_canonicalize_explicit(f, canon_name, s, mapping, base_cols, user_p_sizes, user_p_col_to_name):
    """
    Write function to compute canonical parameter value
    """

    for row in range(len(mapping.indptr)-1):
        expr = ''
        expr_is_const = True
        data = mapping.data[mapping.indptr[row]:mapping.indptr[row + 1]]
        columns = mapping.indices[mapping.indptr[row]:mapping.indptr[row + 1]]
        for (datum, col) in zip(data, columns):
            ex = '(%.20f)+' % datum
            for i, user_p_col in enumerate(base_cols):
                if user_p_col + user_p_sizes[i] > col:
                    expr_is_const = False
                    user_name = user_p_col_to_name[user_p_col]
                    if abs(datum) == 1:
                        ex = '(%sCPG_Params.%s[%d])+' % (sign_to_str[datum], user_name, col - user_p_col)
                    else:
                        ex = '(%.20f*CPG_Params.%s[%d])+' % (datum, user_name, col - user_p_col)
                    break
            expr += ex
        expr = expr[:-1]
        if data.size > 0 and expr_is_const is False:
            f.write('Canon_Params.%s%s[%d] = %s;\n' % (canon_name, s, row, expr))


def write_canonicalize(f, canon_name, s, mapping):
    """
    Write function to compute canonical parameter value
    """

    f.write('// reset values to zero\n')
    f.write('for(i=0; i<%d; i++){\n' % mapping.shape[0])
    f.write('Canon_Params.%s%s[i] = 0;\n' % (canon_name, s))
    f.write('}\n')

    f.write('// compute sparse matrix multiplication\n')
    f.write('for(i=0; i<%d; i++){\n' % mapping.shape[0])
    f.write('for(j=Canon_%s_map.p[i]; j<Canon_%s_map.p[i+1]; j++){\n' % (canon_name, canon_name))
    f.write('Canon_Params.%s%s[i] += Canon_%s_map.x[j]*CPG_Params_Vec[Canon_%s_map.i[j]];\n' %
            (canon_name, s, canon_name, canon_name))
    f.write('}\n')
    f.write('}\n')


def write_param_def(f, param, name):
    """
    Use osqp.codegen.utils for writing vectors and matrices
    """
    if not param_is_empty(param):
        if name.isupper():
            osqp_utils.write_mat(f, param, 'Canon_' + name)
        elif name == 'd':
            f.write('c_float Canon_d = %.20f;\n' % param[0])
        else:
            osqp_utils.write_vec(f, param, 'Canon_' + name, 'c_float')


def write_param_prot(f, param, name):
    """
    Use osqp.codegen.utils for writing vectors and matrices
    """
    if not param_is_empty(param):
        if name.isupper():
            osqp_utils.write_mat_extern(f, param, 'Canon_' + name)
        elif name == 'd':
            f.write('extern c_float Canon_d;\n')
        else:
            osqp_utils.write_vec_extern(f, param, 'Canon_' + name, 'c_float')


def write_dense_mat_def(f, mat, name):
    """
    Write dense matrix to file
    """

    f.write('c_float %s[%d] = {\n' % (name, mat.size))

    # represent matrix as vector (Fortran style)
    for j in range(mat.shape[1]):
        for i in range(mat.shape[0]):
            f.write('(c_float)%.20f,\n' % mat[i, j])

    f.write('};\n')


def write_dense_mat_prot(f, mat, name):
    """
    Write dense matrix to file
    """

    f.write("extern c_float %s[%d];\n" % (name, mat.size))


def write_struct_def(f, fields, casts, values, name, typ):
    """
    Write structure to file
    """

    f.write('%s %s = {\n' % (typ, name))

    # write structure fields
    for field, cast, value in zip(fields, casts, values):
        if value == '0':
            cast = ''
        f.write('.%s = %s%s,\n' % (field, cast, value))

    f.write('};\n')


def write_struct_prot(f, name, typ):
    """
    Write structure to file
    """

    f.write("extern %s %s;\n" % (typ, name))


def write_ecos_setup(f, canon_constants):
    """
    Write ECOS setup function to file
    """
    n = canon_constants['n']
    m = canon_constants['m']
    p = canon_constants['p']
    l = canon_constants['l']
    n_cones = canon_constants['n_cones']
    e = canon_constants['e']

    if p == 0:
        Ax_str = Ap_str = Ai_str = b_str = '0'
    else:
        Ax_str = 'Canon_Params.A->x'
        Ap_str = 'Canon_Params.A->p'
        Ai_str = 'Canon_Params.A->i'
        b_str = 'Canon_Params.b'

    f.write('ecos_workspace = ECOS_setup(%d, %d, %d, %d, %d, (int *) &ecos_q, %d, '
            'Canon_Params.G->x, Canon_Params.G->p, Canon_Params.G->i, '
            '%s, %s, %s, '
            'Canon_Params.c, Canon_Params.h, %s);\n' %
            (n, m, p, l, n_cones, e, Ax_str, Ap_str, Ai_str, b_str))


def write_workspace_def(f, solver_name, explicit, user_p_names, user_p_writable, user_p_flat, var_init, canon_p_ids,
                        canon_p, canon_mappings, var_symmetric, var_offsets, canon_constants):

    f.write('\n#include "cpg_workspace.h"\n')
    if solver_name == 'OSQP':
        f.write('#include "workspace.h"\n')

    canon_casts = []

    f.write('// Canonical parameters\n')
    for canon_p_id in canon_p_ids:
        write_param_def(f, replace_inf(canon_p[canon_p_id]), canon_p_id)
        if canon_p_id.isupper() or canon_p_id == 'd':
            canon_casts.append('')
        else:
            canon_casts.append('(c_float *) ')

    f.write('\n// Struct containing parameters accepted by canonical solver\n')

    struct_values = []
    for canon_p_id in canon_p_ids:
        if type(canon_p[canon_p_id]) == dict:
            length = len(canon_p[canon_p_id]['x'])
        else:
            length = len(canon_p[canon_p_id])
        if length > 0:
            struct_values.append('&Canon_%s' % canon_p_id)
        else:
            struct_values.append('0')

    write_struct_def(f, canon_p_ids, canon_casts, struct_values, 'Canon_Params', 'Canon_Params_t')

    if solver_name == 'ECOS':
        f.write('\n// ECOS array of SOC dimensions\n')
        osqp_utils.write_vec(f, canon_constants['q'], 'ecos_q', 'c_int')
        f.write('\n// ECOS workspace\n')
        f.write('pwork* ecos_workspace = 0;\n')

    if explicit:
        f.write('\n// User-defined parameters\n')

        user_casts = []
        for name, value in user_p_writable.items():
            if np.isscalar(value):
                f.write('c_float %s = %.20f;\n' % (name, value))
                user_casts.append('')
            else:
                osqp_utils.write_vec(f, value, name, 'c_float')
                user_casts.append('(c_float *) ')

        f.write('\n// Struct containing all user-defined parameters\n')
        write_struct_def(f, user_p_names, user_casts, ['&'+name for name in user_p_names], 'CPG_Params', 'CPG_Params_t')
    else:
        f.write('\n// Sparse mappings from user-defined to canonical parameters\n')
        for name, mapping in zip(canon_p_ids, canon_mappings):
            if mapping.nnz > 0:
                osqp_utils.write_mat(f, csc_to_dict(mapping), 'Canon_%s_map' % name)

        f.write('\n// Vector containing flattened user-defined parameters\n')
        osqp_utils.write_vec(f, user_p_flat, 'CPG_Params_Vec', 'c_float')

    f.write('\n// Value of the objective function\n')
    f.write('c_float objective_value = 0;\n')

    results_cast = ['']

    f.write('\n// User-defined variables\n')
    for (name, value), symm in zip(var_init.items(), var_symmetric):
        results_cast.append('(c_float *) ')
        if symm or solver_name == 'ECOS':
            if np.isscalar(value):
                f.write('c_float %s = %.20f;\n' % (name, value))
            else:
                osqp_utils.write_vec(f, value.flatten(order='F'), name, 'c_float')

    f.write('\n// Struct containing CPG objective value and solution\n')
    CPG_Result_fields = ['objective_value'] + list(var_init.keys())
    CPG_Result_values = ['&objective_value']
    for (name, symm, offset) in zip(var_init.keys(), var_symmetric, var_offsets):
        if symm or solver_name == 'ECOS':
            CPG_Result_values.append('&' + name)
        else:
            CPG_Result_values.append('&xsolution + %d' % offset)
    write_struct_def(f, CPG_Result_fields, results_cast, CPG_Result_values, 'CPG_Result', 'CPG_Result_t')

    # Boolean struct for outdated parameter flags
    f.write('\n// Struct containing flags for outdated canonical parameters\n')
    f.write('Canon_Outdated_t Canon_Outdated = {\n')
    for canon_p_id in canon_p_ids:
        f.write('.%s = 0,\n' % canon_p_id)

    f.write('};\n')


def write_workspace_prot(f, solver_name, explicit, user_p_names, user_p_writable, user_p_flat, var_init, canon_p_ids,
                         canon_p, canon_maps, var_symmetric, canon_constants):
    """"
    Write workspace initialization to file
    """

    if solver_name == 'OSQP':
        f.write('\n#include "types.h"\n\n')
    elif solver_name == 'ECOS':
        f.write('\n#include "ecos.h"\n\n')

    # definition safeguard
    f.write('#ifndef CPG_TYPES_H\n')
    f.write('# define CPG_TYPES_H\n\n')

    if solver_name == 'ECOS':
        f.write('typedef double c_float;\n')
        f.write('typedef int c_int;\n\n')

    # struct definitions
    if solver_name == 'ECOS':
        f.write('typedef struct {\n')
        f.write('  c_int    nzmax;\n')
        f.write('  c_int    n;\n')
        f.write('  c_int    m;\n')
        f.write('  c_int    *p;\n')
        f.write('  c_int    *i;\n')
        f.write('  c_float  *x;\n')
        f.write('  c_int    nz;\n')
        f.write('} csc;\n\n')

    f.write('typedef struct {\n')
    for canon_p_id in canon_p_ids:
        f.write('    int         %s;       ///< bool, if canonical parameter %s outdated\n' % (canon_p_id, canon_p_id))
    f.write('} Canon_Outdated_t;\n\n')

    f.write('typedef struct {\n')
    for canon_p_id in canon_p_ids:
        if canon_p_id.isupper():
            f.write('    csc         *%s;      ///< bool, if canonical parameter %s outdated\n' %
                    (canon_p_id, canon_p_id))
        else:
            f.write('    c_float     *%s;      ///< bool, if canonical parameter %s outdated\n' %
                    (canon_p_id, canon_p_id))
    f.write('} Canon_Params_t;\n\n')

    if explicit:
        f.write('typedef struct {\n')

        # single user parameters
        for name in user_p_names:
            f.write('    c_float     *%s;              ///< Your parameter %s\n' % (name, name))

        f.write('} CPG_Params_t;\n\n')

    f.write('typedef struct {\n')
    f.write('    c_float     *objective_value;     ///< Objective function value\n')

    for name in var_init.keys():
        f.write('    c_float     *%s;              ///< Your variable %s\n' % (name, name))

    f.write('} CPG_Result_t;\n\n')

    f.write('#endif // ifndef CPG_TYPES_H\n')

    if solver_name == 'ECOS':
        f.write('\n// ECOS array of SOC dimensions\n')
        osqp_utils.write_vec_extern(f, canon_constants['q'], 'ecos_q', 'c_int')
        f.write('\n// ECOS workspace\n')
        f.write('extern pwork* ecos_workspace;\n')

    f.write('\n// Struct containing flags for outdated canonical parameters\n')
    f.write('extern Canon_Outdated_t Canon_Outdated;\n')

    f.write('\n// Canonical parameters\n')
    for canon_p_id in canon_p_ids:
        write_param_prot(f, canon_p[canon_p_id], canon_p_id)

    f.write('\n// Struct containing canonical parameters\n')
    write_struct_prot(f, 'Canon_Params', 'Canon_Params_t')

    if explicit:
        f.write('\n// User-defined parameters\n')
        for name, value in user_p_writable.items():
            if np.isscalar(value):
                f.write("extern c_float %s;\n" % name)
            else:
                osqp_utils.write_vec_extern(f, value, name, 'c_float')

        f.write('\n// Struct containing all user-defined parameters\n')
        write_struct_prot(f, 'CPG_Params', 'CPG_Params_t')
    else:
        f.write('\n// Sparse mappings from user-defined to canonical parameters\n')
        for name, mapping in zip(canon_p_ids, canon_maps):
            if mapping.nnz > 0:
                osqp_utils.write_mat_extern(f, csc_to_dict(mapping), 'Canon_%s_map' % name)

        f.write('\n// Vector containing flattened user-defined parameters\n')
        osqp_utils.write_vec_extern(f, user_p_flat, 'CPG_Params_Vec', 'c_float')

    f.write('\n// Value of the objective function\n')
    f.write('extern c_float objective_value;\n')

    if any(var_symmetric) or solver_name == 'ECOS':
        f.write('\n// User-defined variables\n')
        for (name, value), symm in zip(var_init.items(), var_symmetric):
            if symm or solver_name == 'ECOS':
                if np.isscalar(value):
                    f.write('extern c_float %s;\n' % name)
                else:
                    osqp_utils.write_vec_extern(f, value.flatten(order='F'), name, 'c_float')

    f.write('\n// Struct containing CPG objective value and solution\n')
    write_struct_prot(f, 'CPG_Result', 'CPG_Result_t')


def write_solve_def(f, solver_name, explicit, canon_p_ids, mappings, user_p_col_to_name, user_p_sizes,
                    var_id_to_indices, is_maximization, user_p_to_canon_outdated, canon_settings_names_to_types,
                    var_symm, canon_p_to_changes, n_var, canon_constants):
    """
    Write parameter initialization function to file
    """

    f.write('\n#include "cpg_solve.h"\n')
    f.write('#include "cpg_workspace.h"\n')
    if solver_name == 'OSQP':
        f.write('#include "workspace.h"\n')
        f.write('#include "osqp.h"\n')

    if not explicit:
        f.write('static c_int i;\n')
        f.write('static c_int j;\n\n')

    if explicit and solver_name == 'ECOS':
        f.write('static c_int i;\n')

    base_cols = list(user_p_col_to_name.keys())

    f.write('// update user-defined parameters\n')
    if explicit:
        for (user_p_name, Canon_outdated_names), user_p_size in zip(user_p_to_canon_outdated.items(), user_p_sizes):
            if user_p_size == 1:
                f.write('void update_%s(c_float val){\n' % user_p_name)
                f.write('*CPG_Params.%s = val;\n' % user_p_name)
            else:
                f.write('void update_%s(c_int idx, c_float val){\n' % user_p_name)
                f.write('CPG_Params.%s[idx] = val;\n' % user_p_name)
            for Canon_outdated_name in Canon_outdated_names:
                f.write('Canon_Outdated.%s = 1;\n' % Canon_outdated_name)
            f.write('}\n')
    else:
        for base_col, (user_p_name, Canon_outdated_names), user_p_size in zip(base_cols,
                                                                              user_p_to_canon_outdated.items(),
                                                                              user_p_sizes):
            if user_p_size == 1:
                f.write('void update_%s(c_float val){\n' % user_p_name)
                f.write('CPG_Params_Vec[%d] = val;\n' % base_col)
            else:
                f.write('void update_%s(c_int idx, c_float val){\n' % user_p_name)
                f.write('CPG_Params_Vec[idx+%d] = val;\n' % base_col)
            for Canon_outdated_name in Canon_outdated_names:
                f.write('Canon_Outdated.%s = 1;\n' % Canon_outdated_name)
            f.write('}\n')

    f.write('\n// map user-defined to canonical parameters\n')

    for canon_name, mapping in zip(canon_p_ids, mappings):
        if mapping.nnz > 0:
            f.write('void canonicalize_Canon_%s(){\n' % canon_name)
            if canon_name.isupper():
                s = '->x'
            else:
                s = ''
            if explicit:
                write_canonicalize_explicit(f, canon_name, s, mapping, base_cols, user_p_sizes, user_p_col_to_name)
            else:
                write_canonicalize(f, canon_name, s, mapping)
            f.write('}\n')

    f.write('\n// retrieve user-defined objective function value\n')
    f.write('void retrieve_value(){\n')

    if solver_name == 'OSQP':
        if is_maximization:
            f.write('objective_value = -(workspace.info->obj_val + *Canon_Params.d);\n')
        else:
            f.write('objective_value = workspace.info->obj_val + *Canon_Params.d;\n')
        sol_str = 'workspace.solution->x'
    elif solver_name == 'ECOS':
        if is_maximization:
            f.write('objective_value = -*Canon_Params.d;\n')
            f.write('for (i = 0; i < %d; i++) {\n' % n_var)
            f.write('objective_value -= ecos_workspace->c[i]*ecos_workspace->x[i];\n')
            f.write('}\n')
        else:
            f.write('objective_value = *Canon_Params.d;\n')
            f.write('for (i = 0; i < %d; i++) {\n' % n_var)
            f.write('objective_value += ecos_workspace->c[i]*ecos_workspace->x[i];\n')
            f.write('}\n')
        sol_str = 'ecos_workspace->x'
    else:
        raise ValueError("Only OSQP and ECOS are supported!")

    f.write('}\n\n')

    f.write('// retrieve solution in terms of user-defined variables\n')
    f.write('void retrieve_solution(){\n')

    for symm, (var_id, indices) in zip(var_symm, var_id_to_indices.items()):
        if symm or solver_name == 'ECOS':
            if len(indices) == 1:
                f.write('%s = %s[%d];\n' % (var_id, sol_str, indices[0]))
            else:
                for i, idx in enumerate(indices):
                    f.write('%s[%d] = %s[%d];\n' % (var_id, i, sol_str, idx))

    f.write('}\n\n')

    f.write('// perform one ASA sequence to solve a problem instance\n')
    f.write('void solve(){\n')

    if solver_name == 'OSQP':

        if canon_p_to_changes['P'] and canon_p_to_changes['A']:
            f.write('if (Canon_Outdated.P && Canon_Outdated.A) {\n')
            f.write('canonicalize_Canon_P();\n')
            f.write('canonicalize_Canon_A();\n')
            f.write('osqp_update_P_A(&workspace, Canon_Params.P->x, 0, 0, Canon_Params.A->x, 0, 0);\n')
            f.write('} else if (Canon_Outdated.P) {\n')
            f.write('canonicalize_Canon_P();\n')
            f.write('osqp_update_P(&workspace, Canon_Params.P->x, 0, 0);\n')
            f.write('} else if (Canon_Outdated.A) {\n')
            f.write('canonicalize_Canon_A();\n')
            f.write('osqp_update_A(&workspace, Canon_Params.A->x, 0, 0);\n')
            f.write('}\n')
        else:
            if canon_p_to_changes['P']:
                f.write('if (Canon_Outdated.P) {\n')
                f.write('canonicalize_Canon_P();\n')
                f.write('osqp_update_P(&workspace, Canon_Params.P->x, 0, 0);\n')
                f.write('}\n')
            if canon_p_to_changes['A']:
                f.write('if (Canon_Outdated.A) {\n')
                f.write('canonicalize_Canon_A();\n')
                f.write('osqp_update_A(&workspace, Canon_Params.A->x, 0, 0);\n')
                f.write('}\n')

        if canon_p_to_changes['q']:
            f.write('if (Canon_Outdated.q) {\n')
            f.write('canonicalize_Canon_q();\n')
            f.write('osqp_update_lin_cost(&workspace, Canon_Params.q);\n')
            f.write('}\n')

        if canon_p_to_changes['d']:
            f.write('if (Canon_Outdated.d) {\n')
            f.write('canonicalize_Canon_d();\n')
            f.write('}\n')

        if canon_p_to_changes['l'] and canon_p_to_changes['u']:
            f.write('if (Canon_Outdated.l && Canon_Outdated.u) {\n')
            f.write('canonicalize_Canon_l();\n')
            f.write('canonicalize_Canon_u();\n')
            f.write('osqp_update_bounds(&workspace, Canon_Params.l, Canon_Params.u);\n')
            f.write('} else if (Canon_Outdated.l) {\n')
            f.write('canonicalize_Canon_l();\n')
            f.write('osqp_update_lower_bound(&workspace, Canon_Params.l);\n')
            f.write('} else if (Canon_Outdated.u) {\n')
            f.write('canonicalize_Canon_u();\n')
            f.write('osqp_update_upper_bound(&workspace, Canon_Params.u);\n')
            f.write('}\n')
        else:
            if canon_p_to_changes['l']:
                f.write('if (Canon_Outdated.l) {\n')
                f.write('canonicalize_Canon_l();\n')
                f.write('osqp_update_lower_bound(&workspace, Canon_Params.l);\n')
                f.write('}\n')
            if canon_p_to_changes['u']:
                f.write('if (Canon_Outdated.u) {\n')
                f.write('canonicalize_Canon_u();\n')
                f.write('osqp_update_upper_bound(&workspace, Canon_Params.u);\n')
                f.write('}\n')

    elif solver_name == 'ECOS':

        for canon_p, changes in canon_p_to_changes.items():
            if changes:
                f.write('if (Canon_Outdated.%s) {\n' % canon_p)
                f.write('canonicalize_Canon_%s();\n' % canon_p)
                f.write('}\n')

    if solver_name == 'OSQP':
        f.write('osqp_solve(&workspace);\n')
    elif solver_name == 'ECOS':
        write_ecos_setup(f, canon_constants)
        f.write('ECOS_solve(ecos_workspace);\n')

    f.write('retrieve_value();\n')
    f.write('retrieve_solution();\n')

    for canon_p_id in canon_p_ids:
        f.write('Canon_Outdated.%s = 0;\n' % canon_p_id)

    f.write('}\n\n')

    if solver_name == 'OSQP':
        f.write('// update OSQP settings\n')
        f.write('void set_OSQP_default_settings(){\n')
        f.write('osqp_set_default_settings(&settings);\n')
        f.write('}\n')
        for name, typ in canon_settings_names_to_types.items():
            f.write('void set_OSQP_%s(%s %s_new){\n' % (name, typ, name))
            f.write('osqp_update_%s(&workspace, %s_new);\n' % (name, name))
            f.write('}\n')


def write_solve_prot(f, solver_name, canon_p_ids, user_p_name_to_size, canon_settings_names_to_types):
    """
    Write function declarations to file
    """

    if solver_name == 'OSQP':
        f.write('\n#include "types.h"\n\n')
    elif solver_name == 'ECOS':
        f.write('\n#include "cpg_workspace.h"\n\n')

    f.write('// map user-defined to canonical parameters\n')
    for canon_p_id in canon_p_ids:
        f.write('extern void canonicalize_Canon_%s();\n' % canon_p_id)

    f.write('\n// retrieve user-defined objective function value\n')
    f.write('extern void retrieve_value();\n')

    f.write('\n// retrieve solution in terms of user-defined variables\n')
    f.write('extern void retrieve_solution();\n')

    f.write('\n// perform one ASA sequence to solve a problem instance\n')
    f.write('extern void solve();\n')

    f.write('\n// update user-defined parameter values\n')

    for name, size in user_p_name_to_size.items():
        if size == 1:
            f.write('extern void update_%s(c_float val);\n' % name)
        else:
            f.write('extern void update_%s(c_int idx, c_float val);\n' % name)

    if solver_name == 'OSQP':
        f.write('\n// update OSQP settings\n')
        f.write('extern void set_OSQP_default_settings();\n')
        for name, typ in canon_settings_names_to_types.items():
            f.write('extern void set_OSQP_%s(%s %s_new);\n' % (name, typ, name))


def write_example_def(f, user_p_writable, var_name_to_size):
    """
    Write main function to file
    """

    f.write('int main(int argc, char *argv[]){\n\n')

    f.write('// initialize user-defined parameter values\n')
    for name, value in user_p_writable.items():
        if np.isscalar(value):
            f.write('update_%s(%.20f);\n' % (name, value))
        else:
            for i in range(len(value)):
                f.write('update_%s(%d, %.20f);\n' % (name, i, value[i]))

    f.write('\n// solve the problem instance\n')
    f.write('solve();\n\n')

    f.write('// printing objective function value for demonstration purpose\n')
    f.write('printf("f = %f \\n", objective_value);\n\n')

    f.write('// printing solution for demonstration purpose\n')

    for name, size in var_name_to_size.items():
        if size == 1:
            f.write('printf("%s = %%f \\n", *CPG_Result.%s);\n' % (name, name))
        else:
            f.write('for(i = 0; i < %d; i++) {\n' % size)
            f.write('printf("%s[%%lld] = %%f \\n", i, CPG_Result.%s[i]);\n' % (name, name))
            f.write('}\n')

    f.write('return 0;\n')
    f.write('}\n')


def write_canon_CMakeLists(f, solver_name):
    """
    Pass sources to parent scope in {OSQP/ECOS}_code/CMakeLists.txt
    """

    if solver_name == 'OSQP':
        f.write('\nset(solver_head "${osqp_headers}" PARENT_SCOPE)')
        f.write('\nset(solver_src "${osqp_src}" PARENT_SCOPE)')
    elif solver_name == 'ECOS':
        f.write('\nset(solver_head "${ecos_headers}" PARENT_SCOPE)')
        f.write('\nset(solver_src "${ecos_sources}" PARENT_SCOPE)')


def write_module(f, solver_name, user_p_name_to_size, var_name_to_size, canon_settings_names, problem_name):
    """
    Write c++ file for pbind11 wrapper
    """

    f.write('extern "C" {\n')
    f.write('    #include "include/cpg_workspace.h"\n')
    f.write('    #include "include/cpg_solve.h"\n')
    if solver_name == 'OSQP':
        f.write('    #include "solver_code/include/workspace.h"\n')
    f.write('}\n\n')
    f.write('namespace py = pybind11;\n\n')
    f.write('static int i;\n\n')

    # cpp function that maps parameters to results
    f.write('CPG_Result_%s_cpp_t solve_cpp(struct CPG_Updated_%s_cpp_t& CPG_Updated_cpp, '
            'struct CPG_Params_%s_cpp_t& CPG_Params_cpp){\n\n'
            % (problem_name, problem_name, problem_name))

    f.write('    // pass changed user-defined parameter values to the solver\n')
    for name, size in user_p_name_to_size.items():
        f.write('    if (CPG_Updated_cpp.%s) {\n' % name)
        if size == 1:
            f.write('        update_%s(CPG_Params_cpp.%s);\n' % (name, name))
        else:
            f.write('        for(i = 0; i < %d; i++) {\n' % size)
            f.write('            update_%s(i, CPG_Params_cpp.%s[i]);\n' % (name, name))
            f.write('        }\n')
        f.write('    }\n')

    # perform ASA procedure
    f.write('\n    // ASA\n')
    f.write('    std::clock_t ASA_start = std::clock();\n')
    f.write('    solve();\n')
    f.write('    std::clock_t ASA_end = std::clock();\n\n')

    # arrange and return results
    f.write('    // arrange and return results\n')
    f.write('    CPG_Info_%s_cpp_t CPG_Info_cpp {};\n' % problem_name)
    if solver_name == 'OSQP':
        f.write('    CPG_Info_cpp.obj_val = objective_value;\n')
        f.write('    CPG_Info_cpp.iter = workspace.info->iter;\n')
        f.write('    CPG_Info_cpp.status = workspace.info->status;\n')
        f.write('    CPG_Info_cpp.pri_res = workspace.info->pri_res;\n')
        f.write('    CPG_Info_cpp.dua_res = workspace.info->dua_res;\n')
    f.write('    CPG_Info_cpp.ASA_proc_time = 1000.0 * (ASA_end-ASA_start) / CLOCKS_PER_SEC;\n')

    f.write('    CPG_Result_%s_cpp_t CPG_Result_cpp {};\n' % problem_name)
    f.write('    CPG_Result_cpp.CPG_Info = CPG_Info_cpp;\n')
    for name, size in var_name_to_size.items():
        if size == 1:
            f.write('    CPG_Result_cpp.%s = *CPG_Result.%s;\n' % (name, name))
        else:
            f.write('    for(i = 0; i < %d; i++) {\n' % size)
            f.write('        CPG_Result_cpp.%s[i] = CPG_Result.%s[i];\n' % (name, name))
            f.write('    }\n')

    # return
    f.write('    return CPG_Result_cpp;\n\n')
    f.write('}\n\n')

    # module
    f.write('PYBIND11_MODULE(cpg_module, m) {\n\n')

    f.write('    py::class_<CPG_Params_%s_cpp_t>(m, "cpg_params")\n' % problem_name)
    f.write('            .def(py::init<>())\n')
    for name in user_p_name_to_size.keys():
        f.write('            .def_readwrite("%s", &CPG_Params_%s_cpp_t::%s)\n' % (name, problem_name, name))
    f.write('            ;\n\n')

    f.write('    py::class_<CPG_Updated_%s_cpp_t>(m, "cpg_updated")\n' % problem_name)
    f.write('            .def(py::init<>())\n')
    for name in user_p_name_to_size.keys():
        f.write('            .def_readwrite("%s", &CPG_Updated_%s_cpp_t::%s)\n' % (name, problem_name, name))
    f.write('            ;\n\n')

    f.write('    py::class_<CPG_Info_%s_cpp_t>(m, "cpg_info")\n' % problem_name)
    f.write('            .def(py::init<>())\n')
    if solver_name == 'OSQP':
        f.write('            .def_readwrite("obj_val", &CPG_Info_%s_cpp_t::obj_val)\n' % problem_name)
        f.write('            .def_readwrite("iter", &CPG_Info_%s_cpp_t::iter)\n' % problem_name)
        f.write('            .def_readwrite("status", &CPG_Info_%s_cpp_t::status)\n' % problem_name)
        f.write('            .def_readwrite("pri_res", &CPG_Info_%s_cpp_t::pri_res)\n' % problem_name)
        f.write('            .def_readwrite("dua_res", &CPG_Info_%s_cpp_t::dua_res)\n' % problem_name)
    f.write('            .def_readwrite("ASA_proc_time", &CPG_Info_%s_cpp_t::ASA_proc_time)\n' % problem_name)
    f.write('            ;\n\n')

    f.write('    py::class_<CPG_Result_%s_cpp_t>(m, "cpg_result")\n' % problem_name)
    f.write('            .def(py::init<>())\n')
    f.write('            .def_readwrite("cpg_info", &CPG_Result_%s_cpp_t::CPG_Info)\n' % problem_name)
    for name in var_name_to_size.keys():
        f.write('            .def_readwrite("%s", &CPG_Result_%s_cpp_t::%s)\n' % (name, problem_name, name))
    f.write('            ;\n\n')

    f.write('    m.def("solve", &solve_cpp);\n\n')

    if solver_name == 'OSQP':
        f.write('    m.def("set_OSQP_default_settings", &set_OSQP_default_settings);\n')
        for name in canon_settings_names:
            f.write('    m.def("set_OSQP_%s", &set_OSQP_%s);\n' % (name, name))

    f.write('\n}')


def write_module_prot(f, solver_name, user_p_name_to_size, var_name_to_size, problem_name):
    """
    Write c++ file for pbind11 wrapper
    """

    # cpp struct containing info on results
    f.write('struct CPG_Info_%s_cpp_t {\n' % problem_name)
    if solver_name == 'OSQP':
        f.write('    double obj_val;\n')
        f.write('    int iter;\n')
        f.write('    char* status;\n')
        f.write('    double pri_res;\n')
        f.write('    double dua_res;\n')
    f.write('    double ASA_proc_time;\n')
    f.write('};\n\n')

    # cpp struct containing user-defined parameters
    f.write('struct CPG_Params_%s_cpp_t {\n' % problem_name)
    for name, size in user_p_name_to_size.items():
        if size == 1:
            f.write('    double %s;\n' % name)
        else:
            f.write('    std::array<double, %d> %s;\n' % (size, name))
    f.write('};\n\n')

    # cpp struct containing update flags for user-defined parameters
    f.write('struct CPG_Updated_%s_cpp_t {\n' % problem_name)
    for name in user_p_name_to_size.keys():
        f.write('    bool %s;\n' % name)
    f.write('};\n\n')

    # cpp struct containing objective value and user-defined variables
    f.write('struct CPG_Result_%s_cpp_t {\n' % problem_name)
    f.write('    CPG_Info_%s_cpp_t CPG_Info;\n' % problem_name)
    for name, size in var_name_to_size.items():
        if size == 1:
            f.write('    double %s;\n' % name)
        else:
            f.write('    std::array<double, %d> %s;\n' % (size, name))
    f.write('};\n\n')

    # cpp function that maps parameters to results
    f.write('CPG_Result_%s_cpp_t solve_cpp(struct CPG_Updated_%s_cpp_t& CPG_Updated_cpp, '
            'struct CPG_Params_%s_cpp_t& CPG_Params_cpp);\n'
            % (problem_name, problem_name, problem_name))


def write_method(f, solver_name, code_dir, user_p_name_to_size, var_name_to_shape):
    """
    Write function to be registered as custom CVXPY solve method
    """

    f.write('from %s import cpg_module\n\n\n' % code_dir.replace('/', '.').replace('\\', '.'))
    f.write('def cpg_solve(prob, updated_params=None, **kwargs):\n\n')
    f.write('    if updated_params is None:\n')
    p_list_string = ''
    for name in user_p_name_to_size.keys():
        p_list_string += '"%s", ' % name
    f.write('        updated_params = [%s]\n' % p_list_string[:-2])
    f.write('\n    upd = cpg_module.cpg_updated()\n')
    f.write('    for p in updated_params:\n')
    f.write('        try:\n')
    f.write('            setattr(upd, p, True)\n')
    f.write('        except AttributeError:\n')
    f.write('            raise(AttributeError("%s is not a parameter." % p))\n\n')

    if solver_name == 'OSQP':
        f.write('    cpg_module.set_OSQP_default_settings()\n')
        f.write('    for key, value in kwargs.items():\n')
        f.write('        try:\n')
        f.write('            eval(\'cpg_module.set_OSQP_%s(value)\' % key)\n')
        f.write('        except AttributeError:\n')
        f.write('            raise(AttributeError(\'Solver setting "%s" not available.\' % key))\n\n')

    f.write('    par = cpg_module.cpg_params()\n')

    for name, size in user_p_name_to_size.items():
        if size == 1:
            f.write('    par.%s = prob.param_dict[\'%s\'].value\n' % (name, name))
        else:
            f.write('    par.%s = list(prob.param_dict[\'%s\'].value.flatten(order=\'F\'))\n' % (name, name))

    f.write('\n    t0 = time.time()\n')
    f.write('    res = cpg_module.solve(upd, par)\n')
    f.write('    t1 = time.time()\n\n')

    f.write('    prob._clear_solution()\n\n')

    for name, shape in var_name_to_shape.items():
        if len(shape) == 2:
            f.write('    prob.var_dict[\'%s\'].value = np.array(res.%s).reshape((%d, %d), order=\'F\')\n' %
                    (name, name, shape[0], shape[1]))
        else:
            f.write('    prob.var_dict[\'%s\'].value = np.array(res.%s)\n' % (name, name))

    if solver_name == 'OSQP':
        f.write('\n    prob._status = res.cpg_info.status\n')
        f.write('    if abs(res.cpg_info.obj_val) == 1e30:\n')
        f.write('        prob._value = np.sign(res.cpg_info.obj_val)*np.inf\n')
        f.write('    else:\n')
        f.write('        prob._value = res.cpg_info.obj_val\n')
        f.write('    primal_vars = {var.id: var.value for var in prob.variables()}\n')
        f.write('    dual_vars = {}\n')
        f.write('    solver_specific_stats = {\'obj_val\': res.cpg_info.obj_val,\n')
        f.write('                             \'status\': res.cpg_info.status,\n')
        f.write('                             \'iter\': res.cpg_info.iter,\n')
        f.write('                             \'pri_res\': res.cpg_info.pri_res,\n')
        f.write('                             \'dua_res\': res.cpg_info.dua_res,\n')
        f.write('                             \'ASA_proc_time\': res.cpg_info.ASA_proc_time}\n')
        f.write('    attr = {\'solve_time\': t1-t0, \'solver_specific_stats\': solver_specific_stats, '
                '\'num_iters\': res.cpg_info.iter}\n')
        f.write('    prob._solution = Solution(prob.status, prob.value, primal_vars, dual_vars, attr)\n')
        f.write('    results_dict = {\'solver_specific_stats\': solver_specific_stats,\n')
        f.write('                    \'num_iters\': res.cpg_info.iter,\n')
        f.write('                    \'solve_time\': t1-t0}\n')
        f.write('    prob._solver_stats = SolverStats(results_dict, \'OSQP\')\n\n')

    f.write('    return prob.value\n')


def replace_html_data(code_dir, explicit, text, user_p_name_to_size, user_p_writable, var_name_to_size,
                      user_p_total_size):
    """
    Replace placeholder strings in html documentation file
    """

    # code_dir
    text = text.replace('$CODEDIR', code_dir)
    text = text.replace('$CDPYTHON', code_dir.replace('/', '.').replace('\\', '.'))

    # type definition of CPG_Params_t or CPG_Params_Vec
    if explicit:
        CPGPARAMSTYPEDEF = '\n// Struct type with user-defined parameters as fields\n'
        CPGPARAMSTYPEDEF += 'typedef struct {\n'
        for name in user_p_name_to_size.keys():
            CPGPARAMSTYPEDEF += ('    c_float     *%s;' % name).ljust(33) + ('///< Your parameter %s\n' % name)
        CPGPARAMSTYPEDEF += '} CPG_Params_t;\n'
    else:
        CPGPARAMSTYPEDEF = ''
    text = text.replace('$CPGPARAMSTYPEDEF', CPGPARAMSTYPEDEF)

    # type definition of CPG_Result_t
    CPGRESULTTYPEDEF = 'typedef struct {\n'
    CPGRESULTTYPEDEF += '    c_float     *objective_value;///< Objective function value\n'
    for name in var_name_to_size.keys():
        CPGRESULTTYPEDEF += ('    c_float     *%s;' % name).ljust(33) + ('///< Your variable %s\n' % name)
    CPGRESULTTYPEDEF += '} CPG_Result_t;'

    text = text.replace('$CPGRESULTTYPEDEF', CPGRESULTTYPEDEF)

    # parameter delarations
    if explicit:
        CPGPARAMDECLARATIONS = ''
        for name, value in user_p_writable.items():
            if np.isscalar(value):
                CPGPARAMDECLARATIONS += 'c_float %s;\n' % name
            else:
                CPGPARAMDECLARATIONS += 'c_float %s[%d];\n' % (name, value.size)
        CPGPARAMDECLARATIONS += '// Struct containing all user-defined parameters\n'
        CPGPARAMDECLARATIONS += 'CPG_Params_t CPG_Params;\n'
    else:
        CPGPARAMDECLARATIONS = 'c_float CPG_Params_Vec[%d];' % (user_p_total_size+1)
    text = text.replace('$CPGPARAMDECLARATIONS', CPGPARAMDECLARATIONS)

    # variable declarations
    CPGVARIABLEDECLARATIONS = ''
    for name, size in var_name_to_size.items():
        if size == 1:
            CPGVARIABLEDECLARATIONS += 'c_float %s;\n' % name
        else:
            CPGVARIABLEDECLARATIONS += 'c_float %s[%d];\n' % (name, size)

    text = text.replace('$CPGVARIABLEDECLARATIONS', CPGVARIABLEDECLARATIONS[:-1])

    # update declarations
    CPGUPDATEDECLARATIONS = ''
    for name, size in user_p_name_to_size.items():
        if size == 1:
            CPGUPDATEDECLARATIONS += 'void update_%s(c_float value);\n' % name
        else:
            CPGUPDATEDECLARATIONS += 'void update_%s(c_int idx, c_float value);\n' % name

    return text.replace('$CPGUPDATEDECLARATIONS', CPGUPDATEDECLARATIONS[:-1])
