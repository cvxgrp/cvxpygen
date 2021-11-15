
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


def write_canonicalize_explicit(f, canon_name, s, mapping, base_cols, user_p_sizes_usp, user_p_col_to_name_usp,
                                user_p_name_to_size_usp, prob_name):
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
                if user_p_col + user_p_sizes_usp[i] > col:
                    expr_is_const = False
                    user_name = user_p_col_to_name_usp[user_p_col]
                    if user_p_name_to_size_usp[user_name] == 1:
                        if abs(datum) == 1:
                            ex = '(%s%sCPG_Params.%s)+' % (sign_to_str[datum], prob_name, user_name)
                        else:
                            ex = '(%.20f*%sCPG_Params.%s)+' % (datum, prob_name, user_name)
                    else:
                        if abs(datum) == 1:
                            ex = '(%s%sCPG_Params.%s[%d])+' % (sign_to_str[datum], prob_name, user_name, col-user_p_col)
                        else:
                            ex = '(%.20f*%sCPG_Params.%s[%d])+' % (datum, prob_name, user_name, col-user_p_col)
                    break
            expr += ex
        expr = expr[:-1]
        if data.size > 0 and expr_is_const is False:
            if canon_name == 'd':
                f.write('  %sCanon_Params.d = %s;\n' % (prob_name, expr))
            else:
                f.write('  %sCanon_Params.%s%s[%d] = %s;\n' % (prob_name, canon_name, s, row, expr))


def write_canonicalize(f, canon_name, s, mapping, prob_name):
    """
    Write function to compute canonical parameter value
    """

    f.write('  for(i=0; i<%d; i++){\n' % mapping.shape[0])
    f.write('    %sCanon_Params.%s%s[i] = 0;\n' % (prob_name, canon_name, s))
    f.write('    for(j=%scanon_%s_map.p[i]; j<%scanon_%s_map.p[i+1]; j++){\n' %
            (prob_name, canon_name, prob_name, canon_name))
    f.write('      %sCanon_Params.%s%s[i] += %scanon_%s_map.x[j]*%scpg_params_vec[%scanon_%s_map.i[j]];\n' %
            (prob_name, canon_name, s, prob_name, canon_name, prob_name, prob_name, canon_name))
    f.write('    }\n')
    f.write('  }\n')


def write_param_def(f, param, name, suffix, prob_name):
    """
    Use osqp.codegen.utils for writing vectors and matrices
    """
    if not param_is_empty(param):
        if name.isupper():
            osqp_utils.write_mat(f, param, '%scanon_%s%s' % (prob_name, name, suffix))
        elif name == 'd':
            f.write('c_float %scanon_d%s = %.20f;\n' % (prob_name, suffix, param[0]))
        else:
            osqp_utils.write_vec(f, param, '%scanon_%s%s' % (prob_name, name, suffix), 'c_float')
        f.write('\n')


def write_param_prot(f, param, name, suffix, prob_name):
    """
    Use osqp.codegen.utils for writing vectors and matrices
    """
    if not param_is_empty(param):
        if name.isupper():
            osqp_utils.write_mat_extern(f, param, '%scanon_%s%s' % (prob_name, name, suffix))
        elif name == 'd':
            f.write('extern c_float %scanon_d%s;\n' % (prob_name, suffix))
        else:
            osqp_utils.write_vec_extern(f, param, '%scanon_%s%s' % (prob_name, name, suffix), 'c_float')


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

    f.write("extern c_float cpg_%s[%d];\n" % (name, mat.size))


def write_struct_def(f, fields, casts, values, name, typ):
    """
    Write structure to file
    """

    f.write('%s %s = {\n' % (typ, name))

    # write structure fields
    for field, cast, value in zip(fields, casts, values):
        if value in ['0', 'SCS_NULL']:
            cast = ''
        f.write('.%s = %s%s,\n' % (field, cast, value))

    f.write('};\n')


def write_struct_prot(f, name, typ):
    """
    Write structure to file
    """

    f.write("extern %s %s;\n" % (typ, name))


def write_ecos_setup(f, canon_constants, prob_name):
    """
    Write ECOS setup function to file
    """
    n = canon_constants['n']
    m = canon_constants['m']
    p = canon_constants['p']
    ell = canon_constants['l']
    n_cones = canon_constants['n_cones']
    e = canon_constants['e']

    if p == 0:
        Ax_str = Ap_str = Ai_str = b_str = '0'
    else:
        Ax_str = '%sCanon_Params_ECOS.A->x' % prob_name
        Ap_str = '%sCanon_Params_ECOS.A->p' % prob_name
        Ai_str = '%sCanon_Params_ECOS.A->i' % prob_name
        b_str = '%sCanon_Params_ECOS.b' % prob_name

    if n_cones == 0:
        ecos_q_str = '0'
    else:
        ecos_q_str = '(int *) &%secos_q' % prob_name

    f.write('  %secos_workspace = ECOS_setup(%d, %d, %d, %d, %d, %s, %d, '
            '%sCanon_Params_ECOS.G->x, %sCanon_Params_ECOS.G->p, %sCanon_Params_ECOS.G->i, '
            '%s, %s, %s, %sCanon_Params_ECOS.c, %sCanon_Params_ECOS.h, %s);\n' %
            (prob_name, n, m, p, ell, n_cones, ecos_q_str, e, prob_name, prob_name, prob_name, Ax_str, Ap_str, Ai_str,
             prob_name, prob_name, b_str))


def write_workspace_def(f, solver_name, explicit, user_p_names, user_p_writable, user_p_flat_usp, var_init, canon_p_ids,
                        canon_p, canon_mappings, var_symmetric, var_offsets, canon_constants,
                        canon_settings_names_to_default, prob_name):

    f.write('\n#include "cpg_workspace.h"\n')
    if solver_name == 'OSQP':
        f.write('#include "workspace.h"\n')

    if explicit:
        f.write('\n// User-defined parameters\n')
        user_casts = []
        user_values = []
        for name, value in user_p_writable.items():
            if np.isscalar(value):
                user_casts.append('')
                user_values.append('%.20f' % value)
            else:
                osqp_utils.write_vec(f, value, prob_name+'cpg_'+name, 'c_float')
                f.write('\n')
                user_casts.append('(c_float *) ')
                user_values.append('&'+prob_name+'cpg_'+name)
        f.write('// Struct containing all user-defined parameters\n')
        write_struct_def(f, user_p_names, user_casts, user_values, '%sCPG_Params' % prob_name, 'CPG_Params_t')
        f.write('\n')
    else:
        f.write('\n// Vector containing flattened user-defined parameters\n')
        osqp_utils.write_vec(f, user_p_flat_usp, '%scpg_params_vec' % prob_name, 'c_float')
        f.write('\n// Sparse mappings from user-defined to canonical parameters\n')
        for name, mapping in zip(canon_p_ids, canon_mappings):
            if mapping.nnz > 0:
                osqp_utils.write_mat(f, csc_to_dict(mapping), '%scanon_%s_map' % (prob_name, name))
                f.write('\n')

    canon_casts = []
    f.write('// Canonical parameters\n')
    for canon_p_id in canon_p_ids:
        if canon_p_id == 'd':
            canon_casts.append('')
        else:
            write_param_def(f, replace_inf(canon_p[canon_p_id]), canon_p_id, '', prob_name)
            if solver_name == 'ECOS':
                write_param_def(f, replace_inf(canon_p[canon_p_id]), canon_p_id, '_ECOS', prob_name)
            if canon_p_id.isupper():
                canon_casts.append('')
            else:
                canon_casts.append('(c_float *) ')

    f.write('// Struct containing canonical parameters\n')

    struct_values = []
    struct_values_ECOS = []
    for canon_p_id in canon_p_ids:
        if type(canon_p[canon_p_id]) == dict:
            length = len(canon_p[canon_p_id]['x'])
        else:
            length = len(canon_p[canon_p_id])
        if length == 0:
            struct_values.append('0')
            if solver_name == 'ECOS':
                struct_values_ECOS.append('0')
        elif length == 1:
            struct_values.append('%.20f' % canon_p[canon_p_id])
            if solver_name == 'ECOS':
                struct_values_ECOS.append('%.20f' % canon_p[canon_p_id])
        else:
            struct_values.append('&%scanon_%s' % (prob_name, canon_p_id))
            if solver_name == 'ECOS':
                struct_values_ECOS.append('&%scanon_%s_ECOS' % (prob_name, canon_p_id))

    write_struct_def(f, canon_p_ids, canon_casts, struct_values, '%sCanon_Params' % prob_name, 'Canon_Params_t')
    f.write('\n')
    if solver_name == 'ECOS':
        write_struct_def(f, canon_p_ids, canon_casts, struct_values_ECOS, '%sCanon_Params_ECOS' % prob_name,
                         'Canon_Params_t')
        f.write('\n')

    # Boolean struct for outdated parameter flags
    f.write('// Struct containing flags for outdated canonical parameters\n')
    f.write('Canon_Outdated_t %sCanon_Outdated = {\n' % prob_name)
    for canon_p_id in canon_p_ids:
        f.write('.%s = 0,\n' % canon_p_id)
    f.write('};\n\n')

    results_cast = []
    if any(var_symmetric) or solver_name == 'ECOS':
        f.write('// User-defined variables\n')
    for (name, value), symm in zip(var_init.items(), var_symmetric):
        if symm or solver_name == 'ECOS':
            if np.isscalar(value):
                results_cast.append('')
            else:
                osqp_utils.write_vec(f, value.flatten(order='F'), prob_name+name, 'c_float')
                f.write('\n')
                results_cast.append('(c_float *) ')
        else:
            results_cast.append('(c_float *) ')
    results_cast.append('')

    f.write('// Struct containing solver info\n')
    CPG_Info_fields = ['obj_val', 'iter', 'status', 'pri_res', 'dua_res']
    if solver_name in ['OSQP', 'SCS']:
        CPG_Info_values = ['0', '0', '"unknown"', '0', '0']
    elif solver_name == 'ECOS':
        CPG_Info_values = ['0', '0', '0', '0', '0']
    else:
        raise ValueError("Problem class cannot be addressed by the OSQP or ECOS solver!")
    info_cast = ['', '', '', '', '']
    write_struct_def(f, CPG_Info_fields, info_cast, CPG_Info_values, '%sCPG_Info' % prob_name, 'CPG_Info_t')

    f.write('\n// Struct containing solution and info\n')
    CPG_Result_fields = list(var_init.keys()) + ['info']
    CPG_Result_values = []
    for ((name, var), symm, offset) in zip(var_init.items(), var_symmetric, var_offsets):
        if symm or solver_name == 'ECOS':
            if np.isscalar(var):
                CPG_Result_values.append('0')
            else:
                CPG_Result_values.append('&' + prob_name + name)
        else:
            if np.isscalar(var):
                CPG_Result_values.append('0')
            else:
                if solver_name == 'OSQP':
                    CPG_Result_values.append('&xsolution + %d' % offset)
                elif solver_name == 'SCS':
                    CPG_Result_values.append('&%sscs_x + %d' % (prob_name, offset))
    CPG_Result_values.append('&%sCPG_Info' % prob_name)
    write_struct_def(f, CPG_Result_fields, results_cast, CPG_Result_values, '%sCPG_Result' % prob_name, 'CPG_Result_t')

    if solver_name == 'SCS':

        f.write('\n// SCS matrix A\n')
        scs_A_fiels = ['x', 'i', 'p', 'm', 'n']
        scs_A_casts = ['(c_float *) ', '(c_int *) ', '(c_int *) ', '', '']
        scs_A_values = ['&%scanon_A_x' % prob_name, '&%scanon_A_i' % prob_name,
                        '&%scanon_A_p' % prob_name, str(canon_constants['m']), str(canon_constants['n'])]
        write_struct_def(f, scs_A_fiels, scs_A_casts, scs_A_values, '%sScs_A' % prob_name, 'ScsMatrix')

        f.write('\n// Struct containing SCS data\n')
        scs_d_fiels = ['m', 'n', 'A', 'P', 'b', 'c']
        scs_d_casts = ['', '', '', '', '(c_float *) ', '(c_float *) ']
        scs_d_values = [str(canon_constants['m']), str(canon_constants['n']), '&%sScs_A' % prob_name, 'SCS_NULL',
                        '&%scanon_b' % prob_name, '&%scanon_c' % prob_name]
        write_struct_def(f, scs_d_fiels, scs_d_casts, scs_d_values, '%sScs_D' % prob_name, 'ScsData')

        if canon_constants['qsize'] > 0:
            f.write('\n// SCS array of SOC dimensions\n')
            osqp_utils.write_vec(f, canon_constants['q'], '%sscs_q' % prob_name, 'c_int')
            k_field_q_str = '&%sscs_q' % prob_name
        else:
            k_field_q_str = 'SCS_NULL'

        f.write('\n// Struct containing SCS cone data\n')
        scs_k_fields = ['z', 'l', 'bu', 'bl', 'bsize', 'q', 'qsize', 's', 'ssize', 'ep', 'ed', 'p', 'psize']
        scs_k_casts = ['', '', '(c_float *) ', '(c_float *) ', '', '(c_int *) ', '', '(c_int *) ', '', '', '',
                       '(c_float *) ', '']
        scs_k_values = [str(canon_constants['z']), str(canon_constants['l']), 'SCS_NULL', 'SCS_NULL', '0',
                        k_field_q_str, str(canon_constants['qsize']), 'SCS_NULL', '0', '0', '0', 'SCS_NULL', '0']
        write_struct_def(f, scs_k_fields, scs_k_casts, scs_k_values, '%sScs_K' % prob_name, 'ScsCone')

        f.write('\n// Struct containing SCS settings\n')
        scs_stgs_fields = list(canon_settings_names_to_default.keys())
        scs_stgs_casts = ['']*len(scs_stgs_fields)
        scs_stgs_values = list(canon_settings_names_to_default.values())
        write_struct_def(f, scs_stgs_fields, scs_stgs_casts, scs_stgs_values, '%sScs_Stgs' % prob_name, 'ScsSettings')

        f.write('\n// SCS solution\n')
        osqp_utils.write_vec(f, np.zeros(canon_constants['n']), '%sscs_x' % prob_name, 'c_float')
        osqp_utils.write_vec(f, np.zeros(canon_constants['m']), '%sscs_y' % prob_name, 'c_float')
        osqp_utils.write_vec(f, np.zeros(canon_constants['m']), '%sscs_s' % prob_name, 'c_float')

        f.write('\n// Struct containing SCS solution\n')
        scs_sol_fields = ['x', 'y', 's']
        scs_sol_casts = ['(c_float *) ', '(c_float *) ', '(c_float *) ']
        scs_sol_values = ['&%sscs_x' % prob_name, '&%sscs_y' % prob_name, '&%sscs_s' % prob_name]
        write_struct_def(f, scs_sol_fields, scs_sol_casts, scs_sol_values, '%sScs_Sol' % prob_name, 'ScsSolution')

        f.write('\n// Struct containing SCS information\n')
        scs_info_fields = ['iter', 'status', 'status_val', 'scale_updates', 'pobj', 'dobj', 'res_pri', 'res_dual',
                           'gap', 'res_infeas', 'res_unbdd_a', 'res_unbdd_p', 'comp_slack', 'setup_time', 'solve_time',
                           'scale', 'rejected_accel_steps', 'accepted_accel_steps', 'lin_sys_time', 'cone_time',
                           'accel_time']
        scs_info_casts = ['']*len(scs_info_fields)
        scs_info_values = ['0', '"unknown"', '0', '0', '0', '0', '99', '99', '99', '99', '99', '99', '99', '0', '0',
                           '1', '0', '0', '0', '0', '0']
        write_struct_def(f, scs_info_fields, scs_info_casts, scs_info_values, '%sScs_Info' % prob_name, 'ScsInfo')

    if solver_name == 'ECOS':

        f.write('\n// Struct containing solver settings\n')
        f.write('Canon_Settings_t %sCanon_Settings = {\n' % prob_name)
        for name, default in canon_settings_names_to_default.items():
            f.write('.%s = %s,\n' % (name, default))
        f.write('};\n')
        if canon_constants['n_cones'] > 0:
            f.write('\n// ECOS array of SOC dimensions\n')
            osqp_utils.write_vec(f, canon_constants['q'], '%secos_q' % prob_name, 'c_int')
        f.write('\n// ECOS workspace\n')
        f.write('pwork* %secos_workspace = 0;\n' % prob_name)
        f.write('\n// ECOS exit flag\n')
        f.write('c_int %secos_flag = -99;\n' % prob_name)


def write_workspace_prot(f, solver_name, explicit, user_p_name_to_size_usp, user_p_writable, user_p_flat_usp, var_init,
                         canon_p_ids, canon_p, canon_maps, var_symmetric, canon_constants,
                         canon_settings_names_to_types, prob_name):
    """"
    Write workspace initialization to file
    """

    if solver_name == 'OSQP':
        f.write('\n#include "types.h"\n\n')
    elif solver_name == 'SCS':
        f.write('\n#include "scs.h"\n\n')
    elif solver_name == 'ECOS':
        f.write('\n#include "ecos.h"\n\n')

    # definition safeguard
    f.write('#ifndef CPG_TYPES_H\n')
    f.write('# define CPG_TYPES_H\n\n')

    if solver_name == 'SCS':
        f.write('typedef scs_float c_float;\n')
        f.write('typedef scs_int c_int;\n\n')
    elif solver_name == 'ECOS':
        f.write('typedef double c_float;\n')
        f.write('typedef int c_int;\n\n')

    # struct definitions
    if solver_name in ['SCS', 'ECOS']:
        f.write('// Compressed sparse column (csc) matrix\n')
        f.write('typedef struct {\n')
        f.write('  c_int      nzmax;\n')
        f.write('  c_int      n;\n')
        f.write('  c_int      m;\n')
        f.write('  c_int      *p;\n')
        f.write('  c_int      *i;\n')
        f.write('  c_float    *x;\n')
        f.write('  c_int      nz;\n')
        f.write('} csc;\n\n')

    if explicit:
        f.write('// User-defined parameters\n')
        f.write('typedef struct {\n')
        # single user parameters
        for name, size in user_p_name_to_size_usp.items():
            if size == 1:
                s = ''
            else:
                s = '*'
            f.write('  c_float    %s   // Your parameter %s\n' % ((s+name+';').ljust(9), name))
        f.write('} CPG_Params_t;\n\n')

    f.write('// Canonical parameters\n')
    f.write('typedef struct {\n')
    for canon_p_id in canon_p_ids:
        if canon_p_id.isupper():
            f.write('  csc        *%s   // Canonical parameter %s\n' %
                    ((canon_p_id+';').ljust(8), canon_p_id))
        else:
            if canon_p_id == 'd':
                s = ''
            else:
                s = '*'
            f.write('  c_float    %s   // Canonical parameter %s\n' %
                    ((s+canon_p_id+';').ljust(9), canon_p_id))
    f.write('} Canon_Params_t;\n\n')

    f.write('// Flags indicating outdated canonical parameters\n')
    f.write('typedef struct {\n')
    for canon_p_id in canon_p_ids:
        f.write('  int        %s    // Bool, if canonical parameter %s outdated\n' %
                ((canon_p_id + ';').ljust(8), canon_p_id))
    f.write('} Canon_Outdated_t;\n\n')

    f.write('// Solver information\n')
    f.write('typedef struct {\n')
    f.write('  c_float    obj_val;    // Objective function value\n')
    f.write('  c_int      iter;       // Number of iterations\n')
    if solver_name in ['OSQP', 'SCS']:
        f.write('  char       *status;    // Solver status\n')
    elif solver_name == 'ECOS':
        f.write('  c_int      status;     // Solver status\n')
    f.write('  c_float    pri_res;    // Primal residual\n')
    f.write('  c_float    dua_res;    // Dual residual\n')
    f.write('} CPG_Info_t;\n\n')

    f.write('// Solution and solver information\n')
    f.write('typedef struct {\n')
    for name, var in var_init.items():
        if np.isscalar(var):
            s = ''
        else:
            s = '*'
        f.write('  c_float    %s   // Your variable %s\n' % ((s+name+';').ljust(9), name))
    f.write('  CPG_Info_t *info;      // Solver info\n')
    f.write('} CPG_Result_t;\n\n')

    if solver_name == 'ECOS':
        f.write('// Solver settings\n')
        f.write('typedef struct {\n')
        for name, typ in canon_settings_names_to_types.items():
            f.write('  %s%s;\n' % (typ.ljust(11), name))
        f.write('} Canon_Settings_t;\n\n')

    f.write('#endif // ifndef CPG_TYPES_H\n')

    if explicit:
        f.write('\n// User-defined parameters\n')
        for name, value in user_p_writable.items():
            if not np.isscalar(value):
                osqp_utils.write_vec_extern(f, value, prob_name+'cpg_'+name, 'c_float')
        f.write('\n// Struct containing all user-defined parameters\n')
        write_struct_prot(f, '%sCPG_Params' % prob_name, 'CPG_Params_t')
    else:
        f.write('\n// Vector containing flattened user-defined parameters\n')
        osqp_utils.write_vec_extern(f, user_p_flat_usp, '%scpg_params_vec' % prob_name, 'c_float')
        f.write('\n// Sparse mappings from user-defined to canonical parameters\n')
        for name, mapping in zip(canon_p_ids, canon_maps):
            if mapping.nnz > 0:
                osqp_utils.write_mat_extern(f, csc_to_dict(mapping), '%scanon_%s_map' % (prob_name, name))

    f.write('\n// Canonical parameters\n')
    for canon_p_id in canon_p_ids:
        if canon_p_id != 'd':
            write_param_prot(f, canon_p[canon_p_id], canon_p_id, '', prob_name)
            if solver_name == 'ECOS':
                write_param_prot(f, canon_p[canon_p_id], canon_p_id, '_ECOS', prob_name)

    f.write('\n// Struct containing canonical parameters\n')
    write_struct_prot(f, '%sCanon_Params' % prob_name, 'Canon_Params_t')
    if solver_name == 'ECOS':
        write_struct_prot(f, '%sCanon_Params_ECOS' % prob_name, 'Canon_Params_t')

    f.write('\n// Struct containing flags for outdated canonical parameters\n')
    f.write('extern Canon_Outdated_t %sCanon_Outdated;\n' % prob_name)

    if any(var_symmetric) or solver_name == 'ECOS':
        f.write('\n// User-defined variables\n')
        for (name, value), symm in zip(var_init.items(), var_symmetric):
            if symm or solver_name == 'ECOS':
                if not np.isscalar(value):
                    osqp_utils.write_vec_extern(f, value.flatten(order='F'), prob_name+'cpg_'+name, 'c_float')

    f.write('\n// Struct containing solver info\n')
    write_struct_prot(f, '%sCPG_Info' % prob_name, 'CPG_Info_t')

    f.write('\n// Struct containing solution and info\n')
    write_struct_prot(f, '%sCPG_Result' % prob_name, 'CPG_Result_t')

    if solver_name == 'SCS':
        f.write('\n// SCS matrix A\n')
        write_struct_prot(f, '%sscs_A' % prob_name, 'ScsMatrix')
        f.write('\n// Struct containing SCS data\n')
        write_struct_prot(f, '%sScs_D' % prob_name, 'ScsData')
        if canon_constants['qsize'] > 0:
            f.write('\n// SCS array of SOC dimensions\n')
            osqp_utils.write_vec_extern(f, canon_constants['q'], '%sscs_q' % prob_name, 'c_int')
        f.write('\n// Struct containing SCS cone data\n')
        write_struct_prot(f, '%sScs_K' % prob_name, 'ScsCone')
        f.write('\n// Struct containing SCS settings\n')
        write_struct_prot(f, '%sScs_Stgs' % prob_name, 'ScsSettings')
        f.write('\n// SCS solution\n')
        osqp_utils.write_vec_extern(f, np.zeros(canon_constants['n']), '%sscs_x' % prob_name, 'c_float')
        osqp_utils.write_vec_extern(f, np.zeros(canon_constants['m']), '%sscs_y' % prob_name, 'c_float')
        osqp_utils.write_vec_extern(f, np.zeros(canon_constants['m']), '%sscs_s' % prob_name, 'c_float')
        f.write('\n// Struct containing SCS solution\n')
        write_struct_prot(f, '%sScs_Sol' % prob_name, 'ScsSolution')
        f.write('\n// Struct containing SCS information\n')
        write_struct_prot(f, '%sScs_Info' % prob_name, 'ScsInfo')

    if solver_name == 'ECOS':
        f.write('\n// Struct containing solver settings\n')
        write_struct_prot(f, '%sCanon_Settings' % prob_name, 'Canon_Settings_t')
        if canon_constants['n_cones'] > 0:
            f.write('\n// ECOS array of SOC dimensions\n')
            osqp_utils.write_vec_extern(f, canon_constants['q'], '%secos_q' % prob_name, 'c_int')
        f.write('\n// ECOS workspace\n')
        f.write('extern pwork* %secos_workspace;\n' % prob_name)
        f.write('\n// ECOS exit flag\n')
        f.write('extern c_int %secos_flag;\n' % prob_name)


def write_solve_def(f, solver_name, explicit, canon_p_ids, mappings, user_p_col_to_name_usp, user_p_sizes_usp,
                    user_p_name_to_size_usp, var_id_to_indices, canon_p_id_to_size, is_maximization,
                    user_p_to_canon_outdated, canon_settings_names_to_types, canon_settings_names_to_default, var_symm,
                    canon_p_to_changes, canon_constants, nonzero_d, ret_sol_func_exists, prob_name):
    """
    Write parameter initialization function to file
    """

    f.write('\n#include "cpg_solve.h"\n')
    f.write('#include "cpg_workspace.h"\n')
    if solver_name == 'OSQP':
        f.write('#include "workspace.h"\n')
        f.write('#include "osqp.h"\n')

    if not explicit:
        f.write('\nstatic c_int i;\n')
        f.write('static c_int j;\n')

    if explicit and solver_name == 'ECOS':
        f.write('\nstatic c_int i;\n')

    base_cols = list(user_p_col_to_name_usp.keys())

    f.write('\n// Update user-defined parameters\n')
    if explicit:
        for (user_p_name, Canon_outdated_names), user_p_size in zip(user_p_to_canon_outdated.items(), user_p_sizes_usp):
            if user_p_size == 1:
                f.write('void %scpg_update_%s(c_float val){\n' % (prob_name, user_p_name))
                f.write('  %sCPG_Params.%s = val;\n' % (prob_name, user_p_name))
            else:
                f.write('void %scpg_update_%s(c_int idx, c_float val){\n' % (prob_name, user_p_name))
                f.write('  %sCPG_Params.%s[idx] = val;\n' % (prob_name, user_p_name))
            for Canon_outdated_name in Canon_outdated_names:
                f.write('  %sCanon_Outdated.%s = 1;\n' % (prob_name, Canon_outdated_name))
            f.write('}\n\n')
    else:
        for base_col, (user_p_name, Canon_outdated_names), user_p_size in zip(base_cols,
                                                                              user_p_to_canon_outdated.items(),
                                                                              user_p_sizes_usp):
            if user_p_size == 1:
                f.write('void %scpg_update_%s(c_float val){\n' % (prob_name, user_p_name))
                f.write('  %scpg_params_vec[%d] = val;\n' % (prob_name, base_col))
            else:
                f.write('void %scpg_update_%s(c_int idx, c_float val){\n' % (prob_name, user_p_name))
                f.write('  %scpg_params_vec[idx+%d] = val;\n' % (prob_name, base_col))
            for Canon_outdated_name in Canon_outdated_names:
                f.write('  %sCanon_Outdated.%s = 1;\n' % (prob_name, Canon_outdated_name))
            f.write('}\n\n')

    f.write('// Map user-defined to canonical parameters\n')

    for canon_name, mapping in zip(canon_p_ids, mappings):
        if mapping.nnz > 0:
            f.write('void %scpg_canonicalize_%s(){\n' % (prob_name, canon_name))
            if canon_name.isupper():
                s = '->x'
            else:
                s = ''
            if explicit:
                write_canonicalize_explicit(f, canon_name, s, mapping, base_cols, user_p_sizes_usp,
                                            user_p_col_to_name_usp, user_p_name_to_size_usp, prob_name)
            else:
                write_canonicalize(f, canon_name, s, mapping, prob_name)
            f.write('}\n\n')

    if solver_name == 'OSQP':
        obj_str = 'workspace.info->obj_val'
        sol_str = 'workspace.solution->x'
    elif solver_name == 'SCS':
        obj_str = '%sScs_Info.pobj' % prob_name
        sol_str = '%sscs_x' % prob_name
    elif solver_name == 'ECOS':
        obj_str = '%secos_workspace->info->pcost' % prob_name
        sol_str = '%secos_workspace->x' % prob_name
    else:
        raise ValueError("Only OSQP and ECOS are supported!")

    if ret_sol_func_exists:
        f.write('// Retrieve solution in terms of user-defined variables\n')
        f.write('void %scpg_retrieve_solution(){\n' % prob_name)
        for symm, (var_id, indices) in zip(var_symm, var_id_to_indices.items()):
            if symm or len(indices) == 1 or solver_name == 'ECOS':
                if len(indices) == 1:
                    f.write('  %sCPG_Result.%s = %s[%d];\n' % (prob_name, var_id, sol_str, indices[0]))
                else:
                    for i, idx in enumerate(indices):
                        f.write('  %sCPG_Result.%s[%d] = %s[%d];\n' % (prob_name, var_id, i, sol_str, idx))
        f.write('}\n\n')

    f.write('// Retrieve solver info\n')
    f.write('void %scpg_retrieve_info(){\n' % prob_name)
    if nonzero_d:
        d_str = ' + *%sCanon_Params.d' % prob_name
    else:
        d_str = ''
    if is_maximization:
        f.write('  %sCPG_Info.obj_val = -(%s%s);\n' % (prob_name, obj_str, d_str))
    else:
        f.write('  %sCPG_Info.obj_val = %s%s;\n' % (prob_name, obj_str, d_str))
    if solver_name == 'OSQP':
        f.write('  %sCPG_Info.iter = workspace.info->iter;\n' % prob_name)
        f.write('  %sCPG_Info.status = workspace.info->status;\n' % prob_name)
        f.write('  %sCPG_Info.pri_res = workspace.info->pri_res;\n' % prob_name)
        f.write('  %sCPG_Info.dua_res = workspace.info->dua_res;\n' % prob_name)
    elif solver_name == 'SCS':
        f.write('  %sCPG_Info.iter = %sScs_Info.iter;\n' % (prob_name, prob_name))
        f.write('  %sCPG_Info.status = %sScs_Info.status;\n' % (prob_name, prob_name))
        f.write('  %sCPG_Info.pri_res = %sScs_Info.res_pri;\n' % (prob_name, prob_name))
        f.write('  %sCPG_Info.dua_res = %sScs_Info.res_dual;\n' % (prob_name, prob_name))
    elif solver_name == 'ECOS':
        f.write('  %sCPG_Info.iter = %secos_workspace->info->iter;\n' % (prob_name, prob_name))
        f.write('  %sCPG_Info.status = %secos_flag;\n' % (prob_name, prob_name))
        f.write('  %sCPG_Info.pri_res = %secos_workspace->info->pres;\n' % (prob_name, prob_name))
        f.write('  %sCPG_Info.dua_res = %secos_workspace->info->dres;\n' % (prob_name, prob_name))
    f.write('}\n\n')

    f.write('// Solve via canonicalization, canonical solve, retrieval\n')
    f.write('void %scpg_solve(){\n' % prob_name)
    f.write('  // Canonicalize if necessary\n')
    if solver_name == 'OSQP':

        if canon_p_to_changes['P'] and canon_p_to_changes['A']:
            f.write('  if (%sCanon_Outdated.P && %sCanon_Outdated.A) {\n' % (prob_name, prob_name))
            f.write('    %scpg_canonicalize_P();\n' % prob_name)
            f.write('    %scpg_canonicalize_A();\n' % prob_name)
            f.write('    osqp_update_P_A(&workspace, %sCanon_Params.P->x, 0, 0, %sCanon_Params.A->x, 0, 0);\n'
                    % (prob_name, prob_name))
            f.write('  } else if (%sCanon_Outdated.P) {\n' % prob_name)
            f.write('    %scpg_canonicalize_P();\n' % prob_name)
            f.write('    osqp_update_P(&workspace, %sCanon_Params.P->x, 0, 0);\n' % prob_name)
            f.write('  } else if (%sCanon_Outdated.A) {\n' % prob_name)
            f.write('    %scpg_canonicalize_A();\n' % prob_name)
            f.write('    osqp_update_A(&workspace, %sCanon_Params.A->x, 0, 0);\n' % prob_name)
            f.write('  }\n')
        else:
            if canon_p_to_changes['P']:
                f.write('  if (%sCanon_Outdated.P) {\n' % prob_name)
                f.write('    %scpg_canonicalize_P();\n' % prob_name)
                f.write('    osqp_update_P(&workspace, %sCanon_Params.P->x, 0, 0);\n' % prob_name)
                f.write('  }\n')
            if canon_p_to_changes['A']:
                f.write('  if (%sCanon_Outdated.A) {\n' % prob_name)
                f.write('    %scpg_canonicalize_A();\n' % prob_name)
                f.write('    osqp_update_A(&workspace, %sCanon_Params.A->x, 0, 0);\n' % prob_name)
                f.write('  }\n')

        if canon_p_to_changes['q']:
            f.write('  if (%sCanon_Outdated.q) {\n' % prob_name)
            f.write('    %scpg_canonicalize_q();\n' % prob_name)
            f.write('    osqp_update_lin_cost(&workspace, %sCanon_Params.q);\n' % prob_name)
            f.write('  }\n')

        if canon_p_to_changes['d']:
            f.write('  if (%sCanon_Outdated.d) {\n' % prob_name)
            f.write('    %scpg_canonicalize_d();\n' % prob_name)
            f.write('  }\n')

        if canon_p_to_changes['l'] and canon_p_to_changes['u']:
            f.write('  if (%sCanon_Outdated.l && %sCanon_Outdated.u) {\n' % (prob_name, prob_name))
            f.write('    %scpg_canonicalize_l();\n' % prob_name)
            f.write('    %scpg_canonicalize_u();\n' % prob_name)
            f.write('    osqp_update_bounds(&workspace, %sCanon_Params.l, %sCanon_Params.u);\n'
                    % (prob_name, prob_name))
            f.write('  } else if (%sCanon_Outdated.l) {\n' % prob_name)
            f.write('    %scpg_canonicalize_l();\n' % prob_name)
            f.write('    osqp_update_lower_bound(&workspace, %sCanon_Params.l);\n' % prob_name)
            f.write('  } else if (%sCanon_Outdated.u) {\n' % prob_name)
            f.write('    %scpg_canonicalize_u();\n' % prob_name)
            f.write('    osqp_update_upper_bound(&workspace, %sCanon_Params.u);\n' % prob_name)
            f.write('  }\n')
        else:
            if canon_p_to_changes['l']:
                f.write('  if (%sCanon_Outdated.l) {\n' % prob_name)
                f.write('    %scpg_canonicalize_l();\n' % prob_name)
                f.write('    osqp_update_lower_bound(&workspace, %sCanon_Params.l);\n' % prob_name)
                f.write('  }\n')
            if canon_p_to_changes['u']:
                f.write('  if (%sCanon_Outdated.u) {\n' % prob_name)
                f.write('    %scpg_canonicalize_u();\n' % prob_name)
                f.write('    osqp_update_upper_bound(&workspace, %sCanon_Params.u);\n' % prob_name)
                f.write('  }\n')

    elif solver_name in ['SCS', 'ECOS']:

        for canon_p, changes in canon_p_to_changes.items():
            if changes:
                f.write('  if (%sCanon_Outdated.%s) {\n' % (prob_name, canon_p))
                f.write('    %scpg_canonicalize_%s();\n' % (prob_name, canon_p))
                f.write('  }\n')

    if solver_name == 'OSQP':
        f.write('  // Solve with OSQP\n')
        f.write('  osqp_solve(&workspace);\n')
    elif solver_name == 'SCS':
        f.write('  // Solve with SCS\n')
        f.write('  scs(&%sScs_D, &%sScs_K, &%sScs_Stgs, &%sScs_Sol, &%sScs_Info);\n' %
                (prob_name, prob_name, prob_name, prob_name, prob_name))
    elif solver_name == 'ECOS':
        f.write('  // Copy raw canonical parameters to addresses where they are scaled by ECOS\n')
        for p_pid, size in canon_p_id_to_size.items():
            if size == 1:
                f.write('  %sCanon_Params_ECOS.%s = %sCanon_Params.%s;\n' % (prob_name, p_pid, prob_name, p_pid))
            elif size > 1:
                f.write('  for (i=0; i<%d; i++){\n' % size)
                if p_pid.isupper():
                    f.write('    %sCanon_Params_ECOS.%s->x[i] = %sCanon_Params.%s->x[i];\n'
                            % (prob_name, p_pid, prob_name, p_pid))
                else:
                    f.write('    %sCanon_Params_ECOS.%s[i] = %sCanon_Params.%s[i];\n'
                            % (prob_name, p_pid, prob_name, p_pid))
                f.write('  }\n')
        f.write('  // Initialize ECOS workspace and settings\n')
        write_ecos_setup(f, canon_constants, prob_name)
        for name in canon_settings_names_to_types.keys():
            f.write('  %secos_workspace->stgs->%s = %sCanon_Settings.%s;\n' % (prob_name, name, prob_name, name))
        f.write('  // Solve with ECOS\n')
        f.write('  %secos_flag = ECOS_solve(%secos_workspace);\n' % (prob_name, prob_name))

    f.write('  // Retrieve results\n')
    if ret_sol_func_exists:
        f.write('  %scpg_retrieve_solution();\n' % prob_name)
    f.write('  %scpg_retrieve_info();\n' % prob_name)

    if solver_name == 'ECOS':
        f.write('  // Clean up ECOS workspace\n')
        f.write('  ECOS_cleanup(%secos_workspace, 0);\n' % prob_name)

    f.write('  // Reset flags for outdated canonical parameters\n')
    for canon_p_id in canon_p_ids:
        f.write('  %sCanon_Outdated.%s = 0;\n' % (prob_name, canon_p_id))

    f.write('}\n\n')

    f.write('// Update solver settings\n')
    f.write('void %scpg_set_solver_default_settings(){\n' % prob_name)
    if solver_name == 'OSQP':
        f.write('  osqp_set_default_settings(&settings);\n')
    elif solver_name == 'SCS':
        f.write('  scs_set_default_settings(&%sScs_Stgs);\n' % prob_name)
    elif solver_name == 'ECOS':
        for name, value in canon_settings_names_to_default.items():
            f.write('  %sCanon_Settings.%s = %s;\n' % (prob_name, name, value))
    f.write('}\n')
    for name, typ in canon_settings_names_to_types.items():
        f.write('\nvoid %scpg_set_solver_%s(%s %s_new){\n' % (prob_name, name, typ, name))
        if solver_name == 'OSQP':
            f.write('  osqp_update_%s(&workspace, %s_new);\n' % (name, name))
        elif solver_name == 'SCS':
            f.write('  %sScs_Stgs.%s = %s_new;\n' % (prob_name, name, name))
        elif solver_name == 'ECOS':
            f.write('  %sCanon_Settings.%s = %s_new;\n' % (prob_name, name, name))
        f.write('}\n')


def write_solve_prot(f, solver_name, canon_p_ids, user_p_name_to_size_usp, canon_settings_names_to_types,
                     ret_sol_func_exists, prob_name):
    """
    Write function declarations to file
    """

    if solver_name == 'OSQP':
        f.write('\n#include "types.h"\n')
    elif solver_name in ['SCS', 'ECOS']:
        f.write('\n#include "cpg_workspace.h"\n')

    f.write('\n// Update user-defined parameter values\n')
    for name, size in user_p_name_to_size_usp.items():
        if size == 1:
            f.write('extern void %scpg_update_%s(c_float val);\n' % (prob_name, name))
        else:
            f.write('extern void %scpg_update_%s(c_int idx, c_float val);\n' % (prob_name, name))

    f.write('\n// Map user-defined to canonical parameters\n')
    for canon_p_id in canon_p_ids:
        f.write('extern void %scpg_canonicalize_%s();\n' % (prob_name, canon_p_id))

    if ret_sol_func_exists:
        f.write('\n// Retrieve solution in terms of user-defined variables\n')
        f.write('extern void %scpg_retrieve_solution();\n' % prob_name)

    f.write('\n// Retrieve solver information\n')
    f.write('extern void %scpg_retrieve_info();\n' % prob_name)

    f.write('\n// Solve via canonicalization, canonical solve, retrieval\n')
    f.write('extern void %scpg_solve();\n' % prob_name)

    f.write('\n// Update solver settings\n')
    f.write('extern void %scpg_set_solver_default_settings();\n' % prob_name)
    for name, typ in canon_settings_names_to_types.items():
        f.write('extern void %scpg_set_solver_%s(%s %s_new);\n' % (prob_name, name, typ, name))


def write_example_def(f, solver_name, user_p_writable, var_name_to_size, prob_name):
    """
    Write main function to file
    """

    f.write('int main(int argc, char *argv[]){\n\n')

    f.write('  // Initialize user-defined parameter values\n')
    for name, value in user_p_writable.items():
        if np.isscalar(value):
            f.write('  %scpg_update_%s(%.20f);\n' % (prob_name, name, value))
        else:
            for i in range(len(value)):
                f.write('  %scpg_update_%s(%d, %.20f);\n' % (prob_name, name, i, value[i]))

    f.write('\n  // Solve the problem instance\n')
    f.write('  %scpg_solve();\n\n' % prob_name)

    f.write('  // Print objective function value\n')
    f.write('  printf("obj = %%f\\n", %sCPG_Result.info->obj_val);\n\n' % prob_name)

    f.write('  // Print solution\n')

    if solver_name == 'OSQP':
        int_format_str = 'lld'
    else:
        int_format_str = 'd'

    for name, size in var_name_to_size.items():
        if size == 1:
            f.write('  printf("%s = %%f\\n", %sCPG_Result.%s);\n\n' % (name, prob_name, name))
        else:
            f.write('  for(i=0; i<%d; i++) {\n' % size)
            f.write('    printf("%s[%%%s] = %%f\\n", i, %sCPG_Result.%s[i]);\n'
                    % (name, int_format_str, prob_name, name))
            f.write('  }\n\n')

    f.write('  return 0;\n\n')
    f.write('}\n')


def replace_cmake_data(cmake_data, prob_name):
    """
    Add 'prob_name' prefix to directory/file lists in top-level CMakeLists.txt
    """

    cmake_data = cmake_data.replace('cpg_include', prob_name+'cpg_include')
    cmake_data = cmake_data.replace('cpg_head', prob_name + 'cpg_head')
    return cmake_data.replace('cpg_src', prob_name + 'cpg_src')


def write_canon_cmake(f, solver_name):
    """
    Pass sources to parent scope in {OSQP/ECOS}_code/CMakeLists.txt
    """

    if solver_name == 'OSQP':
        f.write('\nset(solver_head "${osqp_headers}" PARENT_SCOPE)')
        f.write('\nset(solver_src "${osqp_src}" PARENT_SCOPE)')
    elif solver_name == 'SCS':
        f.write('\nset(solver_head')
        f.write('\n  ${${PROJECT_NAME}_HDR}')
        f.write('\n  ${DIRSRC}/private.h')
        f.write('\n  ${${PROJECT_NAME}_LDL_EXTERNAL_HDR}')
        f.write('\n  ${${PROJECT_NAME}_AMD_EXTERNAL_HDR})')
        f.write('\nset(solver_src')
        f.write('\n  ${${PROJECT_NAME}_SRC}')
        f.write('\n  ${DIRSRC}/private.c')
        f.write('\n  ${EXTERNAL}/qdldl/qdldl.c')
        f.write('\n  ${${PROJECT_NAME}_AMD_EXTERNAL_SRC})')
        f.write('\n\nset(solver_head "${solver_head}" PARENT_SCOPE)')
        f.write('\nset(solver_src "${solver_src}" PARENT_SCOPE)')
    elif solver_name == 'ECOS':
        f.write('\nset(solver_head "${ecos_headers}" PARENT_SCOPE)')
        f.write('\nset(solver_src "${ecos_sources}" PARENT_SCOPE)')


def write_module_def(f, user_p_name_to_size_usp, var_name_to_size, canon_settings_names, prob_name):
    """
    Write c++ file for pbind11 wrapper
    """

    # cpp function that maps parameters to results
    f.write('%sCPG_Result_cpp_t %ssolve_cpp(struct %sCPG_Updated_cpp_t& CPG_Updated_cpp, '
            'struct %sCPG_Params_cpp_t& CPG_Params_cpp){\n\n'
            % (prob_name, prob_name, prob_name, prob_name))

    f.write('    // Pass changed user-defined parameter values to the solver\n')
    for name, size in user_p_name_to_size_usp.items():
        f.write('    if (CPG_Updated_cpp.%s) {\n' % name)
        if size == 1:
            f.write('        %scpg_update_%s(CPG_Params_cpp.%s);\n' % (prob_name, name, name))
        else:
            f.write('        for(i=0; i<%d; i++) {\n' % size)
            f.write('            %scpg_update_%s(i, CPG_Params_cpp.%s[i]);\n' % (prob_name, name, name))
            f.write('        }\n')
        f.write('    }\n')

    # perform ASA procedure
    f.write('\n    // Solve\n')
    f.write('    std::clock_t ASA_start = std::clock();\n')
    f.write('    %scpg_solve();\n' % prob_name)
    f.write('    std::clock_t ASA_end = std::clock();\n\n')

    # arrange and return results
    f.write('    // Arrange and return results\n')
    f.write('    %sCPG_Info_cpp_t CPG_Info_cpp {};\n' % prob_name)
    for field in ['obj_val', 'iter', 'status', 'pri_res', 'dua_res']:
        f.write('    CPG_Info_cpp.%s = %sCPG_Info.%s;\n' % (field, prob_name, field))
    f.write('    CPG_Info_cpp.ASA_proc_time = 1000.0 * (ASA_end-ASA_start) / CLOCKS_PER_SEC;\n')

    f.write('    %sCPG_Result_cpp_t CPG_Result_cpp {};\n' % prob_name)
    f.write('    CPG_Result_cpp.info = CPG_Info_cpp;\n')
    for name, size in var_name_to_size.items():
        if size == 1:
            f.write('    CPG_Result_cpp.%s = %sCPG_Result.%s;\n' % (name, prob_name, name))
        else:
            f.write('    for(i=0; i<%d; i++) {\n' % size)
            f.write('        CPG_Result_cpp.%s[i] = %sCPG_Result.%s[i];\n' % (name, prob_name, name))
            f.write('    }\n')

    # return
    f.write('    return CPG_Result_cpp;\n\n')
    f.write('}\n\n')

    # module
    f.write('PYBIND11_MODULE(cpg_module, m) {\n\n')

    f.write('    py::class_<%sCPG_Params_cpp_t>(m, "%scpg_params")\n' % (prob_name, prob_name))
    f.write('            .def(py::init<>())\n')
    for name in user_p_name_to_size_usp.keys():
        f.write('            .def_readwrite("%s", &%sCPG_Params_cpp_t::%s)\n' % (name, prob_name, name))
    f.write('            ;\n\n')

    f.write('    py::class_<%sCPG_Updated_cpp_t>(m, "%scpg_updated")\n' % (prob_name, prob_name))
    f.write('            .def(py::init<>())\n')
    for name in user_p_name_to_size_usp.keys():
        f.write('            .def_readwrite("%s", &%sCPG_Updated_cpp_t::%s)\n' % (name, prob_name, name))
    f.write('            ;\n\n')

    f.write('    py::class_<%sCPG_Info_cpp_t>(m, "%scpg_info")\n' % (prob_name, prob_name))
    f.write('            .def(py::init<>())\n')
    f.write('            .def_readwrite("obj_val", &%sCPG_Info_cpp_t::obj_val)\n' % prob_name)
    f.write('            .def_readwrite("iter", &%sCPG_Info_cpp_t::iter)\n' % prob_name)
    f.write('            .def_readwrite("status", &%sCPG_Info_cpp_t::status)\n' % prob_name)
    f.write('            .def_readwrite("pri_res", &%sCPG_Info_cpp_t::pri_res)\n' % prob_name)
    f.write('            .def_readwrite("dua_res", &%sCPG_Info_cpp_t::dua_res)\n' % prob_name)
    f.write('            .def_readwrite("ASA_proc_time", &%sCPG_Info_cpp_t::ASA_proc_time)\n' % prob_name)
    f.write('            ;\n\n')

    f.write('    py::class_<%sCPG_Result_cpp_t>(m, "%scpg_result")\n' % (prob_name, prob_name))
    f.write('            .def(py::init<>())\n')
    f.write('            .def_readwrite("cpg_info", &%sCPG_Result_cpp_t::info)\n' % prob_name)
    for name in var_name_to_size.keys():
        f.write('            .def_readwrite("%s", &%sCPG_Result_cpp_t::%s)\n' % (name, prob_name, name))
    f.write('            ;\n\n')

    f.write('    m.def("solve", &%ssolve_cpp);\n\n' % prob_name)

    f.write('    m.def("set_solver_default_settings", &%scpg_set_solver_default_settings);\n' % prob_name)
    for name in canon_settings_names:
        f.write('    m.def("set_solver_%s", &%scpg_set_solver_%s);\n' % (name, prob_name, name))

    f.write('\n}\n')


def write_module_prot(f, solver_name, user_p_name_to_size_usp, var_name_to_size, prob_name):
    """
    Write c++ file for pbind11 wrapper
    """

    # cpp struct containing user-defined parameters
    f.write('\n// User-defined parameters\n')
    f.write('struct %sCPG_Params_cpp_t {\n' % prob_name)
    for name, size in user_p_name_to_size_usp.items():
        if size == 1:
            f.write('    double %s;\n' % name)
        else:
            f.write('    std::array<double, %d> %s;\n' % (size, name))
    f.write('};\n\n')

    # cpp struct containing update flags for user-defined parameters
    f.write('// Flags for updated user-defined parameters\n')
    f.write('struct %sCPG_Updated_cpp_t {\n' % prob_name)
    for name in user_p_name_to_size_usp.keys():
        f.write('    bool %s;\n' % name)
    f.write('};\n\n')

    # cpp struct containing info on results
    f.write('// Solver information\n')
    f.write('struct %sCPG_Info_cpp_t {\n' % prob_name)
    f.write('    double obj_val;\n')
    f.write('    int iter;\n')
    if solver_name in ['OSQP', 'SCS']:
        f.write('    char* status;\n')
    elif solver_name == 'ECOS':
        f.write('    int status;\n')
    f.write('    double pri_res;\n')
    f.write('    double dua_res;\n')
    f.write('    double ASA_proc_time;\n')
    f.write('};\n\n')

    # cpp struct containing objective value and user-defined variables
    f.write('// Solution and solver information\n')
    f.write('struct %sCPG_Result_cpp_t {\n' % prob_name)
    f.write('    %sCPG_Info_cpp_t info;\n' % prob_name)
    for name, size in var_name_to_size.items():
        if size == 1:
            f.write('    double %s;\n' % name)
        else:
            f.write('    std::array<double, %d> %s;\n' % (size, name))
    f.write('};\n\n')

    # cpp function that maps parameters to results
    f.write('// Main solve function\n')
    f.write('%sCPG_Result_cpp_t %ssolve_cpp(struct %sCPG_Updated_cpp_t& CPG_Updated_cpp, '
            'struct %sCPG_Params_cpp_t& CPG_Params_cpp);\n'
            % (prob_name, prob_name, prob_name, prob_name))


def write_method(f, solver_name, code_dir, user_p_name_to_size_usp, user_p_name_to_sparsity,
                 user_p_name_to_sparsity_type, var_name_to_shape, problem_name):
    """
    Write function to be registered as custom CVXPY solve method
    """

    f.write('from %s import cpg_module\n\n\n' % code_dir.replace('/', '.').replace('\\', '.'))

    if solver_name == 'ECOS':
        indent = ' ' * 24
        f.write('status_int_to_string = {0: "Optimal solution found", \n' +
                indent + '1: "Certificate of primal infeasibility found", \n' +
                indent + '2: "Certificate of dual infeasibility found", \n' +
                indent + '10: "Optimal solution found subject to reduced tolerances", \n' +
                indent + '11: "Certificate of primal infeasibility found subject to reduced tolerances", \n' +
                indent + '12: "Certificate of dual infeasibility found subject to reduced tolerances", \n' +
                indent + '-1: "Maximum number of iterations reached", \n' +
                indent + '-2: "Numerical problems (unreliable search direction)", \n' +
                indent + '-3: "Numerical problems (slacks or multipliers outside cone)", \n' +
                indent + '-4: "Interrupted by signal or CTRL-C", \n' +
                indent + '-7: "Unknown problem in solver", \n' +
                indent + '-99: "Unknown problem before solving"}\n\n\n')

    f.write('def cpg_solve(prob, updated_params=None, **kwargs):\n\n')
    f.write('    # set flags for updated parameters\n')
    f.write('    upd = cpg_module.%scpg_updated()\n' % problem_name)
    f.write('    if updated_params is None:\n')
    p_list_string = ''
    for name in user_p_name_to_size_usp.keys():
        p_list_string += '"%s", ' % name
    f.write('        updated_params = [%s]\n' % p_list_string[:-2])
    f.write('    for p in updated_params:\n')
    f.write('        try:\n')
    f.write('            setattr(upd, p, True)\n')
    f.write('        except AttributeError:\n')
    f.write('            raise(AttributeError("%s is not a parameter." % p))\n\n')

    f.write('    # set solver settings\n')
    f.write('    cpg_module.set_solver_default_settings()\n')
    f.write('    for key, value in kwargs.items():\n')
    if solver_name == 'ECOS':
        f.write('        if key == "max_iters":\n')
        f.write('            key = "maxit"\n')
    f.write('        try:\n')
    f.write('            eval(\'cpg_module.set_solver_%s(value)\' % key)\n')
    f.write('        except AttributeError:\n')
    f.write('            raise(AttributeError(\'Solver setting "%s" not available.\' % key))\n\n')

    f.write('    # set parameter values\n')
    f.write('    par = cpg_module.%scpg_params()\n' % problem_name)
    for name, size in user_p_name_to_size_usp.items():
        if name in user_p_name_to_sparsity.keys():
            if user_p_name_to_sparsity_type[name] == 'diag':
                f.write('    %s_coordinates = [(i, i) for i in range(prob.param_dict[\'%s\'].shape[0])]\n'
                        % (name, name))
            else:
                f.write('    %s_coordinates = prob.param_dict[\'%s\'].attributes[\'sparsity\']\n' % (name, name))
            if size == 1:
                f.write('    par.%s = prob.param_dict[\'%s\'].value[coordinates]\n' % (name, name))
            else:
                f.write('    %s_value = []\n' % name)
                f.write('    for j in range(prob.param_dict[\'%s\'].shape[1]):\n' % name)
                f.write('        for i in range(prob.param_dict[\'%s\'].shape[0]):\n' % name)
                f.write('            if (i, j) in %s_coordinates:\n' % name)
                f.write('                %s_value.append(prob.param_dict[\'%s\'].value[i, j])\n' % (name, name))
                f.write('            elif prob.param_dict[\'%s\'].value[i, j] != 0:\n' % name)
                f.write('                warnings.warn(\'Ignoring nonzero value outside of '
                        'sparsity pattern for parameter %s!\')\n' % name)
                f.write('    par.%s = %s_value\n' % (name, name))
        else:
            if size == 1:
                f.write('    par.%s = prob.param_dict[\'%s\'].value\n' % (name, name))
            else:
                f.write('    par.%s = list(prob.param_dict[\'%s\'].value.flatten(order=\'F\'))\n' % (name, name))

    f.write('\n    # solve\n')
    f.write('    t0 = time.time()\n')
    f.write('    res = cpg_module.solve(upd, par)\n')
    f.write('    t1 = time.time()\n\n')

    f.write('    # store solution in problem object\n')
    f.write('    prob._clear_solution()\n')
    for name, shape in var_name_to_shape.items():
        if len(shape) == 2:
            f.write('    prob.var_dict[\'%s\'].value = np.array(res.%s).reshape((%d, %d), order=\'F\')\n' %
                    (name, name, shape[0], shape[1]))
        else:
            f.write('    prob.var_dict[\'%s\'].value = np.array(res.%s)\n' % (name, name))

    f.write('\n    # store additional solver information in problem object\n')
    if solver_name in ['OSQP', 'SCS']:
        f.write('    prob._status = res.cpg_info.status\n')
    elif solver_name == 'ECOS':
        f.write('    prob._status = status_int_to_string[res.cpg_info.status]\n')
    f.write('    if abs(res.cpg_info.obj_val) == 1e30:\n')
    f.write('        prob._value = np.sign(res.cpg_info.obj_val)*np.inf\n')
    f.write('    else:\n')
    f.write('        prob._value = res.cpg_info.obj_val\n')
    f.write('    primal_vars = {var.id: var.value for var in prob.variables()}\n')
    f.write('    dual_vars = {}\n')
    f.write('    solver_specific_stats = {\'obj_val\': res.cpg_info.obj_val,\n')
    f.write('                             \'status\': prob._status,\n')
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
    f.write('    prob._solver_stats = SolverStats(results_dict, \'%s\')\n\n' % solver_name)

    f.write('    return prob.value\n')


def replace_html_data(code_dir, solver_name, explicit, text, user_p_name_to_size_usp, user_p_writable, var_name_to_size,
                      user_p_total_size, canon_p_ids, canon_p_id_to_size, canon_settings_names_to_types,
                      canon_constants, canon_maps, ret_sol_func_exists, prob_name):
    """
    Replace placeholder strings in html documentation file
    """

    # code_dir
    text = text.replace('$CODEDIR', code_dir)
    text = text.replace('$CDPYTHON', code_dir.replace('/', '.').replace('\\', '.'))

    # solver name and docu
    text = text.replace('$CPGSOLVERNAME', solver_name)
    if solver_name == 'OSQP':
        text = text.replace('$CPGSOLVERDOCUURL', 'https://osqp.org/docs/codegen/python.html')
    elif solver_name == 'SCS':
        text = text.replace('$CPGSOLVERDOCUURL', 'https://www.cvxgrp.org/scs/api/c.html')
    elif solver_name == 'ECOS':
        text = text.replace('$CPGSOLVERDOCUURL', 'https://github.com/embotech/ecos/wiki/Usage-from-C')

    # CMake prefix
    text = text.replace('$CPGCMAKELISTS', prob_name+'cpg')
        
    # basic type definitions
    if solver_name in ['SCS', 'ECOS']:
        if solver_name == 'SCS':
            CPGBASICTYPEDEF = '\ntypedef scs_float c_float;\n'
            CPGBASICTYPEDEF += 'typedef scs_int c_int;\n\n'
        else:
            CPGBASICTYPEDEF = '\ntypedef double c_float;\n'
            CPGBASICTYPEDEF += 'typedef int c_int;\n\n'
        CPGBASICTYPEDEF += '\n// Compressed sparse column (csc) matrix\n'
        CPGBASICTYPEDEF += 'typedef struct {\n'
        CPGBASICTYPEDEF += '  c_int      nzmax;\n'
        CPGBASICTYPEDEF += '  c_int      n;\n'
        CPGBASICTYPEDEF += '  c_int      m;\n'
        CPGBASICTYPEDEF += '  c_int      *p;\n'
        CPGBASICTYPEDEF += '  c_int      *i;\n'
        CPGBASICTYPEDEF += '  c_float    *x;\n'
        CPGBASICTYPEDEF += '  c_int      nz;\n'
        CPGBASICTYPEDEF += '} csc;\n'
        text = text.replace('$CPGBASICTYPEDEF', CPGBASICTYPEDEF)
    else:
        text = text.replace('$CPGBASICTYPEDEF\n', '')

    # type definition of CPG_Params_t or cpg_params_vec
    if explicit:
        CPGUSERPARAMSTYPEDEF = '\n// Struct type with user-defined parameters as fields\n'
        CPGUSERPARAMSTYPEDEF += 'typedef struct {\n'
        for name, size in user_p_name_to_size_usp.items():
            if size == 1:
                s = ''
            else:
                s = '*'
            CPGUSERPARAMSTYPEDEF += ('  c_float    %s   // Your parameter %s\n' % ((s+name+';').ljust(9), name))
        CPGUSERPARAMSTYPEDEF += '} CPG_Params_t;\n'
        text = text.replace('$CPGUSERPARAMSTYPEDEF', CPGUSERPARAMSTYPEDEF)
    else:
        text = text.replace('$CPGUSERPARAMSTYPEDEF\n', '')

    # type definition of Canon_Params_t
    CPGCANONPARAMSTYPEDEF = '\n// Struct type with canonical parameters as fields\n'
    CPGCANONPARAMSTYPEDEF += 'typedef struct {\n'
    for name in canon_p_ids:
        if name.isupper():
            CPGCANONPARAMSTYPEDEF += ('  csc        *%s   // Canonical parameter %s\n' % ((name+';').ljust(8), name))
        else:
            if name == 'd':
                s = ''
            else:
                s = '*'
            CPGCANONPARAMSTYPEDEF += ('  c_float    %s   // Canonical parameter %s\n' % ((s+name+';').ljust(9), name))
    CPGCANONPARAMSTYPEDEF += '} Canon_Params_t;\n'
    text = text.replace('$CPGCANONPARAMSTYPEDEF', CPGCANONPARAMSTYPEDEF)

    # type definition of Canon_Outdated_t
    CPGOUTDATEDTYPEDEF = '\n// Struct type with booleans as fields, ' \
                         'indicating if respective canonical parameter is outdated\n'
    CPGOUTDATEDTYPEDEF += 'typedef struct {\n'
    for name in canon_p_ids:
        CPGOUTDATEDTYPEDEF += ('  int        %s    // Bool, if canonical parameter %s outdated\n'
                               % ((name + ';').ljust(8), name))
    CPGOUTDATEDTYPEDEF += '} Canon_Outdated_t;\n'
    text = text.replace('$CPGOUTDATEDTYPEDEF', CPGOUTDATEDTYPEDEF)

    # type definition of CPG_Info_t
    CPGINFOTYPEDEF = '\n// Solver information\n'
    CPGINFOTYPEDEF += 'typedef struct {\n'
    CPGINFOTYPEDEF += '  c_float    obj_val;    // Objective function value\n'
    CPGINFOTYPEDEF += '  c_int      iter;       // Number of iterations\n'
    if solver_name in ['OSQP', 'SCS']:
        CPGINFOTYPEDEF += '  char       *status;    // Solver status\n'
    elif solver_name == 'ECOS':
        CPGINFOTYPEDEF += '  c_int      status;     // Solver status\n'
    CPGINFOTYPEDEF += '  c_float    pri_res;    // Primal residual\n'
    CPGINFOTYPEDEF += '  c_float    dua_res;    // Dual residual\n'
    CPGINFOTYPEDEF += '} CPG_Info_t;\n'
    text = text.replace('$CPGINFOTYPEDEF', CPGINFOTYPEDEF)

    # type definition of CPG_Result_t
    CPGRESULTTYPEDEF = '\n// Struct type with user-defined objective value and solution as fields\n'
    CPGRESULTTYPEDEF += 'typedef struct {\n'
    for name, size in var_name_to_size.items():
        if size == 1:
            s = ''
        else:
            s = '*'
        CPGRESULTTYPEDEF += ('  c_float    %s   // Your variable %s\n' % ((s+name+';').ljust(9), name))
    CPGRESULTTYPEDEF += '  CPG_Info_t *info;      // Solver information\n'
    CPGRESULTTYPEDEF += '} CPG_Result_t;\n'
    text = text.replace('$CPGRESULTTYPEDEF', CPGRESULTTYPEDEF)

    if solver_name == 'ECOS':
        CPGSETTINGSTYPEDEF = '\n// Solver settings\n'
        CPGSETTINGSTYPEDEF += 'typedef struct {\n'
        for name, typ in canon_settings_names_to_types.items():
            CPGSETTINGSTYPEDEF += '  %s%s;\n' % (typ.ljust(11), name)
        CPGSETTINGSTYPEDEF += '} Canon_Settings_t;\n'
        text = text.replace('$CPGSETTINGSTYPEDEF', CPGSETTINGSTYPEDEF)
    else:
        text = text.replace('$CPGSETTINGSTYPEDEF\n', '')

    # parameter delarations
    if explicit:
        CPGPARAMDECLARATIONS = '\n// User-defined parameters\n'
        for name, value in user_p_writable.items():
            if not np.isscalar(value):
                CPGPARAMDECLARATIONS += 'c_float %s[%d];\n' % (prob_name+'cpg_'+name, value.size)
        CPGPARAMDECLARATIONS += '\n\n// Struct containing all user-defined parameters\n'
        CPGPARAMDECLARATIONS += 'CPG_Params_t %sCPG_Params;\n' % prob_name
    else:
        CPGPARAMDECLARATIONS = '\n// Vector containing flattened user-defined parameters\n'
        CPGPARAMDECLARATIONS += 'c_float %scpg_params_vec[%d];\n' % (prob_name, user_p_total_size + 1)
        CPGPARAMDECLARATIONS += '\n\n// Sparse mappings from user-defined to canonical parameters\n'
        for name, mapping in zip(canon_p_ids, canon_maps):
            if mapping.nnz > 0:
                CPGPARAMDECLARATIONS += 'csc %scanon_%s_map;\n' % (prob_name, name)
    text = text.replace('$CPGPARAMDECLARATIONS', CPGPARAMDECLARATIONS)

    # canonical parameter declarations
    CPGCANONPARAMDECLARATIONS = '\n// Canonical parameters\n'
    for p_id, size in canon_p_id_to_size.items():
        if size > 0:
            if p_id.isupper():
                CPGCANONPARAMDECLARATIONS += 'csc %scanon_%s;\n' % (prob_name, p_id)
                if solver_name == 'ECOS':
                    CPGCANONPARAMDECLARATIONS += 'csc %scanon_%s_ECOS;\n' % (prob_name, p_id)
            elif p_id != 'd':
                CPGCANONPARAMDECLARATIONS += 'c_float %scanon_%s[%d];\n' % (prob_name, p_id, size)
                if solver_name == 'ECOS':
                    CPGCANONPARAMDECLARATIONS += 'c_float %scanon_%s_ECOS[%d];\n' % (prob_name, p_id, size)
    CPGCANONPARAMDECLARATIONS += '\n\n// Struct containing canonical parameters\n'
    CPGCANONPARAMDECLARATIONS += 'Canon_Params_t %sCanon_Params;\n' % prob_name
    if solver_name == 'ECOS':
        CPGCANONPARAMDECLARATIONS += 'Canon_Params_t %sCanon_Params_ECOS;\n' % prob_name
    text = text.replace('$CPGCANONPARAMDECLARATIONS', CPGCANONPARAMDECLARATIONS)

    # outdated declarations
    CPGCANONOUTDATEDDECLARATION = '\n// Struct containing flags for outdated canonical parameters\n'
    CPGCANONOUTDATEDDECLARATION += 'Canon_Outdated_t %sCanon_Outdated;\n' % prob_name
    text = text.replace('$CPGCANONOUTDATEDDECLARATION', CPGCANONOUTDATEDDECLARATION)

    # variable declarations
    CPGVARIABLEDECLARATIONS = '\n// User-defined variables\n'
    for name, size in var_name_to_size.items():
        if size > 1:
            CPGVARIABLEDECLARATIONS += 'c_float %s[%d];\n' % (prob_name+'cpg_'+name, size)
    text = text.replace('$CPGVARIABLEDECLARATIONS', CPGVARIABLEDECLARATIONS)

    # solver info and result declarations
    CPGINFORESULTDECLARATION = '\n// Struct containing solver info\n'
    CPGINFORESULTDECLARATION += 'CPG_Info_t %sCPG_Info;\n\n' % prob_name
    CPGINFORESULTDECLARATION += '\n// Struct containing user-defined objective value and solution\n'
    CPGINFORESULTDECLARATION += 'CPG_Result_t %sCPG_Result;\n' % prob_name
    text = text.replace('$CPGINFORESULTDECLARATION', CPGINFORESULTDECLARATION)

    # extra declarations
    if solver_name == 'SCS':
        CPGEXTRADECLARATIONS = '\n// SCS matrix A\n'
        CPGEXTRADECLARATIONS += 'ScsMatrix %sscs_A;\n\n' % prob_name
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS data\n'
        CPGEXTRADECLARATIONS += 'ScsData %sScs_D;\n\n' % prob_name
        if canon_constants['qsize'] > 0:
            CPGEXTRADECLARATIONS += '\n// SCS array of SOC dimensions\n'
            CPGEXTRADECLARATIONS += 'c_int %sscs_q[%d];\n\n' % (prob_name, canon_constants['qsize'])
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS cone data\n'
        CPGEXTRADECLARATIONS += 'ScsCone %sScs_K;\n\n' % prob_name
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS settings\n'
        CPGEXTRADECLARATIONS += 'ScsSettings %sScs_Stgs;\n\n' % prob_name
        CPGEXTRADECLARATIONS += '\n// SCS solution\n'
        CPGEXTRADECLARATIONS += 'c_float %sscs_x[%d];\n' % (prob_name, canon_constants['n'])
        CPGEXTRADECLARATIONS += 'c_float %sscs_y[%d];\n' % (prob_name, canon_constants['m'])
        CPGEXTRADECLARATIONS += 'c_float %sscs_s[%d];\n\n' % (prob_name, canon_constants['m'])
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS solution\n'
        CPGEXTRADECLARATIONS += 'ScsSolution %sScs_Sol;\n\n' % prob_name
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS information\n'
        CPGEXTRADECLARATIONS += 'ScsInfo %sScs_Info;\n' % prob_name
        text = text.replace('$CPGEXTRADECLARATIONS', CPGEXTRADECLARATIONS)
    elif solver_name == 'ECOS':
        CPGEXTRADECLARATIONS = '\n// Struct containing solver settings\n'
        CPGEXTRADECLARATIONS += 'Canon_Settings_t %sCanon_Settings;\n\n' % prob_name
        if canon_constants['n_cones'] > 0:
            CPGEXTRADECLARATIONS += '\n// ECOS array of SOC dimensions\n'
            CPGEXTRADECLARATIONS += 'c_int %secos_q[%d];\n\n' % (prob_name, canon_constants['n_cones'])
        CPGEXTRADECLARATIONS += '\n// ECOS workspace\n'
        CPGEXTRADECLARATIONS += 'pwork* %secos_workspace;\n\n' % prob_name
        CPGEXTRADECLARATIONS += '\n// ECOS exit flag\n'
        CPGEXTRADECLARATIONS += 'c_int %secos_flag;\n' % prob_name
        text = text.replace('$CPGEXTRADECLARATIONS', CPGEXTRADECLARATIONS)
    else:
        text = text.replace('$CPGEXTRADECLARATIONS\n', '')

    # update declarations
    CPGUPDATEDECLARATIONS = '\n// Update user-defined parameter values\n'
    for name, size in user_p_name_to_size_usp.items():
        if size == 1:
            CPGUPDATEDECLARATIONS += 'void %scpg_update_%s(c_float value);\n' % (prob_name, name)
        else:
            CPGUPDATEDECLARATIONS += 'void %scpg_update_%s(c_int idx, c_float value);\n' % (prob_name, name)
    text = text.replace('$CPGUPDATEDECLARATIONS', CPGUPDATEDECLARATIONS)

    # canonicalize declarations
    CPGCANONICALIZEDECLARATIONS = '\n// Map user-defined to canonical parameters\n'
    for p_id in canon_p_ids:
        CPGCANONICALIZEDECLARATIONS += 'void %scpg_canonicalize_%s();\n' % (prob_name, p_id)
    text = text.replace('$CPGCANONICALIZEDECLARATIONS', CPGCANONICALIZEDECLARATIONS)

    # retrieve and solve declarations
    CPGRETRIEVESOLVEDECLARATIONS = ''
    if ret_sol_func_exists:
        CPGRETRIEVESOLVEDECLARATIONS += '\n// Retrieve solution in terms of user-defined variables\n'
        CPGRETRIEVESOLVEDECLARATIONS += 'void %scpg_retrieve_solution();\n\n' % prob_name
    CPGRETRIEVESOLVEDECLARATIONS += '\n// Retrieve solver information\n'
    CPGRETRIEVESOLVEDECLARATIONS += 'void %scpg_retrieve_info();\n\n' % prob_name
    CPGRETRIEVESOLVEDECLARATIONS += '\n// Solve via canonicalization, canonical solve, retrieval\n'
    CPGRETRIEVESOLVEDECLARATIONS += 'void %scpg_solve();\n' % prob_name
    text = text.replace('$CPGRETRIEVESOLVEDECLARATIONS', CPGRETRIEVESOLVEDECLARATIONS)

    # settings declarations
    CPGSETTINGSDECLARATIONS = '\n// Update solver settings\n'
    CPGSETTINGSDECLARATIONS += 'void %scpg_set_solver_default_settings();\n' % prob_name
    CPGSETTINGSDECLARATIONS += 'void %scpg_set_solver_&lt;setting_name&gt;' \
                               '(&lt;setting_type&gt;, &lt;setting_name&gt;_new);\n' % prob_name
    CPGSETTINGSDECLARATIONS += '...\n'

    return text.replace('$CPGSETTINGSDECLARATIONS', CPGSETTINGSDECLARATIONS)
