
import numpy as np
from osqp.codegen import utils as osqp_utils


def write_types(f, names):
    """
    Write user parameters as struct type to file
    """
    f.write('typedef struct {\n')

    # single user parameters
    for name in names:
        f.write('    c_float     *%s;\n' % name)

    f.write('} CPG_Workspace_t;\n\n')

    f.write('#endif // ifndef CPG_TYPES_H\n')


def write_osqp(f, param, name):
    """
    Use osqp.codegen.utils for writing vectors and matrices
    """
    if name in ['P', 'A']:
        osqp_utils.write_mat(f, param, name)
    else:
        osqp_utils.write_vec(f, param, name, 'c_float')


def write_osqp_extern(f, param, name):
    """
    Use osqp.codegen.utils for writing vectors and matrices
    """
    if name in ['P', 'A']:
        osqp_utils.write_mat_extern(f, param, name)
    else:
        osqp_utils.write_vec_extern(f, param, name, 'c_float')


def write_dense_mat(f, mat, name):
    """
    Write dense matrix to file
    """

    f.write('c_float %s[%d] = {\n' % (name, mat.size))

    # represent matrix as vector (Fortran style)
    for j in range(mat.shape[1]):
        for i in range(mat.shape[0]):
            f.write('(c_float)%.20f,\n' % mat[i, j])

    f.write('};\n')


def write_dense_mat_extern(f, mat, name):
    """
    Write dense matrix to file
    """

    f.write("extern c_float %s[%d];\n" % (name, mat.size))


def write_struct(f, fields, values, name, typ):
    """
    Write structure to file
    """

    f.write('%s %s = {\n' % (typ, name))

    # write structure fields
    for i in range(len(fields)):
        if fields[i] in ['P', 'A']:
            cast = ''
        else:
            cast = '(c_float *) '
        f.write('.%s = %s&%s,\n' % (fields[i], cast, values[i]))

    f.write('};\n')


def write_struct_extern(f, name, typ):
    """
    Write structure to file
    """

    f.write("extern %s %s;\n" % (typ, name))


def write_update(f, OSQP_p_ids, nonconstant_OSQP_p_ids, mappings, user_p_col_to_name, sizes, n_eq, problem_data_index_A, var_id_to_indices):
    """
    Write parameter initialization function to file
    """

    f.write('void canonicalize_params(){\n')

    base_cols = list(user_p_col_to_name.keys())

    for OSQP_name, mapping in zip(OSQP_p_ids, mappings):

        if OSQP_name in ['P', 'A']:
            s = '->x'
        else:
            s = ''

        if OSQP_name in ['l', 'u']:
            sign = -1
            (Alu_indices, Alu_indptr, _) = problem_data_index_A
            OSQP_rows = Alu_indices[Alu_indptr[-2]:Alu_indptr[-1]]
            if OSQP_name == 'l':
                n_rows = np.count_nonzero(OSQP_rows < n_eq)
            else:
                n_rows = len(mapping.indptr)-1
        else:
            sign = 1
            n_rows = len(mapping.indptr)-1
            OSQP_rows = np.arange(n_rows)

        for row in range(n_rows):
            expr = ''
            data = mapping.data[mapping.indptr[row]:mapping.indptr[row+1]]
            columns = mapping.indices[mapping.indptr[row]:mapping.indptr[row+1]]
            for (datum, col) in zip(data, columns):
                ex = '(%.20f)+' % (sign*datum)
                for i, user_p_col in enumerate(base_cols):
                    if user_p_col + sizes[i] > col:
                        user_name = user_p_col_to_name[user_p_col]
                        ex = '(%.20f*CPG_Workspace.%s[%d])+' % (sign*datum, user_name, col-user_p_col)
                        break
                expr += ex
            expr = expr[:-1]
            if data.size > 0:
                OSQP_row = OSQP_rows[row]
                f.write('OSQP_Workspace.%s%s[%d] = %s;\n' % (OSQP_name, s, OSQP_row, expr))

    f.write('}\n\n')

    f.write('void init_params(){\n')

    f.write('canonicalize_params();\n')
    f.write('osqp_update_P(&workspace, OSQP_Workspace.P->x, 0, 0);\n')
    f.write('osqp_update_lin_cost(&workspace, OSQP_Workspace.q);\n')
    f.write('osqp_update_A(&workspace, OSQP_Workspace.A->x, 0, 0);\n')
    f.write('osqp_update_bounds(&workspace, OSQP_Workspace.l, OSQP_Workspace.u);\n')

    f.write('}\n\n')

    f.write('void update_params(){\n')

    f.write('canonicalize_params();\n')

    if 'P' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_P(&workspace, OSQP_Workspace.P->x, 0, 0);\n')

    if 'q' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_lin_cost(&workspace, OSQP_Workspace.q);\n')

    if 'A' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_A(&workspace, OSQP_Workspace.A->x, 0, 0);\n')

    if 'l' in nonconstant_OSQP_p_ids and 'u' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_bounds(&workspace, OSQP_Workspace.l, OSQP_Workspace.u);\n')
    elif 'l' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_lower_bound(&workspace, OSQP_Workspace.l);\n')
    elif 'u' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_upper_bound(&workspace, OSQP_Workspace.u);\n')

    f.write('}\n\n')

    f.write('void retrieve_solution(){\n')

    for var_id, indices in var_id_to_indices.items():
        for i, idx in enumerate(indices):
            f.write('%s[%d] = workspace.solution->x[%d];\n' % (var_id, i, idx))

    f.write('}\n')


def write_update_extern(f):
    """
    Write parameter initialization function to file
    """

    f.write('extern void canonicalize_params();\n')
    f.write('extern void init_params();\n')
    f.write('extern void update_params();\n')
    f.write('extern void retrieve_solution();\n')


def write_solve(f):
    """
    Write solve function to file
    """

    f.write('void solve(){\n')
    f.write('update_params();\n')
    f.write('osqp_solve(&workspace);\n')
    f.write('retrieve_solution();\n')
    f.write('}\n')


def write_solve_extern(f):
    """
    Write solve function to file
    """

    f.write('extern void solve();\n')


def write_main(f, user_p_writable, var_name_to_size):
    """
    Write main function to file
    """

    f.write('int main(int argc, char *argv[]){\n')

    for name, value in user_p_writable.items():
        for i in range(len(value)):
            f.write('CPG_Workspace.%s[%d] = %.20f;\n' % (name, i, value[i]))

    f.write('init_params();\n')
    f.write('solve();\n')

    f.write('// printing solution\n')

    for name, size in var_name_to_size.items():
        f.write('for(int i = 0; i < %d; i++) {\n' % size)
        f.write('printf("%s[%%d] = %%f \\n", i, %s[i]);\n' % (name, name))
        f.write('}\n')

    f.write('return 0;\n')
    f.write('}\n')


def write_OSQP_CMakeLists(f):
    """
    Pass sources to parent scope in OSQP_code/CMakeLists.txt
    """

    f.write('\nset(osqp_src "${osqp_src}" PARENT_SCOPE)')