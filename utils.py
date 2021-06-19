
import numpy as np
from osqp.codegen import utils as osqp_utils


def write_row_sum(f, OSQP_name, rows_to_sum_OSQP, shape):
    """
    Write row summation of decomposed OSQP param matrix to file
    """

    if OSQP_name in ['P', 'A']:
        s = '->x'
    else:
        s = ''

    for row in rows_to_sum_OSQP:
        expr = ''
        for col in range(shape[1]):
            expr += 'OSQP_work->%s_decomposed[%d]+' % (OSQP_name, col*shape[0]+row)
        expr = expr[:-1]
        f.write('OSQP_work->%s%s[%d] = %s;\n' % (OSQP_name, s, row, expr))


def write_types(f, names):
    """
    Write user parameters as struct type to file
    """
    f.write('typedef struct {\n')

    # single user parameters
    for name in names:
        f.write('    c_float     *%s;\n' % name)

    f.write('} Workspace_t;\n\n')

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


def write_update_decomposed(f, OSQP_names, mappings, offsets, user_name, n_eq, problem_data_index_A):
    """
    Write decomposed parameter update function to file
    """

    # remember which rows of decomposed OSQP matrices need to be summed
    rows_to_sum = []

    f.write('void update_decomposed_%s(Workspace_t *work, OSQP_Workspace_t *OSQP_work){\n' % user_name)

    # consider all OSQP parameters that user parameter maps to
    for (OSQP_name, mapping, offset) in zip(OSQP_names, mappings, offsets):
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
        rows_to_sum_OSQP = []
        for row in range(n_rows):
            expr = ''
            data = mapping.data[mapping.indptr[row]:mapping.indptr[row+1]]
            columns = mapping.indices[mapping.indptr[row]:mapping.indptr[row+1]]
            for (datum, col) in zip(data, columns):
                expr += '(%.20f*work->%s[%d])+' % (sign*datum, user_name, col)
            expr = expr[:-1]
            if data.size > 0:
                OSQP_row = OSQP_rows[row]
                f.write('OSQP_work->%s_decomposed[%d] = %s;\n' % (OSQP_name, offset+OSQP_row, expr))
                rows_to_sum_OSQP.append(OSQP_row)
        rows_to_sum.append(rows_to_sum_OSQP)

    f.write('}\n')

    return rows_to_sum


def write_update_decomposed_extern(f, name):
    """
    Write decomposed parameter update function to file
    """

    f.write('extern void update_decomposed_%s(Workspace_t *work, OSQP_Workspace_t *OSQP_work);\n' % name)


def write_update(f, OSQP_names, rows_to_sum, shapes, name):
    """
    Write parameter update function to file
    """

    f.write('void update_%s(Workspace_t *work, OSQP_Workspace_t *OSQP_work){\n' % name)
    f.write('update_decomposed_%s(work, OSQP_work);\n' % name)

    for (OSQP_name, rows_to_sum_OSQP, shape) in zip(OSQP_names, rows_to_sum, shapes):
        write_row_sum(f, OSQP_name, rows_to_sum_OSQP, shape)

    f.write('}\n')


def write_update_extern(f, name):
    """
    Write parameter update function to file
    """

    f.write('extern void update_%s(Workspace_t *work, OSQP_Workspace_t *OSQP_work);\n' % name)


def write_init(f, OSQP_names, user_names, shapes):
    """
    Write parameter initialization function to file
    """

    f.write('void init_params(Workspace_t *work, OSQP_Workspace_t *OSQP_work){\n')

    for user_name in user_names:
        f.write('update_decomposed_%s(work, OSQP_work);\n' % user_name)

    for (OSQP_name, shape) in zip(OSQP_names, shapes):
        write_row_sum(f, OSQP_name, range(shape[0]), shape)

    f.write('}\n')


def write_init_extern(f):
    """
    Write parameter initialization function to file
    """

    f.write('extern void init_params(Workspace_t *work, OSQP_Workspace_t *OSQP_work);\n')


def write_solve(f):
    """
    Write solve function to file
    """

    f.write('void solve(OSQP_Workspace_t *work){\n')
    f.write('printf("Hello World!\\n");\n')
    f.write('}\n')


def write_solve_extern(f):
    """
    Write solve function to file
    """

    f.write('extern void solve(OSQP_Workspace_t *work);\n')


def write_main(f, user_p_writable):
    """
    Write main function to file
    """

    f.write('int main(int argc, char *argv[]){\n')

    for name, value in user_p_writable.items():
        for i in range(len(value)):
            f.write('Workspace.%s[%d] = %.20f;\n' % (name, i, value[i]))

    f.write('init_params(&Workspace, &OSQP_Workspace);\n')
    f.write('solve(&OSQP_Workspace);\n')
    f.write('return 0;\n')
    f.write('}\n')
