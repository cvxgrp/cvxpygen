
import numpy as np
from osqp.codegen import utils as osqp_utils


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


def write_osqp(f, param, name):
    """
    Use osqp.codegen.utils for writing vectors and matrices
    """
    if name in ['P', 'A']:
        osqp_utils.write_mat(f, param, 'OSQP_' + name)
    elif name == 'd':
        f.write('c_float OSQP_d = %.20f;\n' % param[0])
    else:
        osqp_utils.write_vec(f, param, 'OSQP_' + name, 'c_float')


def write_osqp_extern(f, param, name):
    """
    Use osqp.codegen.utils for writing vectors and matrices
    """
    if name in ['P', 'A']:
        osqp_utils.write_mat_extern(f, param, 'OSQP_' + name)
    elif name == 'd':
        f.write('extern c_float OSQP_d;\n')
    else:
        osqp_utils.write_vec_extern(f, param, 'OSQP_' + name, 'c_float')


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


def write_struct(f, fields, casts, values, name, typ):
    """
    Write structure to file
    """

    f.write('%s %s = {\n' % (typ, name))

    # write structure fields
    for field, cast, value in zip(fields, casts, values):
        f.write('.%s = %s&%s,\n' % (field, cast, value))

    f.write('};\n')


def write_struct_extern(f, name, typ):
    """
    Write structure to file
    """

    f.write("extern %s %s;\n" % (typ, name))


def write_workspace(f, user_p_names, user_p_writable, var_init, OSQP_p_ids, OSQP_p):

    OSQP_casts = []

    f.write('// Parameters accepted by OSQP\n')
    for OSQP_p_id in OSQP_p_ids:
        write_osqp(f, replace_inf(OSQP_p[OSQP_p_id]), OSQP_p_id)
        if OSQP_p_id in ['P', 'A', 'd']:
            OSQP_casts.append('')
        else:
            OSQP_casts.append('(c_float *) ')

    f.write('\n// Struct containing parameters accepted by OSQP\n')

    write_struct(f, OSQP_p_ids, OSQP_casts, ['OSQP_'+p for p in OSQP_p_ids], 'OSQP_Params', 'OSQP_Params_t')

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
    write_struct(f, user_p_names, user_casts, user_p_names, 'CPG_Params', 'CPG_Params_t')

    f.write('\n// Value of the objective function\n')
    f.write('c_float objective_value = 0;\n')

    results_cast = ['']

    f.write('\n// User-defined variables\n')
    for name, value in var_init.items():
        if np.isscalar(value):
            f.write('c_float %s = %.20f;\n' % (name, value))
            results_cast.append('')
        else:
            osqp_utils.write_vec(f, value.flatten(order='F'), name, 'c_float')
            results_cast.append('(c_float *) ')

    f.write('\n// Struct containing CPG objective value and solution\n')
    CPG_Result_fields = ['objective_value'] + list(var_init.keys())
    write_struct(f, CPG_Result_fields, results_cast, CPG_Result_fields, 'CPG_Result', 'CPG_Result_t')


def write_workspace_extern(f, user_p_names, user_p_writable, var_init, OSQP_p_ids, OSQP_p):
    """"
    Write workspace initialization to file
    """

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

    f.write('\n// Parameters accepted by OSQP\n')
    for OSQP_p_id in OSQP_p_ids:
        write_osqp_extern(f, OSQP_p[OSQP_p_id], OSQP_p_id)

    f.write('\n// Struct containing parameters accepted by OSQP\n')
    write_struct_extern(f, 'OSQP_Params', 'OSQP_Params_t')

    f.write('\n// User-defined parameters\n')
    for name, value in user_p_writable.items():
        if np.isscalar(value):
            f.write("extern c_float %s;\n" % name)
        else:
            osqp_utils.write_vec_extern(f, value, name, 'c_float')

    f.write('\n// Struct containing all user-defined parameters\n')
    write_struct_extern(f, 'CPG_Params', 'CPG_Params_t')

    f.write('\n// Value of the objective function\n')
    f.write('extern c_float objective_value;\n')

    f.write('\n// User-defined variables\n')
    for name, value in var_init.items():
        if np.isscalar(value):
            f.write("extern c_float %s;\n" % name)
        else:
            osqp_utils.write_vec_extern(f, value.flatten(order='F'), name, 'c_float')

    f.write('\n// Struct containing CPG objective value and solution\n')
    write_struct_extern(f, 'CPG_Result', 'CPG_Result_t')


def write_solve(f, OSQP_p_ids, nonconstant_OSQP_p_ids, mappings, user_p_col_to_name, sizes, n_eq, problem_data_index_A, var_id_to_indices):
    """
    Write parameter initialization function to file
    """

    f.write('// map user-defined to OSQP-accepted parameters\n')
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
                        ex = '(%.20f*CPG_Params.%s[%d])+' % (sign*datum, user_name, col-user_p_col)
                        break
                expr += ex
            expr = expr[:-1]
            if data.size > 0:
                OSQP_row = OSQP_rows[row]
                f.write('OSQP_Params.%s%s[%d] = %s;\n' % (OSQP_name, s, OSQP_row, expr))

    f.write('}\n\n')

    f.write('// initialize all OSQP-accepted parameters\n')
    f.write('void init_params(){\n')

    f.write('canonicalize_params();\n')
    f.write('osqp_update_P(&workspace, OSQP_Params.P->x, 0, 0);\n')
    f.write('osqp_update_lin_cost(&workspace, OSQP_Params.q);\n')
    f.write('osqp_update_A(&workspace, OSQP_Params.A->x, 0, 0);\n')
    f.write('osqp_update_bounds(&workspace, OSQP_Params.l, OSQP_Params.u);\n')

    f.write('}\n\n')

    f.write('// update OSQP-accepted parameters that depend on user-defined parameters\n')
    f.write('void update_params(){\n')
    f.write('canonicalize_params();\n')

    if 'P' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_P(&workspace, OSQP_Params.P->x, 0, 0);\n')

    if 'q' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_lin_cost(&workspace, OSQP_Params.q);\n')

    if 'A' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_A(&workspace, OSQP_Params.A->x, 0, 0);\n')

    if 'l' in nonconstant_OSQP_p_ids and 'u' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_bounds(&workspace, OSQP_Params.l, OSQP_Params.u);\n')
    elif 'l' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_lower_bound(&workspace, OSQP_Params.l);\n')
    elif 'u' in nonconstant_OSQP_p_ids:
        f.write('osqp_update_upper_bound(&workspace, OSQP_Params.u);\n')

    f.write('}\n\n')

    f.write('// retrieve user-defined objective function value\n')
    f.write('void retrieve_value(){\n')
    f.write('objective_value = workspace.info->obj_val + *OSQP_Params.d;\n')
    f.write('}\n\n')

    f.write('// retrieve solution in terms of user-defined variables\n')
    f.write('void retrieve_solution(){\n')

    for var_id, indices in var_id_to_indices.items():
        if len(indices) == 1:
            f.write('%s = workspace.solution->x[%d];\n' % (var_id, indices[0]))
        else:
            for i, idx in enumerate(indices):
                f.write('%s[%d] = workspace.solution->x[%d];\n' % (var_id, i, idx))

    f.write('}\n\n')

    f.write('// perform one ASA sequence to solve a problem instance\n')
    f.write('void solve(){\n')
    f.write('update_params();\n')
    f.write('osqp_solve(&workspace);\n')
    f.write('retrieve_value();\n')
    f.write('retrieve_solution();\n')
    f.write('}\n')


def write_main(f, user_p_writable, var_name_to_size):
    """
    Write main function to file
    """

    f.write('int main(int argc, char *argv[]){\n\n')

    f.write('// initialize user-defined parameter values\n')
    for name, value in user_p_writable.items():
        if np.isscalar(value):
            f.write('*CPG_Params.%s = %.20f;\n' % (name, value))
        else:
            for i in range(len(value)):
                f.write('CPG_Params.%s[%d] = %.20f;\n' % (name, i, value[i]))

    f.write('\n// initialize OSQP-accepted parameter values, this must be done once before solving for the first time\n')
    f.write('init_params();\n\n')

    f.write('// solve the problem instance\n')
    f.write('solve();\n\n')

    f.write('// printing objective function value for demonstration purpose\n')
    f.write('printf("f = %f \\n", objective_value);\n\n')

    f.write('// printing solution for demonstration purpose\n')

    for name, size in var_name_to_size.items():
        if size == 1:
            f.write('printf("%s = %%f \\n", %s);\n' % (name, name))
        else:
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


def write_module(f, user_p_name_to_size, var_name_to_size):
    """
    Write c++ file for pbind11 wrapper
    """

    # cpp struct containing user-defined parameters
    f.write('struct CPG_Params_cpp_t {\n')
    for name, size in user_p_name_to_size.items():
        if size == 1:
            f.write('    double %s;\n' % name)
        else:
            f.write('    std::array<double, %d> %s;\n' % (size, name))
    f.write('};\n\n')

    # cpp struct containing objective value and user-defined variables
    f.write('struct CPG_Result_cpp_t {\n')
    f.write('    double objective_value;\n')
    for name, size in var_name_to_size.items():
        if size == 1:
            f.write('    double %s;\n' % name)
        else:
            f.write('    std::array<double, %d> %s;\n' % (size, name))
    f.write('};\n\n')

    # cpp function that maps parameters to results
    f.write('CPG_Result_cpp_t solve_cpp(struct CPG_Params_cpp_t& CPG_Params_cpp){\n\n')
    f.write('    // pass parameter values to C variables\n')
    for name, size in user_p_name_to_size.items():
        if size == 1:
            f.write('    %s = CPG_Params_cpp.%s;\n' % (name, name))
        else:
            f.write('    for(int i = 0; i < %d; i++) {\n' % size)
            f.write('        %s[i] = CPG_Params_cpp.%s[i];\n' % (name, name))
            f.write('    }\n')

    # perform ASA procedure
    f.write('\n    // ASA\n')
    f.write('    init_params();\n')
    f.write('    solve();\n\n')

    # arrange and return results
    f.write('    // arrange and return results\n')
    f.write('    CPG_Result_cpp_t CPG_Result_cpp {};\n')
    f.write('    CPG_Result_cpp.objective_value = objective_value;\n')
    for name, size in var_name_to_size.items():
        if size == 1:
            f.write('    CPG_Result_cpp.%s = %s;\n' % (name, name))
        else:
            f.write('    for(int i = 0; i < %d; i++) {\n' % size)
            f.write('        CPG_Result_cpp.%s[i] = %s[i];\n' % (name, name))
            f.write('    }\n')

    # return
    f.write('    return CPG_Result_cpp;\n\n')
    f.write('}\n\n')

    # module
    f.write('PYBIND11_MODULE(cpg_module, m) {\n\n')
    f.write('    py::class_<CPG_Params_cpp_t>(m, "cpg_params")\n')
    f.write('            .def(py::init<>())\n')
    for name in user_p_name_to_size.keys():
        f.write('            .def_readwrite("%s", &CPG_Params_cpp_t::%s)\n' % (name, name))
    f.write('            ;\n\n')

    f.write('    py::class_<CPG_Result_cpp_t>(m, "cpg_result")\n')
    f.write('            .def(py::init<>())\n')
    f.write('            .def_readwrite("objective_value", &CPG_Result_cpp_t::objective_value)\n')
    for name in var_name_to_size.keys():
        f.write('            .def_readwrite("%s", &CPG_Result_cpp_t::%s)\n' % (name, name))
    f.write('            ;\n\n')

    f.write('    m.def("solve", &solve_cpp);\n\n')
    f.write('}')


def write_method(f, code_dir, user_p_name_to_size, var_name_to_shape):
    """
    Write function to be registered as custom CVXPY solve method
    """

    f.write('from %s.build import cpg_module\n\n\n' % code_dir.replace('/', '.'))
    f.write('def cpg_solve(prob):\n\n')
    f.write('    par = cpg_module.cpg_params()\n')

    for name, size in user_p_name_to_size.items():
        if size == 1:
            f.write('    par.%s = prob.param_dict[\'%s\'].value\n' % (name, name))
        else:
            f.write('    par.%s = list(prob.param_dict[\'%s\'].value.flatten(order=\'F\'))\n' % (name, name))

    f.write('\n    res = cpg_module.solve(par)\n\n')

    for name, shape in var_name_to_shape.items():
        if len(shape) == 2:
            f.write('    prob.var_dict[\'%s\'].value = np.array(res.%s).reshape((%d, %d), order=\'F\')\n' % (name, name, shape[0], shape[1]))
        else:
            f.write('    prob.var_dict[\'%s\'].value = np.array(res.%s)\n' % (name, name))

    f.write('\n    return res.objective_value\n')


def replace_html(code_dir, text, user_p_names, user_p_writable, var_name_to_size):
    """
    Replace placeholder strings in html documentation file
    """

    # code_dir
    text = text.replace('$CODEDIR', code_dir)
    text = text.replace('$CDPYTHON', code_dir.replace('/', '.'))

    # type definition of CPG_Params_t
    CPGPARAMSTYPEDEF = 'typedef struct {\n'
    for name in user_p_names:
        CPGPARAMSTYPEDEF += ('    c_float     *%s;' % name).ljust(33) + ('///< Your parameter %s\n' % name)
    CPGPARAMSTYPEDEF += '} CPG_Params_t;'

    text = text.replace('$CPGPARAMSTYPEDEF', CPGPARAMSTYPEDEF)

    # type definition of CPG_Result_t
    CPGRESULTTYPEDEF = 'typedef struct {\n'
    CPGRESULTTYPEDEF += '    c_float     *objective_value;///< Objective function value\n'
    for name in var_name_to_size.keys():
        CPGRESULTTYPEDEF += ('    c_float     *%s;' % name).ljust(33) + ('///< Your variable %s\n' % name)
    CPGRESULTTYPEDEF += '} CPG_Result_t;'

    text = text.replace('$CPGRESULTTYPEDEF', CPGRESULTTYPEDEF)

    # parameter delarations
    CPGPARAMDECLARATIONS = ''
    for name, value in user_p_writable.items():
        if np.isscalar(value):
            CPGPARAMDECLARATIONS += 'c_float %s;\n' % name
        else:
            CPGPARAMDECLARATIONS += 'c_float %s[%d];\n' % (name, value.size)

    text = text.replace('$CPGPARAMDECLARATIONS', CPGPARAMDECLARATIONS[:-2])

    # variable declarations
    CPGVARIABLEDECLARATIONS = ''
    for name, size in var_name_to_size.items():
        if size == 1:
            CPGVARIABLEDECLARATIONS += 'c_float %s;\n' % name
        else:
            CPGVARIABLEDECLARATIONS += 'c_float %s[%d];\n' % (name, size)

    return text.replace('$CPGVARIABLEDECLARATIONS', CPGVARIABLEDECLARATIONS[:-1])
