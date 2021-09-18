
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
            osqp_utils.write_vec(f, value.flatten(), name, 'c_float')
            results_cast.append('(c_float *) ')

    f.write('\n// Struct containing CPG objective value and solution\n')
    CPG_Result_fields = ['objective_value'] + list(var_init.keys())
    write_struct(f, CPG_Result_fields, results_cast, CPG_Result_fields, 'CPG_Result', 'CPG_Result_t')

    # Boolean struct for outdated parameter flags
    f.write('OSQP_Outdated_t OSQP_Outdated = {\n')
    for OSQP_p_id in OSQP_p_ids:
        f.write('.%s = 1,\n' % OSQP_p_id)

    f.write('};\n')


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
            osqp_utils.write_vec_extern(f, value.flatten(), name, 'c_float')

    f.write('\n// Struct containing CPG objective value and solution\n')
    write_struct_extern(f, 'CPG_Result', 'CPG_Result_t')


def write_solve(f, OSQP_p_ids, nonconstant_OSQP_p_ids, mappings, user_p_col_to_name, sizes, n_eq, problem_data_index_A,
                var_id_to_indices, is_maximization, user_p_to_OSQP_outdated, OSQP_settings_names_to_types):
    """
    Write parameter initialization function to file
    """

    f.write('// update user-defined parameters\n')

    for user_p_name, OSQP_outdated_names in user_p_to_OSQP_outdated.items():
        f.write('void update_%s(){\n' % user_p_name)
        for OSQP_outdated_name in OSQP_outdated_names:
            f.write('OSQP_Outdated.%s = 1;\n' % OSQP_outdated_name)
        f.write('}\n')

    f.write('\n// map user-defined to OSQP-accepted parameters\n')

    base_cols = list(user_p_col_to_name.keys())

    for OSQP_name, mapping in zip(OSQP_p_ids, mappings):

        f.write('void canonicalize_OSQP_%s(){\n' % OSQP_name)

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

        f.write('}\n')

    f.write('\n// retrieve user-defined objective function value\n')
    f.write('void retrieve_value(){\n')

    if is_maximization:
        f.write('objective_value = -(workspace.info->obj_val + *OSQP_Params.d);\n')
    else:
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

    f.write('if (OSQP_Outdated.P && OSQP_Outdated.A) {\n')
    f.write('canonicalize_OSQP_P();\n')
    f.write('canonicalize_OSQP_A();\n')
    f.write('osqp_update_P_A(&workspace, OSQP_Params.P->x, 0, 0, OSQP_Params.A->x, 0, 0);\n')
    f.write('} else if (OSQP_Outdated.P) {\n')
    f.write('canonicalize_OSQP_P();\n')
    f.write('osqp_update_P(&workspace, OSQP_Params.P->x, 0, 0);\n')
    f.write('} else if (OSQP_Outdated.P) {\n')
    f.write('canonicalize_OSQP_A();\n')
    f.write('osqp_update_A(&workspace, OSQP_Params.A->x, 0, 0);\n')
    f.write('}\n')

    f.write('if (OSQP_Outdated.q) {\n')
    f.write('canonicalize_OSQP_q();\n')
    f.write('osqp_update_lin_cost(&workspace, OSQP_Params.q);\n')
    f.write('}\n')

    f.write('if (OSQP_Outdated.d) {\n')
    f.write('canonicalize_OSQP_d();\n')
    f.write('}\n')

    f.write('if (OSQP_Outdated.l && OSQP_Outdated.u) {\n')
    f.write('canonicalize_OSQP_l();\n')
    f.write('canonicalize_OSQP_u();\n')
    f.write('osqp_update_bounds(&workspace, OSQP_Params.l, OSQP_Params.u);\n')
    f.write('} else if (OSQP_Outdated.l) {\n')
    f.write('canonicalize_OSQP_l();\n')
    f.write('osqp_update_lower_bound(&workspace, OSQP_Params.l);\n')
    f.write('} else if (OSQP_Outdated.u) {\n')
    f.write('canonicalize_OSQP_u();\n')
    f.write('osqp_update_upper_bound(&workspace, OSQP_Params.u);\n')
    f.write('}\n')

    f.write('osqp_solve(&workspace);\n')
    f.write('retrieve_value();\n')
    f.write('retrieve_solution();\n')

    for OSQP_p_id in OSQP_p_ids:
        f.write('OSQP_Outdated.%s = 0;\n' % OSQP_p_id)

    f.write('}\n\n')

    f.write('// update OSQP settings\n')
    f.write('void set_OSQP_default_settings(){\n')
    f.write('osqp_set_default_settings(&settings);\n')
    f.write('}\n')
    for name, typ in OSQP_settings_names_to_types.items():
        f.write('void set_OSQP_%s(%s %s_new){\n' % (name, typ, name))
        f.write('osqp_update_%s(&workspace, %s_new);\n' % (name, name))
        f.write('}\n')


def write_solve_extern(f, user_p_names, OSQP_settings_names_to_types):
    """
    Write function declarations to file
    """

    for name in user_p_names:
        f.write('extern void update_%s();\n' % name)

    f.write('\n// update OSQP settings\n')
    f.write('extern void set_OSQP_default_settings();\n')
    for name, typ in OSQP_settings_names_to_types.items():
        f.write('extern void set_OSQP_%s(%s %s_new);\n' % (name, typ, name))


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

    f.write('\n// pass changed user-defined parameter values to the solver\n')
    for name in user_p_writable.keys():
        f.write('update_%s();\n' % name)

    f.write('\n// solve the problem instance\n')
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


def write_module(f, user_p_name_to_size, var_name_to_size, OSQP_settings_names):
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

    # cpp struct containing update flags for user-defined parameters
    f.write('struct CPG_Updated_cpp_t {\n')
    for name in user_p_name_to_size.keys():
        f.write('    bool %s;\n' % name)
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
    f.write('CPG_Result_cpp_t solve_cpp(struct CPG_Updated_cpp_t& CPG_Updated_cpp, struct CPG_Params_cpp_t& CPG_Params_cpp){\n\n')

    f.write('    // pass changed user-defined parameter values to the solver\n')
    for name, size in user_p_name_to_size.items():
        f.write('    if (CPG_Updated_cpp.%s) {\n' % name)
        if size == 1:
            f.write('        %s = CPG_Params_cpp.%s;\n' % (name, name))
        else:
            f.write('        for(int i = 0; i < %d; i++) {\n' % size)
            f.write('            %s[i] = CPG_Params_cpp.%s[i];\n' % (name, name))
            f.write('        }\n')
        f.write('        update_%s();\n' % name)
        f.write('    }\n')

    # perform ASA procedure
    f.write('\n    // ASA\n')
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

    f.write('    py::class_<CPG_Updated_cpp_t>(m, "cpg_updated")\n')
    f.write('            .def(py::init<>())\n')
    for name in user_p_name_to_size.keys():
        f.write('            .def_readwrite("%s", &CPG_Updated_cpp_t::%s)\n' % (name, name))
    f.write('            ;\n\n')

    f.write('    py::class_<CPG_Result_cpp_t>(m, "cpg_result")\n')
    f.write('            .def(py::init<>())\n')
    f.write('            .def_readwrite("objective_value", &CPG_Result_cpp_t::objective_value)\n')
    for name in var_name_to_size.keys():
        f.write('            .def_readwrite("%s", &CPG_Result_cpp_t::%s)\n' % (name, name))
    f.write('            ;\n\n')

    f.write('    m.def("solve", &solve_cpp);\n\n')

    f.write('    m.def("set_OSQP_default_settings", &set_OSQP_default_settings);\n')
    for name in OSQP_settings_names:
        f.write('    m.def("set_OSQP_%s", &set_OSQP_%s);\n' % (name, name))

    f.write('\n}')


def write_method(f, code_dir, user_p_name_to_size, var_name_to_shape):
    """
    Write function to be registered as custom CVXPY solve method
    """

    f.write('from %s import cpg_module\n\n\n' % code_dir.replace('/', '.'))
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
            f.write('    par.%s = list(prob.param_dict[\'%s\'].value.flatten())\n' % (name, name))

    f.write('\n    res = cpg_module.solve(upd, par)\n\n')

    for name, shape in var_name_to_shape.items():
        if len(shape) == 2:
            f.write('    prob.var_dict[\'%s\'].value = np.array(res.%s).reshape((%d, %d))\n' % (name, name, shape[0], shape[1]))
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

    text = text.replace('$CPGVARIABLEDECLARATIONS', CPGVARIABLEDECLARATIONS[:-1])

    # update declarations
    CPGUPDATEDECLARATIONS = ''
    for name in user_p_names:
        CPGUPDATEDECLARATIONS += 'void update_%s();\n' % name

    return text.replace('$CPGUPDATEDECLARATIONS', CPGUPDATEDECLARATIONS[:-1])
