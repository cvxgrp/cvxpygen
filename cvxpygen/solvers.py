import os
import shutil
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from platform import system

import numpy as np

from cvxpygen.utils import replace_in_file, write_struct_prot, write_struct_def, write_vec_prot, write_vec_def
from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, ConstraintInfo, AffineMap, \
    ParameterCanon, ResultPointerInfo


def get_interface_class(solver_name: str) -> "SolverInterface":
    mapping = {
        'OSQP': OSQPInterface,
        'SCS': SCSInterface,
        'ECOS': ECOSInterface,
    }
    interface = mapping.get(solver_name, None)
    if interface is None:
        raise ValueError(f'Unsupported solver: {solver_name}.')
    return interface


class SolverInterface(ABC):

    def __init__(self, solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                 indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings):
        self.solver_name = solver_name
        self.n_var = n_var
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.indices_obj = indices_obj
        self.indptr_obj = indptr_obj
        self.shape_obj = shape_obj
        self.indices_constr = indices_constr
        self.indptr_constr = indptr_constr
        self.shape_constr = shape_constr
        self.canon_constants = canon_constants
        self.enable_settings = enable_settings

        self.configure_settings()

    @property
    @abstractmethod
    def canon_p_ids(self):
        pass

    @property
    @abstractmethod
    def canon_p_ids_constr_vec(self):
        pass

    @property
    @abstractmethod
    def sign_constr_vec(self):
        pass

    @property
    @abstractmethod
    def stgs_names(self):
        pass

    @property
    @abstractmethod
    def stgs_types(self):
        pass

    @property
    @abstractmethod
    def stgs_defaults(self):
        pass

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return any(variable_info.sym) or any([s == 1 for s in variable_info.sizes])

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return any([s == 1 for s in dual_variable_info.sizes])

    def configure_settings(self) -> None:
        for i, s in enumerate(self.stgs_names):
            if s in self.enable_settings:
                self.stgs_enabled[i] = True
        for s in set(self.enable_settings)-set(self.stgs_names):
            warnings.warn('Cannot enable setting %s for solver %s' % (s, self.solver_name))

    def get_affine_map(self, p_id, param_prob, constraint_info: ConstraintInfo) -> AffineMap:
        affine_map = AffineMap()

        if p_id == 'c':
            affine_map.mapping = param_prob.c[:-1]
        elif p_id == 'd':
            affine_map.mapping = param_prob.c[-1]
        elif p_id == 'A':
            affine_map.mapping_rows = constraint_info.mapping_rows_eq[
                constraint_info.mapping_rows_eq < constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_eq, self.n_var)
        elif p_id == 'G':
            affine_map.mapping_rows = constraint_info.mapping_rows_ineq[
                constraint_info.mapping_rows_ineq < constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_ineq, self.n_var)
        elif p_id == 'b':
            affine_map.mapping_rows = constraint_info.mapping_rows_eq[
                constraint_info.mapping_rows_eq >= constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_eq, 1)
        elif p_id == 'h':
            affine_map.mapping_rows = constraint_info.mapping_rows_ineq[
                constraint_info.mapping_rows_ineq >= constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_ineq, 1)
        else:
            raise ValueError(f'Unknown parameter name: {p_id}.')

        if p_id in ['A', 'b']:
            affine_map.indices = self.indices_constr[affine_map.mapping_rows]
        elif p_id in ['G', 'h']:
            affine_map.indices = self.indices_constr[affine_map.mapping_rows] - self.n_eq

        if p_id.isupper():
            affine_map.mapping = -param_prob.reduced_A.reduced_mat[affine_map.mapping_rows]

        return affine_map

    @property
    def stgs_names_enabled(self):
        return [name for name, enabled in zip(self.stgs_names, self.stgs_enabled) if enabled]

    @property
    def stgs_names_to_type(self):
        return {name: typ for name, typ, enabled in zip(self.stgs_names, self.stgs_types, self.stgs_enabled)
                if enabled}

    @property
    def stgs_names_to_default(self):
        return {name: typ for name, typ, enabled in zip(self.stgs_names, self.stgs_defaults, self.stgs_enabled)
                if enabled}

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        pass

    @abstractmethod
    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                      parameter_canon: ParameterCanon) -> None:
        pass

    @staticmethod
    def declare_workspace(self, f, prefix) -> None:
        pass

    @staticmethod
    def define_workspace(self, f, prefix) -> None:
        pass


class OSQPInterface(SolverInterface):
    solver_name = 'OSQP'
    canon_p_ids = ['P', 'q', 'd', 'A', 'l', 'u']
    canon_p_ids_constr_vec = ['l', 'u']
    sign_constr_vec = -1

    # header and source files
    header_files = ['osqp.h', 'types.h', 'workspace.h']
    cmake_headers = ['${osqp_headers}']
    cmake_sources = ['${osqp_src}']

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # workspace
    ws_allocated_in_solver_code = True
    result_ptrs = ResultPointerInfo(
        objective_value = 'workspace.info->obj_val',
        iterations = 'workspace.info->iter',
        status = 'workspace.info->status',
        primal_residual = 'workspace.info->pri_res',
        dual_residual = 'workspace.info->dua_res',
        primal_solution = 'xsolution', #'workspace.solution->x',
        dual_solution = '%ssolution' #'workspace.solution->%s'
    )

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = False

    # float and integer types
    numeric_types = {'float': 'c_float', 'int': 'c_int'}

    # solver settings
    stgs_statically_allocated = True
    stgs_set_function = {'name': 'osqp_update_%s', 'ptr_name': 'workspace'}
    stgs_reset_function = {'name': 'osqp_set_default_settings', 'ptr_name': 'settings'}
    stgs_names = ['rho', 'max_iter', 'eps_abs', 'eps_rel', 'eps_prim_inf', 'eps_dual_inf',
                      'alpha', 'scaled_termination', 'check_termination', 'warm_start',
                      'verbose', 'polish', 'polish_refine_iter', 'delta']
    stgs_types = ['cpg_float', 'cpg_int', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float',
                      'cpg_int', 'cpg_int', 'cpg_int', 'cpg_int', 'cpg_int', 'cpg_int', 'cpg_float']
    stgs_enabled = [True, True, True, True, True, True, True, True, True, True,
                        False, False, False, False]
    stgs_defaults = []

    # docu
    docu = 'https://osqp.org/docs/codegen/python.html'

    def __init__(self, data, p_prob, enable_settings):
        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        indices_obj, indptr_obj, shape_obj = p_prob.reduced_P.problem_data_index
        indices_constr, indptr_constr, shape_constr = p_prob.reduced_A.problem_data_index

        canon_constants = {}

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                      parameter_canon: ParameterCanon) -> None:
        import osqp

        # OSQP codegen
        osqp_obj = osqp.OSQP()
        osqp_obj.setup(P=parameter_canon.p_csc['P'], q=parameter_canon.p['q'],
                       A=parameter_canon.p_csc['A'], l=parameter_canon.p['l'],
                       u=parameter_canon.p['u'])
        if system() == 'Windows':
            cmake_generator = 'MinGW Makefiles'
        elif system() == 'Linux' or system() == 'Darwin':
            cmake_generator = 'Unix Makefiles'
        else:
            raise ValueError(f'Unsupported OS {system()}.')

        osqp_obj.codegen(os.path.join(code_dir, 'c', 'solver_code'), project_type=cmake_generator,
                         parameters='matrices', force_rewrite=True)

        # copy license files
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'osqp-python', 'LICENSE'),
                        os.path.join(solver_code_dir, 'LICENSE'))
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'LICENSE'), code_dir)

        # modify for extra settings
        if 'verbose' in self.enable_settings:
            replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                [('message(STATUS "Disabling printing for embedded")', 'message(STATUS "Not disabling printing for embedded by user request")'),
                 ('set(PRINTING OFF)', '')])
            replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'constants.h'),
                [('# ifdef __cplusplus\n}', '#  define VERBOSE (1)\n\n# ifdef __cplusplus\n}')])
            replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'types.h'),
                [('} OSQPInfo;', '  c_int status_polish;\n} OSQPInfo;'),
                 ('} OSQPSettings;', '  c_int polish;\n  c_int verbose;\n} OSQPSettings;'),
                 ('# ifndef EMBEDDED\n  c_int nthreads; ///< number of threads active\n# endif // ifndef EMBEDDED', '  c_int nthreads;')])
            replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'osqp.h'),
                [('# ifdef __cplusplus\n}', 'c_int osqp_update_verbose(OSQPWorkspace *work, c_int verbose_new);\n\n# ifdef __cplusplus\n}')])
            replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'src', 'osqp', 'util.c'),
                [('// Print Settings', '/* Print Settings'),
                 ('LINSYS_SOLVER_NAME[settings->linsys_solver]);', 'LINSYS_SOLVER_NAME[settings->linsys_solver]);*/')])
            replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'src', 'osqp', 'osqp.c'),
                [('void osqp_set_default_settings(OSQPSettings *settings) {', 'void osqp_set_default_settings(OSQPSettings *settings) {\n  settings->verbose = VERBOSE;'),
                 ('c_int osqp_update_verbose', '#endif // EMBEDDED\n\nc_int osqp_update_verbose'),
                 ('verbose = verbose_new;\n\n  return 0;\n}\n\n#endif // EMBEDDED', 'verbose = verbose_new;\n\n  return 0;\n}')])

    def get_affine_map(self, p_id, param_prob, constraint_info: ConstraintInfo) -> AffineMap:
        affine_map = AffineMap()

        if p_id == 'P':
            affine_map.mapping = param_prob.reduced_P.reduced_mat
            affine_map.indices = self.indices_obj
            affine_map.shape = (self.n_var, self.n_var)
        elif p_id == 'q':
            affine_map.mapping = param_prob.q[:-1]
        elif p_id == 'd':
            affine_map.mapping = param_prob.q[-1]
        elif p_id == 'A':
            affine_map.mapping = param_prob.reduced_A.reduced_mat[
                                 :constraint_info.n_data_constr_mat]
            affine_map.indices = self.indices_constr[:constraint_info.n_data_constr_mat]
            affine_map.shape = (self.n_eq + self.n_ineq, self.n_var)
        elif p_id == 'l':
            mapping_rows_eq = np.nonzero(self.indices_constr < self.n_eq)[0]
            affine_map.mapping_rows = mapping_rows_eq[
                mapping_rows_eq >= constraint_info.n_data_constr_mat]  # mapping to the finite part of l
            affine_map.indices = self.indices_constr[affine_map.mapping_rows]
            affine_map.shape = (self.n_eq, 1)
        elif p_id == 'u':
            affine_map.mapping_rows = np.arange(constraint_info.n_data_constr_mat,
                                                constraint_info.n_data_constr)
            affine_map.indices = self.indices_constr[affine_map.mapping_rows]
            affine_map.shape = (self.n_eq + self.n_ineq, 1)
        else:
            raise ValueError(f'Unknown OSQP parameter name {p_id}.')

        return affine_map


class SCSInterface(SolverInterface):
    solver_name = 'SCS'
    canon_p_ids = ['c', 'd', 'A', 'b']
    canon_p_ids_constr_vec = ['b']
    sign_constr_vec = 1

    # header files
    header_files = ['scs.h']
    cmake_headers = [
        '${${PROJECT_NAME}_HDR}',
        '${DIRSRC}/private.h',
        '${${PROJECT_NAME}_LDL_EXTERNAL_HDR}',
        '${${PROJECT_NAME}_AMD_EXTERNAL_HDR}',
        ]
    cmake_sources = [
        '${${PROJECT_NAME}_SRC}',
        '${DIRSRC}/private.c',
        '${EXTERNAL}/qdldl/qdldl.c',
        '${${PROJECT_NAME}_AMD_EXTERNAL_SRC}'
        ]

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # workspace
    ws_allocated_in_solver_code = False
    result_ptrs = ResultPointerInfo(
        objective_value = 'Scs_Info.pobj',
        iterations = 'Scs_Info.iter',
        status = 'Scs_Info.status',
        primal_residual = 'Scs_Info.res_pri',
        dual_residual = 'Scs_Info.res_dual',
        primal_solution = 'scs_x',
        dual_solution = 'scs_%s'
    )

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = False

    # float and integer types
    numeric_types = {'float': 'scs_float', 'int': 'scs_int'}

    # solver settings
    stgs_statically_allocated = False
    stgs_set_function = None
    stgs_reset_function = {'name': 'scs_set_default_settings', 'ptr_name': None} # set 'ptr_name' to None if stgs not statically allocated in solver code
    stgs_names = ['normalize', 'scale', 'adaptive_scale', 'rho_x', 'max_iters', 'eps_abs',
                      'eps_rel',
                      'eps_infeas', 'alpha', 'time_limit_secs', 'verbose', 'warm_start',
                      'acceleration_lookback',
                      'acceleration_interval', 'write_data_filename', 'log_csv_filename']
    stgs_types = ['cpg_int', 'cpg_float', 'cpg_int', 'cpg_float', 'cpg_int', 'cpg_float', 'cpg_float',
                      'cpg_float', 'cpg_float',
                      'cpg_float', 'cpg_int', 'cpg_int', 'cpg_int', 'cpg_int', 'const char*', 'const char*']
    stgs_enabled = [True, True, True, True, True, True, True, True, True, True, True, True,
                        True, True, True, True]
    stgs_defaults = ['1', '0.1', '1', '1e-6', '1e5', '1e-4', '1e-4', '1e-7', '1.5', '0', '0',
                         '0', '0', '1',
                         'SCS_NULL', 'SCS_NULL']

    # docu
    docu = 'https://www.cvxgrp.org/scs/api/c.html'

    def __init__(self, data, p_prob, enable_settings):
        n_var = p_prob.x.size
        n_eq = data['A'].shape[0]
        n_ineq = 0

        indices_obj, indptr_obj, shape_obj = None, None, None

        indices_constr, indptr_constr, shape_constr = p_prob.reduced_A.problem_data_index

        canon_constants = {'n': n_var, 'm': n_eq, 'z': p_prob.cone_dims.zero,
                           'l': p_prob.cone_dims.nonneg,
                           'q': np.array(p_prob.cone_dims.soc),
                           'qsize': len(p_prob.cone_dims.soc)}

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0 or len(cone_dims.psd) > 0 or len(cone_dims.p3d) > 0:
            raise ValueError(
                'Code generation with SCS and exponential, positive semidefinite, or power cones '
                'is not supported yet.')

    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                      parameter_canon: ParameterCanon) -> None:

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'linsys', 'cmake']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'scs', dtc),
                            os.path.join(solver_code_dir, dtc))
        files_to_copy = ['scs.mk', 'CMakeLists.txt', 'LICENSE.txt']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'scs', fl),
                            os.path.join(solver_code_dir, fl))
        shutil.copy(os.path.join(cvxpygen_directory, 'template', 'LICENSE'), code_dir)

        # disable BLAS and LAPACK
        with open(os.path.join(code_dir, 'c', 'solver_code', 'scs.mk'), 'r') as f:
            scs_mk_data = f.read()
        scs_mk_data = scs_mk_data.replace('USE_LAPACK = 1', 'USE_LAPACK = 0')
        with open(os.path.join(code_dir, 'c', 'solver_code', 'scs.mk'), 'w') as f:
            f.write(scs_mk_data)

        # modify CMakeLists.txt
        with open(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'), 'r') as f:
            cmake_data = f.read()
        cmake_data = cmake_data.replace(' include/', ' ${CMAKE_CURRENT_SOURCE_DIR}/include/')
        cmake_data = cmake_data.replace(' src/', ' ${CMAKE_CURRENT_SOURCE_DIR}/src/')
        cmake_data = cmake_data.replace(' ${LINSYS}/', ' ${CMAKE_CURRENT_SOURCE_DIR}/${LINSYS}/')
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

    def declare_workspace(self, f, prefix) -> None:
        f.write('\n// SCS matrix A\n')
        write_struct_prot(f, '%sscs_A' % prefix, 'ScsMatrix')
        f.write('\n// Struct containing SCS data\n')
        write_struct_prot(f, '%sScs_D' % prefix, 'ScsData')
        if self.canon_constants['qsize'] > 0:
            f.write('\n// SCS array of SOC dimensions\n')
            write_vec_prot(f, self.canon_constants['q'], '%sscs_q' % prefix, 'cpg_int')
        f.write('\n// Struct containing SCS cone data\n')
        write_struct_prot(f, '%sScs_K' % prefix, 'ScsCone')
        f.write('\n// Struct containing SCS settings\n')
        write_struct_prot(f, prefix + 'Canon_Settings', 'ScsSettings')
        f.write('\n// SCS solution\n')
        write_vec_prot(f, np.zeros(self.canon_constants['n']), '%sscs_x' % prefix, 'cpg_float')
        write_vec_prot(f, np.zeros(self.canon_constants['m']), '%sscs_y' % prefix, 'cpg_float')
        write_vec_prot(f, np.zeros(self.canon_constants['m']), '%sscs_s' % prefix, 'cpg_float')
        f.write('\n// Struct containing SCS solution\n')
        write_struct_prot(f, '%sScs_Sol' % prefix, 'ScsSolution')
        f.write('\n// Struct containing SCS information\n')
        write_struct_prot(f, '%sScs_Info' % prefix, 'ScsInfo')
        f.write('\n// Pointer to struct containing SCS workspace\n')
        write_struct_prot(f, '%sScs_Work' % prefix, 'ScsWork*')

    def define_workspace(self, f, prefix) -> None:
        f.write('\n// SCS matrix A\n')
        scs_A_fields = ['x', 'i', 'p', 'm', 'n']
        scs_A_casts = ['(cpg_float *) ', '(cpg_int *) ', '(cpg_int *) ', '', '']
        scs_A_values = ['&%scanon_A_x' % prefix, '&%scanon_A_i' % prefix,
                        '&%scanon_A_p' % prefix, str(self.canon_constants['m']),
                        str(self.canon_constants['n'])]
        write_struct_def(f, scs_A_fields, scs_A_casts, scs_A_values, '%sScs_A' % prefix, 'ScsMatrix')

        f.write('\n// Struct containing SCS data\n')
        scs_d_fields = ['m', 'n', 'A', 'P', 'b', 'c']
        scs_d_casts = ['', '', '', '', '(cpg_float *) ', '(cpg_float *) ']
        scs_d_values = [str(self.canon_constants['m']), str(self.canon_constants['n']),
                        '&%sScs_A' % prefix, 'SCS_NULL', '&%scanon_b' % prefix, '&%scanon_c' % prefix]
        write_struct_def(f, scs_d_fields, scs_d_casts, scs_d_values, '%sScs_D' % prefix, 'ScsData')

        if self.canon_constants['qsize'] > 0:
            f.write('\n// SCS array of SOC dimensions\n')
            write_vec_def(f, self.canon_constants['q'], '%sscs_q' % prefix, 'cpg_int')
            k_field_q_str = '&%sscs_q' % prefix
        else:
            k_field_q_str = 'SCS_NULL'

        f.write('\n// Struct containing SCS cone data\n')
        scs_k_fields = ['z', 'l', 'bu', 'bl', 'bsize', 'q', 'qsize', 's', 'ssize', 'ep', 'ed', 'p', 'psize']
        scs_k_casts = ['', '', '(cpg_float *) ', '(cpg_float *) ', '', '(cpg_int *) ', '', '(cpg_int *) ', '', '', '',
                       '(cpg_float *) ', '']
        scs_k_values = [str(self.canon_constants['z']), str(self.canon_constants['l']), 'SCS_NULL', 'SCS_NULL', '0',
                        k_field_q_str, str(self.canon_constants['qsize']), 'SCS_NULL', '0', '0', '0', 'SCS_NULL', '0']
        write_struct_def(f, scs_k_fields, scs_k_casts, scs_k_values, '%sScs_K' % prefix, 'ScsCone')

        f.write('\n// Struct containing SCS settings\n')
        scs_stgs_fields = list(self.stgs_names_to_default.keys())
        scs_stgs_casts = ['']*len(scs_stgs_fields)
        scs_stgs_values = list(self.stgs_names_to_default.values())
        write_struct_def(f, scs_stgs_fields, scs_stgs_casts, scs_stgs_values, prefix + 'Canon_Settings', 'ScsSettings')

        f.write('\n// SCS solution\n')
        write_vec_def(f, np.zeros(self.canon_constants['n']), '%sscs_x' % prefix, 'cpg_float')
        write_vec_def(f, np.zeros(self.canon_constants['m']), '%sscs_y' % prefix, 'cpg_float')
        write_vec_def(f, np.zeros(self.canon_constants['m']), '%sscs_s' % prefix, 'cpg_float')

        f.write('\n// Struct containing SCS solution\n')
        scs_sol_fields = ['x', 'y', 's']
        scs_sol_casts = ['(cpg_float *) ', '(cpg_float *) ', '(cpg_float *) ']
        scs_sol_values = ['&%sscs_x' % prefix, '&%sscs_y' % prefix, '&%sscs_s' % prefix]
        write_struct_def(f, scs_sol_fields, scs_sol_casts, scs_sol_values, '%sScs_Sol' % prefix, 'ScsSolution')

        f.write('\n// Struct containing SCS information\n')
        scs_info_fields = ['iter', 'status', 'status_val', 'scale_updates', 'pobj', 'dobj', 'res_pri', 'res_dual',
                           'gap', 'res_infeas', 'res_unbdd_a', 'res_unbdd_p', 'comp_slack', 'setup_time', 'solve_time',
                           'scale', 'rejected_accel_steps', 'accepted_accel_steps', 'lin_sys_time', 'cone_time',
                           'accel_time']
        scs_info_casts = ['']*len(scs_info_fields)
        scs_info_values = ['0', '"unknown"', '0', '0', '0', '0', '99', '99', '99', '99', '99', '99', '99', '0', '0',
                           '1', '0', '0', '0', '0', '0']
        write_struct_def(f, scs_info_fields, scs_info_casts, scs_info_values, '%sScs_Info' % prefix, 'ScsInfo')

        f.write('\n// Pointer to struct containing SCS workspace\n')
        f.write('ScsWork* %sScs_Work = 0;\n' % prefix)


class ECOSInterface(SolverInterface):
    solver_name = 'ECOS'
    canon_p_ids = ['c', 'd', 'A', 'b', 'G', 'h']
    canon_p_ids_constr_vec = ['b', 'h']
    sign_constr_vec = 1

    # header files
    header_files = ['ecos.h']
    cmake_headers = ['${ecos_headers}']
    cmake_sources = ['${ecos_sources}']

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = True

    # workspace
    ws_allocated_in_solver_code = False
    result_ptrs = ResultPointerInfo(
        objective_value = 'ecos_workspace->info->pcost',
        iterations = 'ecos_workspace->info->iter',
        status = 'ecos_flag',
        primal_residual = 'ecos_workspace->info->pres',
        dual_residual = 'ecos_workspace->info->dres',
        primal_solution = 'ecos_workspace->x',
        dual_solution = 'ecos_workspace->%s'
    )

    # solution vectors statically allocated
    sol_statically_allocated = False

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'double', 'int': 'int'}

    # solver settings
    stgs_statically_allocated = False
    stgs_set_function = None
    stgs_reset_function = None
    stgs_names = ['feastol', 'abstol', 'reltol', 'feastol_inacc', 'abstol_inacc',
                      'reltol_inacc', 'maxit']
    stgs_types = ['cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_float', 'cpg_int']
    stgs_enabled = [True, True, True, True, True, True, True]
    stgs_defaults = ['1e-8', '1e-8', '1e-8', '1e-4', '5e-5', '5e-5', '100']

    # docu
    docu = 'https://github.com/embotech/ecos/wiki/Usage-from-C'

    def __init__(self, data, p_prob, enable_settings):
        n_var = p_prob.x.size
        n_eq = p_prob.cone_dims.zero
        n_ineq = data['G'].shape[0]

        indices_obj, indptr_obj, shape_obj = None, None, None

        indices_constr, indptr_constr, shape_constr = p_prob.reduced_A.problem_data_index

        canon_constants = {'n': n_var, 'm': n_ineq, 'p': n_eq,
                           'l': p_prob.cone_dims.nonneg,
                           'n_cones': len(p_prob.cone_dims.soc),
                           'q': np.array(p_prob.cone_dims.soc),
                           'e': p_prob.cone_dims.exp}

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, indices_obj, indptr_obj, shape_obj,
                         indices_constr, indptr_constr, shape_constr, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0:
            raise ValueError(
                'Code generation with ECOS and exponential cones is not supported yet.')

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                      parameter_canon: ParameterCanon) -> None:

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'external', 'ecos_bb']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'ecos', dtc),
                            os.path.join(solver_code_dir, dtc))
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', 'CMakeLists.txt'),
                        os.path.join(solver_code_dir, 'CMakeLists.txt'))
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', 'COPYING'),
                        os.path.join(solver_code_dir, 'COPYING'))
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', 'COPYING'),
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
                                        indent + "os.path.join('c', 'solver_code', 'external', 'SuiteSparse_config'),\n" +
                                        indent + "os.path.join('c', 'solver_code', 'external', 'amd', 'include'),\n" +
                                        indent + "os.path.join('c', 'solver_code', 'external', 'ldl', 'include'),")
        setup_text = setup_text.replace("license='Apache 2.0'", "license='GPL 3.0'")
        with open(os.path.join(code_dir, 'setup.py'), 'w') as f:
            f.write(setup_text)

    def declare_workspace(self, f, prefix) -> None:
        f.write('\n// Struct containing solver settings\n')
        write_struct_prot(f, prefix + 'Canon_Settings', 'Canon_Settings_t')
        if self.canon_constants['n_cones'] > 0:
            f.write('\n// ECOS array of SOC dimensions\n')
            write_vec_prot(f, self.canon_constants['q'], '%secos_q' % prefix, 'cpg_int')
        f.write('\n// ECOS workspace\n')
        f.write('extern pwork* %secos_workspace;\n' % prefix)
        f.write('\n// ECOS exit flag\n')
        f.write('extern cpg_int %secos_flag;\n' % prefix)

    def define_workspace(self, f, prefix) -> None:
        f.write('\n// Struct containing solver settings\n')
        f.write('Canon_Settings_t %sCanon_Settings = {\n' % prefix)
        for name, default in self.stgs_names_to_default.items():
            f.write('.%s = %s,\n' % (name, default))
        f.write('};\n')
        if self.canon_constants['n_cones'] > 0:
            f.write('\n// ECOS array of SOC dimensions\n')
            write_vec_def(f, self.canon_constants['q'], '%secos_q' % prefix, 'cpg_int')
        f.write('\n// ECOS workspace\n')
        f.write('pwork* %secos_workspace = 0;\n' % prefix)
        f.write('\n// ECOS exit flag\n')
        f.write('cpg_int %secos_flag = -99;\n' % prefix)
