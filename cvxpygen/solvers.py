import os
import shutil
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from platform import system

import numpy as np

from cvxpygen import utils
from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, ConstraintInfo, AffineMap, \
    ParameterCanon


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
    def settings_names(self):
        pass

    @property
    @abstractmethod
    def settings_types(self):
        pass

    @property
    @abstractmethod
    def settings_defaults(self):
        pass

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return any(variable_info.sym) or any([s == 1 for s in variable_info.sizes])

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return any([s == 1 for s in dual_variable_info.sizes])

    def configure_settings(self) -> None:
        for i, s in enumerate(self.settings_names):
            if s in self.enable_settings:
                self.settings_enabled[i] = True
        for s in set(self.enable_settings)-set(self.settings_names):
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
    def settings_names_enabled(self):
        return [name for name, enabled in zip(self.settings_names, self.settings_enabled) if enabled]

    @property
    def settings_names_to_type(self):
        return {name: typ for name, typ, enabled in zip(self.settings_names, self.settings_types, self.settings_enabled)
                if enabled}

    @property
    def settings_names_to_default(self):
        return {name: typ for name, typ, enabled in zip(self.settings_names, self.settings_defaults, self.settings_enabled)
                if enabled}

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        pass

    @abstractmethod
    def generate_code(self, code_dir, solver_code_dir, cvxpygen_directory,
                      parameter_canon: ParameterCanon) -> None:
        pass


class OSQPInterface(SolverInterface):
    solver_name = 'OSQP'
    canon_p_ids = ['P', 'q', 'd', 'A', 'l', 'u']
    canon_p_ids_constr_vec = ['l', 'u']
    sign_constr_vec = -1

    # header files
    header_files = ['osqp.h', 'types.h', 'workspace.h']

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = False

    # solver settings
    settings_names = ['rho', 'max_iter', 'eps_abs', 'eps_rel', 'eps_prim_inf', 'eps_dual_inf',
                      'alpha', 'scaled_termination', 'check_termination', 'warm_start',
                      'verbose', 'polish', 'polish_refine_iter', 'delta']
    settings_types = ['c_float', 'c_int', 'c_float', 'c_float', 'c_float', 'c_float', 'c_float',
                      'c_int', 'c_int', 'c_int', 'c_int', 'c_int', 'c_int', 'c_float']
    settings_enabled = [True, True, True, True, True, True, True, True, True, True,
                        False, False, False, False]
    settings_defaults = []

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
            utils.replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                [('message(STATUS "Disabling printing for embedded")', 'message(STATUS "Not disabling printing for embedded by user request")'),
                 ('set(PRINTING OFF)', '')])
            utils.replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'constants.h'),
                [('# ifdef __cplusplus\n}', '#  define VERBOSE (1)\n\n# ifdef __cplusplus\n}')])
            utils.replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'types.h'),
                [('} OSQPInfo;', '  c_int status_polish;\n} OSQPInfo;'),
                 ('} OSQPSettings;', '  c_int polish;\n  c_int verbose;\n} OSQPSettings;'),
                 ('# ifndef EMBEDDED\n  c_int nthreads; ///< number of threads active\n# endif // ifndef EMBEDDED', '  c_int nthreads;')])
            utils.replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'osqp.h'),
                [('# ifdef __cplusplus\n}', 'c_int osqp_update_verbose(OSQPWorkspace *work, c_int verbose_new);\n\n# ifdef __cplusplus\n}')])
            utils.replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'src', 'osqp', 'util.c'),
                [('// Print Settings', '/* Print Settings'),
                 ('LINSYS_SOLVER_NAME[settings->linsys_solver]);', 'LINSYS_SOLVER_NAME[settings->linsys_solver]);*/')])
            utils.replace_in_file(os.path.join(code_dir, 'c', 'solver_code', 'src', 'osqp', 'osqp.c'),
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

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = False

    # solver settings
    settings_names = ['normalize', 'scale', 'adaptive_scale', 'rho_x', 'max_iters', 'eps_abs',
                      'eps_rel',
                      'eps_infeas', 'alpha', 'time_limit_secs', 'verbose', 'warm_start',
                      'acceleration_lookback',
                      'acceleration_interval', 'write_data_filename', 'log_csv_filename']
    settings_types = ['c_int', 'c_float', 'c_int', 'c_float', 'c_int', 'c_float', 'c_float',
                      'c_float', 'c_float',
                      'c_float', 'c_int', 'c_int', 'c_int', 'c_int', 'const char*', 'const char*']
    settings_enabled = [True, True, True, True, True, True, True, True, True, True, True, True,
                        True, True, True, True]
    settings_defaults = ['1', '0.1', '1', '1e-6', '1e5', '1e-4', '1e-4', '1e-7', '1.5', '0', '0',
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


class ECOSInterface(SolverInterface):
    solver_name = 'ECOS'
    canon_p_ids = ['c', 'd', 'A', 'b', 'G', 'h']
    canon_p_ids_constr_vec = ['b', 'h']
    sign_constr_vec = 1

    # header files
    header_files = ['ecos.h']

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = True

    # solution vectors statically allocated
    sol_statically_allocated = False

    # solver status as integer vs. string
    status_is_int = True

    # solver settings
    settings_names = ['feastol', 'abstol', 'reltol', 'feastol_inacc', 'abstol_inacc',
                      'reltol_inacc', 'maxit']
    settings_types = ['c_float', 'c_float', 'c_float', 'c_float', 'c_float', 'c_float', 'c_int']
    settings_enabled = [True, True, True, True, True, True, True]
    settings_defaults = ['1e-8', '1e-8', '1e-8', '1e-4', '5e-5', '5e-5', '100']

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
