"""
Copyright 2025 Maximilian Schaller
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import shutil
import numpy as np
import cvxpy as cp
from scipy import sparse
from pdaqp import MPQP
from itertools import product
from cvxpy.utilities import key_utils as ku


def offline_solve_and_codegen_explicit(problem, canon, solver_code_dir, solver_opts, explicit_flag):
    
    # set maximum number of regions and maximum number of floating point numbers
    max_regions = solver_opts.get('max_regions', 500) if solver_opts else 500
    max_floats = solver_opts.get('max_floats', 1e6) if solver_opts else 1e6
    
    # set precision for storing explicit solution
    c_float_store = '_Float16' if solver_opts and (solver_opts.get('fp16', False) or solver_opts.get('FP16', False)) else 'float'
    
    # check that P and A are constants
    for p_id in ['P', 'A']:
        if canon.parameter_canon.p_id_to_changes[p_id]:
            raise ValueError(f'Explicit mode: Matrices are not constant!')
        
    A = canon.parameter_canon.p['A'].toarray()
    m, n = A.shape
    
    H = canon.parameter_canon.p['P'].toarray() + 1e-6 * np.eye(n)

    f = np.zeros_like(canon.parameter_canon.p['q'])
    F = np.hstack([np.eye(n), np.zeros((n, m))])
    
    b = np.zeros_like(canon.parameter_canon.p['u'])
    B = np.hstack([np.zeros((m, n)), np.eye(m)])
    
    # remove any zero rows from A and corresponding rows of b and B
    A_mask = np.any(A != 0, axis=1)
    A = A[A_mask, :]
    b = b[A_mask]
    B = B[A_mask, :]
    
    # extract bounds on theta (thmin, thmax) and user-defined params (lower, upper)
    thmin, thmax, lower, upper = get_parameter_delta_bounds(problem, canon)
    
    # eliminate theta components that are fixed
    th_mask = thmin != thmax
    f += F[:, ~th_mask] @ thmin[~th_mask]
    b += B[:, ~th_mask] @ thmin[~th_mask]
    F = F[:, th_mask]
    B = B[:, th_mask]
    thmin, thmax = thmin[th_mask], thmax[th_mask]
    
    # eliminate theta components that are multiplied with zero
    th_mask_multiplied_nonzero = np.any(F != 0, axis=0) | np.any(B != 0, axis=0)
    F = F[:, th_mask_multiplied_nonzero]
    B = B[:, th_mask_multiplied_nonzero]
    thmin, thmax = thmin[th_mask_multiplied_nonzero], thmax[th_mask_multiplied_nonzero]
    
    th_mask_resulting = th_mask.copy()
    th_mask_resulting[th_mask] = th_mask_multiplied_nonzero

    # extract indices of equality constraints
    eq_m = canon.parameter_canon.p_id_to_size['l']
    eq_inds = np.arange(eq_m)[A_mask[:eq_m]]
    
    # print dimensions info
    sys.stdout.write(f'Generating explicit solver for (canonicalized) parametric QP with\n'
                     f'{len(f)} variables,\n'
                     f'{len(eq_inds)} linear equality constraints,\n'
                     f'{len(b)-len(eq_inds)} linear inequality constraints, and\n'
                     f'{len(thmin)} parameters ...\n')

    # extract variables to store
    all_names = [name for name in canon.prim_variable_info.name_to_offset]
    stored_vars = solver_opts.get('stored_vars', None) if solver_opts else None
    names_and_inds = []
    if stored_vars is not None:
        for s in stored_vars:
            v = s.variables()[0]
            sl  =  s.get_data()
            if sl is None: # Variable => store all
                names_and_inds.append((v.name(), None))
            else:
                sl = [ku.format_slice(key,sh,len(v.shape)) if not ku.is_special_slice(key) else key
                              for key,sh in zip(sl[0],v.shape)]
                ranges = [np.arange(s.start,s.stop,s.step) if type(s) == slice else s for s in sl]
                inds = [np.ravel_multi_index(id, v.shape,order='F') for id in product(*ranges)]
                names_and_inds.append((v.name(), inds))
    else: # by default, store all variables
        for name in all_names: names_and_inds.append((name,None))

    shift=0
    out_inds = np.empty(0, dtype=int)
    added_names = []
    for name,inds in names_and_inds:
        offset = canon.prim_variable_info.name_to_offset.get(name,None)
        if offset is not None:
            size = canon.prim_variable_info.name_to_size[name]
            inds = np.array(inds,dtype='int') if inds else np.arange(0,size)

            out_inds = np.append(out_inds, offset+inds)

            canon.prim_variable_info.name_to_offset[name] = shift
            if size == 1:
                canon.prim_variable_info.name_to_indices[name] = np.array([shift])
            else:
                canon.prim_variable_info.name_to_indices[name] = np.full(size,-1)
                canon.prim_variable_info.name_to_indices[name][inds] = np.arange(0,len(inds))
            added_names.append(name)
            shift+=len(inds)
        #else:
            # XXX wanted to store variable that does not exist

    # Remove non-stored variables from canonicalization
    for i,name in enumerate(all_names):
        if name not in added_names:
            del canon.prim_variable_info.name_to_offset[name]
            del canon.prim_variable_info.name_to_indices[name]
            del canon.prim_variable_info.name_to_size[name]
            del canon.prim_variable_info.name_to_shape[name]
            del canon.prim_variable_info.name_to_init[name]
            del canon.prim_variable_info.name_to_sym[name]
            del canon.prim_variable_info.sizes[i]
            del canon.prim_variable_info.sym[i]
            canon.prim_variable_info.reduced = True

    # construct explicit solution
    mpqp = MPQP(H, f, F, A, b, B, thmin, thmax, eq_inds=eq_inds, out_inds=out_inds)
    mpqp.solve(settings={'region_limit': max_regions, 'store_dual': (explicit_flag==2)})
    if str(mpqp.solution_info.status) != 'Solved':
        raise Exception(f'Could not compute explicit solution: {mpqp.solution_info.status}')

    codegen_status = mpqp.codegen(dir=solver_code_dir, max_reals=max_floats, dual=(explicit_flag==2), c_float_store=c_float_store)
    if codegen_status < 0:
        raise Exception('Could not generate explicit solver. Consider increasing max_reals in solver_opts.')
    
    include_dir = os.path.join(solver_code_dir, 'include')
    src_dir = os.path.join(solver_code_dir, 'src')
    os.makedirs(include_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    shutil.move(os.path.join(solver_code_dir, 'pdaqp.c'), os.path.join(src_dir, 'pdaqp.c'))
    shutil.move(os.path.join(solver_code_dir, 'pdaqp.h'), os.path.join(include_dir, 'pdaqp.h'))
            
    # create solver_code_dir/CMakeLists.txt
    with open(os.path.join(solver_code_dir, 'CMakeLists.txt'), 'w') as fl:
        fl.write('list (APPEND pdaqp_src ${CMAKE_CURRENT_SOURCE_DIR}/src/pdaqp.c)\n')
        fl.write('list (APPEND pdaqp_head ${CMAKE_CURRENT_SOURCE_DIR}/include/pdaqp.h)\n')
        fl.write('set (solver_src ${pdaqp_src} PARENT_SCOPE)\n')
        fl.write('set (solver_head ${pdaqp_head} PARENT_SCOPE)\n')
        
    canon.parameter_canon.th_mask = th_mask_resulting
    canon.parameter_canon.n_param_reduced = np.count_nonzero(th_mask_resulting)
    canon.parameter_canon.n_dual_reduced = len(b)
    canon.parameter_info.lower = lower
    canon.parameter_info.upper = upper
    
    
def get_parameter_delta_bounds(problem, canon):
    
    parameter_info = canon.parameter_info
    parameter_canon = canon.parameter_canon
    
    # extract bounds on user-defined parameter deltas
    lower = -1e30 * np.ones_like(parameter_info.flat_usp)
    upper = 1e30 * np.ones_like(parameter_info.flat_usp)
    lower[-1], upper[-1] = 1, 1
    
    for i, constraint in enumerate(problem.constraints):
        
        # consider pure parameter constraints
        if not constraint.variables() and constraint.parameters():
            
            lhs, rhs = constraint.args
            
            # consider simple bounds
            if isinstance(lhs, cp.Parameter) and isinstance(rhs, cp.Constant):
                col = parameter_info.id_to_col[lhs.id]
                upper[col:col + lhs.size] = rhs.value
            elif isinstance(lhs, cp.Constant) and isinstance(rhs, cp.Parameter):
                col = parameter_info.id_to_col[rhs.id]
                lower[col:col + rhs.size] = lhs.value
            else:
                raise ValueError('Explicit mode: Parameter constraints must be simple bounds!')
            
            # remove dual variables corresponding to parameter constraints
            canon.dual_variable_info.name_to_init.pop(f'd{i}')
            canon.dual_variable_info.name_to_vec.pop(f'd{i}')
            canon.dual_variable_info.name_to_offset.pop(f'd{i}')
            canon.dual_variable_info.name_to_indices.pop(f'd{i}')
            canon.dual_variable_info.name_to_size.pop(f'd{i}')
            canon.dual_variable_info.name_to_shape.pop(f'd{i}')
            canon.dual_variable_info.sizes[i] = -1
    
    canon.dual_variable_info.sizes = [s for s in canon.dual_variable_info.sizes if s != -1]
    
    # map to Delta (q, u)
    id_to_mapping = parameter_canon.p_id_to_mapping
    C_qu = sparse.vstack([id_to_mapping['q'], id_to_mapping['u']])
    lower_mapped, upper_mapped = C_qu @ lower, C_qu @ upper
    
    return np.minimum(lower_mapped, upper_mapped), np.maximum(lower_mapped, upper_mapped), lower, upper
