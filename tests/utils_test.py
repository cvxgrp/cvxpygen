
import numpy as np


def get_dual_vec(prob):
    dual_values = []
    for constr in prob.constraints:
        if constr.args[0].size == 1:
            dual_values.append(np.atleast_1d(constr.dual_value))
        else:
            dual_values.append(constr.dual_value.flatten())
    return np.concatenate(dual_values)


def check(prob, solver, name, func_get_primal_vec):

    if solver == 'OSQP':
        val_py = prob.solve(solver='OSQP', eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polish=False,
                            adaptive_rho_interval=int(1e6), warm_start=False)
    elif solver == 'SCS':
        val_py = prob.solve(solver='SCS', warm_start=False, verbose=False)
    else:
        val_py = prob.solve(solver='ECOS')
    prim_py = func_get_primal_vec(prob, name)
    dual_py = get_dual_vec(prob)
    if solver == 'OSQP':
        val_cg = prob.solve(method='CPG', warm_start=False)
    elif solver == 'SCS':
        val_cg = prob.solve(method='CPG', warm_start=False, verbose=False)
    else:
        val_cg = prob.solve(method='CPG')
    prim_cg = func_get_primal_vec(prob, name)
    dual_cg = get_dual_vec(prob)
    prim_py_norm = np.linalg.norm(prim_py, 2)
    dual_py_norm = np.linalg.norm(dual_py, 2)

    return val_py, prim_py, dual_py, val_cg, prim_cg, dual_cg, prim_py_norm, dual_py_norm
