
import cvxpy as cp
from cvxpygen import cpg
import torch
from cvxpylayers.torch import CvxpyLayer as LayerTorch
import numpy as np
import time
import sys

if __name__ == "__main__":

    '''
    1. Generate Code
    '''

    # define CVXPY problem
    m, n = 3, 2
    x = cp.Variable(n, name='x')
    A = cp.Parameter((m, n), name='A', sparsity=((0, 0, 1), (0, 1, 1)))
    b = cp.Parameter(m, name='b')
    problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

    # assign parameter values and test-solve
    np.random.seed(0)
    A.value = np.zeros((m, n))
    A.value[0, 0] = 1 #np.random.randn()
    A.value[0, 1] = 1 #np.random.randn()
    A.value[1, 1] = 1 #np.random.randn()
    b.value = np.array([2., 3., 4.]) #np.random.randn(m)
    t0 = time.time()
    problem.solve(solver='OSQP')
    print('CVXPY\nSolve time: %.3f s' % ((time.time()-t0)))

    if False:
        # generate code
        cpg.generate_code(problem, code_dir='nonneg_LS', solver='OSQP')

        '''
        2. Solve & Compare
        '''

        # import extension module and register custom CVXPY solve method
        from nonneg_LS.cpg_solver import cpg_solve
        problem.register_solve('cpg', cpg_solve)

        # solve problem conventionally
        t0 = time.time()
        val = problem.solve(solver='OSQP')
        t1 = time.time()
        sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)

        # solve problem with C code via python wrapper
        t0 = time.time()
        val = problem.solve(method='cpg', updated_params=['A', 'b'])
        t1 = time.time()
        sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)
        
        A.value[0, 0] = -1
        A.value[0, 1] = -1
        A.value[1, 1] = -1

        # solve problem conventionally
        t0 = time.time()
        val = problem.solve(solver='OSQP')
        t1 = time.time()
        sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)

        # solve problem with C code via python wrapper
        t0 = time.time()
        val = problem.solve(method='CPG', updated_params=['A', 'b'], verbos=False)
        t1 = time.time()
        sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)
        
        print('done')


    if True:
        # generate code
        #cpg.generate_code(problem, code_dir='nonneg_LS', solver='OSQP', prefix='pre', gradient=True, wrapper=True)

        '''
        2. Solve & Compare
        '''

        # import extension module and register custom CVXPY solve method
        from nonneg_LS.cpg_solver import cpg_solve, cpg_gradient, cvxpylayers_solve, cvxpylayers_gradient
        problem.register_solve('cpg', cpg_solve)

        # solve problem conventionally
        t0 = time.time()
        val = problem.solve(solver='SCS', requires_grad=True)
        t1 = time.time()
        sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)
        
        x.gradient = np.array([0.1, 0.1])
        t0 = time.time()
        problem.backward()
        t1 = time.time()
        print('Gradient cvxpy:\n', A.gradient, '\n', b.gradient)
        print('Gradient time: %.3f ms' % (1000*(t1-t0)))
        
        # solve problem with C code via python wrapper
        t0 = time.time()
        val = problem.solve(method='cpg', updated_params=['A', 'b'])
        t1 = time.time()
        sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)
        
        x.gradient = np.array([0.1, 0.1])
        t0 = time.time()
        cpg_gradient(problem)
        t1 = time.time()
        print('Gradient cvxpygen:\n', A.gradient, '\n', b.gradient)
        print('Gradient time: %.3f ms' % (1000*(t1-t0)))
        print('done')
        
        layer_torch = LayerTorch(problem, parameters=[A, b], variables=[x], custom_method=(cvxpylayers_solve, cvxpylayers_gradient))
        A_tch = torch.tensor(A.value, requires_grad=True)
        b_tch = torch.tensor(b.value, requires_grad=True)
        
        t0 = time.time()
        sol_torch, = layer_torch(A_tch, b_tch, solver_args={'problem': problem, 'updated_params': ['A', 'b']})
        t1 = time.time()
        sys.stdout.write('\nCVXPYLayers with CVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(sol_torch.detach().numpy()))
        
        # given a gradient [0.1, 0.1] in sol_torch, compute the gradient in A and b
        sum_torch = 0.1 * sol_torch.sum()
        t0 = time.time()
        sum_torch.backward()
        t1 = time.time()
        print('Gradient cvxpylayers:\n', A_tch.grad, '\n', b_tch.grad)
        print('Gradient time: %.3f ms' % (1000*(t1-t0)))
        print('done')
        
    if False:
        
        ### update parmeters
        
        #b.value = np.array([-2, -3, 4])
        A.value[0, 0] = -1
        A.value[0, 1] = -1
        A.value[1, 1] = -1
        
        # solve problem conventionally
        t0 = time.time()
        val = problem.solve(solver='SCS', requires_grad=True)
        t1 = time.time()
        sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)
        
        x.gradient = np.array([0.1, 0.1])
        t0 = time.time()
        problem.backward()
        t1 = time.time()
        print('Gradient cvxpy:\n', A.gradient, '\n', b.gradient)
        print('Gradient time: %.3f ms' % (1000*(t1-t0)))
        
        # solve problem with C code via python wrapper
        t0 = time.time()
        val = problem.solve(method='cpg', updated_params=['A', 'b'])
        t1 = time.time()
        sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)
        
        x.gradient = np.array([0.1, 0.1])
        t0 = time.time()
        cpg_gradient(problem)
        t1 = time.time()
        print('Gradient cvxpygen:\n', A.gradient, '\n', b.gradient)
        print('Gradient time: %.3f ms' % (1000*(t1-t0)))
        print('done')
        
        
    #if False:
        
        A.value[0, 0] = 1
        A.value[0, 1] = 1
        A.value[1, 1] = 1
        
        # solve problem conventionally
        t0 = time.time()
        val = problem.solve(solver='SCS', requires_grad=True)
        t1 = time.time()
        sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)
        
        x.gradient = np.array([0.1, 0.1])
        t0 = time.time()
        problem.backward()
        t1 = time.time()
        print('Gradient cvxpy:\n', A.gradient, '\n', b.gradient)
        print('Gradient time: %.3f ms' % (1000*(t1-t0)))
        
        # solve problem with C code via python wrapper
        t0 = time.time()
        val = problem.solve(method='cpg', updated_params=['A', 'b'])
        t1 = time.time()
        sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
        sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
        sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
        sys.stdout.write('Objective function value: %.6f\n' % val)
        
        x.gradient = np.array([0.1, 0.1])
        t0 = time.time()
        cpg_gradient(problem)
        t1 = time.time()
        print('Gradient cvxpygen:\n', A.gradient, '\n', b.gradient)
        print('Gradient time: %.3f ms' % (1000*(t1-t0)))
        print('done')
