
import cvxpy as cp
from cvxpygen import cpg
from cvxpylayers.torch import CvxpyLayer
import torch
import numpy as np
import numpy.linalg as la
import scipy as sp
import control
import matplotlib.pyplot as plt


# problem

n, m = 6, 3

u = cp.Variable(m, name='u')
g = cp.Parameter(n, name='g')
H = cp.Parameter((n, m), name='H')

objective = cp.Minimize(cp.sum_squares(g + H @ u) + cp.sum_squares(u))
constraints = [cp.abs(u) <= 1]

problem = cp.Problem(objective, constraints)


# dynamics

sim_length = 1000

np.random.seed(0)
A = np.diag(np.random.rand(n) * 0.01 + 0.99)
A[0, 0] = 1.0
A[1, 1] = 1.0
A[3, 3] = 1.0
B = 0.01 * (-1 + 2 * np.random.rand(n, m))

# check controllability

C = control.ctrb(A, B)
assert la.matrix_rank(C) == n

# cost

Q, R = np.eye(n), np.eye(m)
P = sp.linalg.solve_discrete_are(A, B, Q, R)

K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

# noise

W = 0.1 * np.random.randn(sim_length, n)

A_tch = torch.tensor(A, dtype=torch.float64)
B_tch = torch.tensor(B, dtype=torch.float64)
Q_tch = torch.tensor(Q, dtype=torch.float64)
R_tch = torch.tensor(R, dtype=torch.float64)
K_tch = torch.tensor(K, dtype=torch.float64)
W_tch = torch.tensor(W, dtype=torch.float64)


# generate code

cpg.generate_code(problem, code_dir='diff_ADP', solver='OSQP', gradient=True)
from diff_ADP.cpg_solver import forward, backward


# projected gradient descent

layer = CvxpyLayer(problem, parameters=[g, H], variables=[u], custom_method=(forward, backward))
solver_args = {'problem': problem, 'warm_start': True, 'eps_abs': 1e-5, 'eps_rel': 1e-5}

def simulate(Psq):
        
    H_tch = Psq @ B_tch
    
    U = torch.zeros((sim_length, m), dtype=torch.float64)
    X = torch.zeros((sim_length, n), dtype=torch.float64)
    
    for i in range(sim_length - 1):
                
        g_tch = Psq @ (A_tch @ X[i])
        solver_args['updated_params'] = ['g', 'H'] if i == 0 else ['g']
            
        U[i], = layer(g_tch, H_tch, solver_args=solver_args)
        X[i+1] = A_tch @ X[i] + B_tch @ U[i] + W_tch[i]
        
    return X, U

def get_performance(X, U):
    p_Q = 0
    p_R = 0
    for i in range(X.shape[0]):
        p_Q += X[i] @ Q_tch @ X[i]
        p_R += U[i] @ R_tch @ U[i]
    return (p_Q + p_R) / X.shape[0]

def get_perf_grad(Psq, get_grad=True):

    Psq_tch = torch.tensor(Psq, dtype=torch.float64, requires_grad=True)            
    X, U = simulate(Psq_tch)
    perf = get_performance(X, U)

    if get_grad:
        perf.backward()
        return perf.detach().item(), Psq_tch.grad.detach().numpy()
    else:
        return perf.detach().item()


# approximate optimal performance objective
        
U = torch.zeros((sim_length, m), dtype=torch.float64)
X = torch.zeros((sim_length, n), dtype=torch.float64)

for i in range(sim_length - 1):
    U[i] = -K_tch @ X[i]
    X[i+1] = A_tch @ X[i] + B_tch @ U[i] + W_tch[i]

fstar_approx = get_performance(X, U).detach().item()


# project

def project_to_psd(M):
    eigenvalues, eigenvectors = la.eigh(M)
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def project_root(root, target):
    rec = root.T @ root
    proj = target + project_to_psd(rec - target)
    return la.cholesky(proj).T

def project_Psq(Psq):
    return project_root(Psq, P)


# tuning

performance = []
Psqrt = la.cholesky(P).T

eps_rel, eps_abs = 1e-2, 1e-2

k = 0

while True:

    perf, grad_P = get_perf_grad(Psqrt, get_grad=True)
    
    if k == 0:
        performance.append(perf)
        grad_squared = np.sum(grad_P**2)
        alpha = min((perf - fstar_approx) / grad_squared, 1)

    while True:
        Psqrt_candidate = project_Psq(Psqrt - alpha * grad_P)

        crit = np.sqrt(np.sum((Psqrt_candidate - Psqrt)**2))
        norm_theta = np.sqrt(np.sum((Psqrt)**2))
        if crit < eps_rel * norm_theta + eps_abs:
            break
        
        perf = get_perf_grad(Psqrt_candidate, get_grad=False)
        if perf < performance[-1]:
            Psqrt = Psqrt_candidate
            alpha *= 1.2
            break
        alpha /= 1.5

    if crit < eps_rel * norm_theta + eps_abs:
        break

    performance.append(perf)
    k += 1
    
    
# plot

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

plt.figure(figsize=(8, 3))
plt.plot(performance, '-o', color='black')
plt.xlabel('Iteration')
plt.xticks(np.arange(len(performance)))
plt.ylabel('$p$')
plt.grid()
plt.show()
