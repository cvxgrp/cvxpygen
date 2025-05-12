
import cvxpy as cp
from cvxpygen import cpg
from cvxpylayers.torch import CvxpyLayer
import torch
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


# dimensions and data

m, n, J = 100, 20, 10
f = m // J
m_outliers = m // 10

np.random.seed(0)
G_all = np.random.randn(m, n)
x_bar = np.random.randn(n)
h_all = G_all @ x_bar + 0.1 * np.random.randn(m)

def make_points_outliers(points):
    return np.sign(points) * (2 + 2 * np.random.rand(len(points)))

# add outliers to features

for j in range(n):
    outlier_indices = np.random.randint(0, m, size = m_outliers)
    G_all[outlier_indices, j] = make_points_outliers(G_all[outlier_indices, j])
    
    
# problem

x = cp.Variable(n, name='x')
G = cp.Parameter((f * (J - 1), n),name='G')
h = cp.Parameter(f * (J - 1), name='h')
la = cp.Parameter(nonneg=True, name='la')
om = cp.Parameter(nonneg=True, name='om')

problem = cp.Problem(cp.Minimize(cp.sum_squares(G @ x - h) + la * cp.sum_squares(x) + om * cp.norm(x, 1)))


# generate code

cpg.generate_code(problem, code_dir='diff_ML', solver='OSQP', gradient=True, wrapper=True)
from diff_ML.cpg_solver import forward, backward


# projected gradient descent

layer = CvxpyLayer(problem, parameters=[G, h, la, om], variables=[x], custom_method=(forward, backward))
solver_args = {'problem': problem, 'warm_start': True, 'eps_abs': 1e-5, 'eps_rel': 1e-5}

def get_perf_grad(w_value, logla_value, logom_value, get_grad=True):

    G_tch = torch.tensor(G_all, dtype=torch.float64)
    h_tch = torch.tensor(h_all, dtype=torch.float64)
    
    w_tch = torch.tensor(w_value, dtype=torch.float64, requires_grad=get_grad)
    logla = torch.tensor(logla_value, dtype=torch.float64, requires_grad=get_grad)
    logom = torch.tensor(logom_value, dtype=torch.float64, requires_grad=get_grad)
    
    G_winsorized = torch.clamp(G_tch, min=-w_tch, max=w_tch)
    la_val = torch.pow(10, logla)
    om_val = torch.pow(10, logom)

    perf = 0.0

    for j in range(J):

        G_train = torch.vstack((G_winsorized[:j * f], G_winsorized[(j + 1) * f:]))
        G_valid = G_winsorized[j * f:(j + 1) * f]

        h_train = torch.cat((h_tch[:j * f], h_tch[(j + 1) * f:]))
        h_valid = h_tch[j * f:(j + 1) * f]

        x_sol, = layer(G_train, h_train, la_val, om_val, solver_args=solver_args)

        perf_fold = torch.sqrt(torch.mean((G_valid @ x_sol - h_valid) ** 2))
        perf += perf_fold / J

    if get_grad:
        perf_fold.backward()
        return perf.detach().numpy(), w_tch.grad.detach().numpy(), logla.grad.detach().numpy(), logom.grad.detach().numpy()
    else:
        return perf.detach().numpy()


# project

def project_w(w):
    return np.clip(w, 1, 3)

def project_logla(logla):
    return np.clip(logla, -3, 3)

def project_logom(logom):
    return np.clip(logom, -3, 3)


# approximate optimal performance objective

fstar_approx = 0.1


# tuning
    
performance = []

w_value = 3 * np.ones(n)
logla_value = 0.
logom_value = 0.

eps_rel, eps_abs = 1e-3, 1e-3

k = 0

while True:

    perf, grad_w, grad_logla, grad_logom = get_perf_grad(w_value, logla_value, logom_value, get_grad=True)

    if k == 0:
        performance.append(perf)
        grad_squared = np.sum(grad_w**2) + grad_logla**2 + grad_logom**2
        init_subop = perf - fstar_approx
        alpha = min(init_subop / grad_squared, 1)

    while True:
        w_value_candidate = project_w(w_value - alpha * grad_w)
        logla_value_candidate = project_logla(logla_value - alpha * grad_logla)
        logom_value_candidate = project_logom(logom_value - alpha * grad_logom)

        crit = np.sqrt(np.sum((w_value_candidate - w_value)**2) + (logla_value_candidate-logla_value)**2 + (logom_value_candidate-logom_value)**2)
        norm_theta = np.sqrt(np.sum((w_value)**2) + (logla_value)**2 + (logom_value)**2)
        if crit < eps_rel * norm_theta + eps_abs:
            break
        
        perf = get_perf_grad(w_value_candidate, logla_value_candidate, logom_value_candidate, get_grad=False)
        if perf < performance[-1]:
            w_value = w_value_candidate
            logla_value = logla_value_candidate
            logom_value = logom_value_candidate
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
plt.rcParams['axes.axisbelow'] = True

plt.figure(figsize=(8, 3))

plt.plot(performance, '-o', color='black')
plt.xlabel('Iteration')
plt.xticks(np.arange(len(performance)))
plt.ylabel('$p$')
plt.grid()
plt.show()

plt.figure(figsize=(8, 3))

xlim = [0.5, n+0.5]

plt.grid()
plt.plot(xlim, 1 * np.ones(2), 'k--')
plt.plot(xlim, 3 * np.ones(2), 'k--')
plt.scatter(np.arange(n) + 1, 3 * np.ones(n), color='m', s=20)
plt.scatter(np.arange(n) + 1, w_value, color='b', s=20)
plt.xlabel('Feature')
plt.xticks(np.arange(n) + 1)
plt.xlim(xlim)
plt.ylabel('Magnitude threshold')
plt.show()

print(f'lambda = {10**logla_value}, gamma = {10**logom_value}')
