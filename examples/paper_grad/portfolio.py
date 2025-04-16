
import cvxpy as cp
from cvxpygen import cpg
from cvxpylayers.torch import CvxpyLayer
import torch
import pandas as pd
import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt


def tch(x, grad=False):
    return torch.tensor(x, dtype=torch.float64, requires_grad=grad)


# setup

N, K = 25, 5
kappa = 0.001
h_burnin, h_tune, h_test = 260, 520, 260


# load stock prices and compute returns

prices_df = pd.read_csv('prices.csv', index_col=0)
prices_df = prices_df.dropna(axis='columns')
prices = np.array(prices_df)
returns = np.diff(prices, axis=0) / prices[:-1]
returns_tch = tch(returns)


# estimate mean and covariance
    
mu_ = []

for t in range(h_burnin-1, returns.shape[0]):
    mu = np.zeros(N)
    for tau in range(h_burnin):
        mu += 100*returns[t-tau, :N] / h_burnin
    mu_.append(mu)

Sigma = np.zeros((N, N))

for t in range(h_burnin):
    r = 100*returns[t, :N]
    m = mu_[0]
    Sigma += np.outer(r - m, r - m) / h_burnin
    
Lambda, U = eigs(Sigma, k=K)
Lambda = np.maximum(np.real(Lambda), 0)
U = np.real(U)
diag_Sigma_k = np.array([np.sum(Lambda * U[i, :]**2) for i in range(N)])

F = np.matmul(U, np.diag(np.sqrt(Lambda)))
D_sqrt = np.sqrt(np.maximum(np.diag(Sigma) - diag_Sigma_k, 0))

mu_tch = [tch(m) for m in mu_]
F_tch = tch(F)
D_sqrt_tch = tch(D_sqrt)


# problem

w = cp.Variable(N, name='w')
w_delta = cp.Variable(N, name='w_delta')

mu = cp.Parameter(N, name='mu')
grisk = cp.Parameter(name='grisk', nonneg=True)
ghold = cp.Parameter(name='ghold', nonneg=True)
gtc = cp.Parameter(name='gtc', nonneg=True)
L = cp.Parameter(name='L')
w_pre = cp.Parameter(N, name='w_pre')

objective = cp.Maximize(
    mu @ w
    -grisk * (cp.sum_squares(F.T @ w) + cp.sum_squares(cp.multiply(D_sqrt, w)))
    -ghold * kappa * cp.sum(cp.neg(w))
    -gtc * kappa * cp.norm(w_delta, 1)
)

constraints = [
    cp.sum(w) == 1,
    cp.norm(w, 1) <= L,
    w_delta == w - w_pre
]

problem = cp.Problem(objective, constraints)


# generate code

cpg.generate_code(problem, code_dir='diff_portfolio', solver='OSQP', gradient=True, wrapper=True)
from diff_portfolio.cpg_solver import forward, backward


# project gradient descent

layer = CvxpyLayer(problem, parameters=[mu, grisk, ghold, gtc, L, w_pre], variables=[w, w_delta], custom_method=(forward, backward))
solver_args = {'problem': problem, 'warm_start': True, 'eps_abs': 1e-5, 'eps_rel': 1e-5}


# backtest

def backtest(grisk, ghold, gtc, L, offset, length, grad=False):
    
    w_prevs = [torch.ones(N, dtype=torch.float64, requires_grad=grad) / N for _ in range(length)]
    value = [torch.tensor(1., dtype=torch.float64, requires_grad=grad) for _ in range(length)]

    ws = [torch.ones(N, dtype=torch.float64, requires_grad=grad) / N for _ in range(length)]

    for i_sim in range(length):
        
        i = i_sim + offset
        
        solver_args['updated_params'] = ['w_pre', 'mu']
        if i_sim == 0:
            solver_args['updated_params'].extend(['grisk', 'ghold', 'gtc', 'L'])
    
        ws[i_sim], _ = layer(mu_tch[i_sim][:N], grisk, ghold, gtc, L, w_prevs[i_sim], solver_args=solver_args)
    
        if i_sim + 1 < length:
            value[i_sim + 1] = value[i_sim] * (1. + torch.dot(ws[i_sim], returns_tch[i, :N])) \
                                - kappa * torch.sum(torch.relu(-ws[i_sim])) \
                                - kappa * torch.sum(torch.abs(ws[i_sim] - w_prevs[i_sim]))
        
            w_prevs[i_sim + 1] = ws[i_sim] * (1. + returns_tch[i, :N]) * (value[i_sim] / value[i_sim + 1])
    
    return value


# compute Sharpe ratio

def get_sharpe(V):
    
    V_tensor = torch.stack(V)
    Rp = (V_tensor[1:] - V_tensor[:-1]) / V_tensor[:-1]
    Rebar = torch.mean(Rp)
    sigmae = torch.std(Rp)

    return 5 * torch.sqrt(torch.tensor(10.0)) * Rebar / sigmae


# compute gradient

def get_perf_grad(log_grisk, log_ghold, log_gtc, L, get_grad=True):

    log_grisk_tch = tch(log_grisk, grad=get_grad)
    log_ghold_tch = tch(log_ghold, grad=get_grad)
    log_gtc_tch = tch(log_gtc, grad=get_grad)
    L_tch = tch(L, grad=get_grad)

    grisk = torch.pow(10, log_grisk_tch)
    ghold = torch.pow(10, log_ghold_tch)
    gtc = torch.pow(10, log_gtc_tch)
    
    v = backtest(grisk, ghold, gtc, L_tch, offset=h_burnin, length=h_tune, grad=get_grad)
    S = get_sharpe(v)
    
    if get_grad:
        S.backward()
        grad_log_grisk = log_grisk_tch.grad.detach().numpy()
        grad_log_ghold = log_ghold_tch.grad.detach().numpy()
        grad_log_gtc = log_gtc_tch.grad.detach().numpy()
        grad_L = L_tch.grad.detach().numpy()
        return S.detach().item(), grad_log_grisk, grad_log_ghold, grad_log_gtc, grad_L
    else:
        return S.detach().item()


# approximate optimal performance objective

fstar_approx = 1


# project

def project(a, lb, ub):
    return min(max(a, lb), ub)

def project_log_grisk(log_grisk):
    return project(log_grisk, -3, 3)

def project_log_ghold(log_ghold):
    return project(log_ghold, -3, 3)

def project_log_gtc(log_gtc):
    return project(log_gtc, -3, 3)

def project_L(L):
    return project(L, 1.001, 2)


# tuning

log_grisk, log_ghold, log_gtc, L = 0, 0, 0, 1
eps_rel, eps_abs = 0.03, 0.03
performance = []

k = 0

while True:

    perf, grad_log_grisk, grad_log_ghold, grad_log_gtc, grad_L = get_perf_grad(log_grisk, log_ghold, log_gtc, L, get_grad=True)

    if k == 0:
        performance.append(perf)
        grad_squared = grad_log_grisk**2 + grad_log_ghold**2 + grad_log_gtc**2 + grad_L**2
        alpha = min((fstar_approx - perf) / grad_squared, 1)

    while True:
        log_grisk_candidate = project_log_grisk(log_grisk + alpha * grad_log_grisk)
        log_ghold_candidate = project_log_ghold(log_ghold + alpha * grad_log_ghold)
        log_gtc_candidate = project_log_gtc(log_gtc + alpha * grad_log_gtc)
        L_candidate = project_L(L + alpha * grad_L)
        perf = get_perf_grad(log_grisk_candidate, log_ghold_candidate, log_gtc_candidate, L_candidate, get_grad=False)

        crit = np.sqrt(
            (log_grisk_candidate-log_grisk)**2 + \
            (log_ghold_candidate-log_ghold)**2 + \
            (log_gtc_candidate-log_gtc)**2 + \
            (L_candidate-L)**2
        )
        norm_theta = np.sqrt(log_grisk**2 + log_ghold**2 + log_gtc**2 + L**2)        
        if crit < eps_rel * norm_theta + eps_abs:
            break
        
        if perf > performance[-1]:
            log_grisk = log_grisk_candidate
            log_ghold = log_ghold_candidate
            log_gtc = log_gtc_candidate
            L = L_candidate
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

print(f'grisk={10**log_grisk}, ghold={10**log_ghold}, gtc={10**log_gtc}, L={L}')


# continue simulating out-of-sample and compute Sharpe ratios

v_prior = backtest(grisk=tch(1), ghold=tch(1), gtc=tch(1), L=tch(1), offset=h_burnin, length=h_tune)
v = backtest(grisk=tch(10**log_grisk), ghold=tch(10**log_ghold), gtc=tch(10**log_gtc), L=tch(L), offset=h_burnin, length=h_tune)
v_prior_oos = backtest(grisk=tch(1), ghold=tch(1), gtc=tch(1), L=tch(1), offset=h_burnin + h_tune, length=h_test)
v_oos = backtest(grisk=tch(10**log_grisk), ghold=tch(10**log_ghold), gtc=tch(10**log_gtc), L=tch(L), offset=h_burnin + h_tune, length=h_test)

sr_prior = get_sharpe(v_prior)
sr = get_sharpe(v)
sr_prior_oos = get_sharpe(v_prior_oos)
sr_oos = get_sharpe(v_oos)


# plot

v_plot = np.concatenate(([1.0], v))
v_prior_plot = np.concatenate(([1.0], v_prior))

v_oos_plot = np.array(v_oos) * v_plot[-1]
v_prior_oos_plot = np.array(v_prior_oos) * v_prior_plot[-1]

periods = np.arange(h_tune + 1)
periods_oos = np.arange(h_tune + 1, h_tune + 1 + h_test)

plt.figure(figsize=(8, 3))

plt.plot(periods, v_plot, color='blue')
plt.plot(periods, v_prior_plot, '--', linewidth=1, color='blue')

plt.plot(periods_oos, v_oos_plot, color='magenta')
plt.plot(periods_oos, v_prior_oos_plot, '--', linewidth=1, color='magenta')

ylim = [0.95, max(np.concatenate((v_plot, v_prior_plot, v_oos_plot, v_prior_oos_plot))) + 0.05]

plt.plot([h_tune + 1, h_tune + 1], ylim, 'black')

plt.grid()
plt.xlabel('Trading period')
plt.xlim([min(periods), max(periods_oos)])
plt.ylabel('Portfolio value')
plt.ylim(ylim)
plt.show()

print(f'\nin-sample, before tuning:     SR={np.round(sr_prior, 2)}')
print(f'in-sample, after tuning:      SR={np.round(sr, 2)}\n')

print(f'out-of-sample, before tuning: SR={np.round(sr_prior_oos, 2)}')
print(f'out-of-sample, after tuning:  SR={np.round(sr_oos, 2)}')
