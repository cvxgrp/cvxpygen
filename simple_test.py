import cvxpy as cp
import numpy as np
from cvxpygen import cpg
import time, sys

T = 15
tspan = 20
dt = tspan / (T - 1)
x0 = cp.Parameter(6)
g = 9.807
tvc_max = np.deg2rad(45.0)
rho1 = 100.0
rho2 = 500.0
m_dry = 25.0
m_fuel = 10.0
Isp = 100.0

g0 = 9.807
m0 = m_dry + m_fuel
a = 1 / (Isp * g0)
nx = 6
nu = 3

A = np.array(
    [
        [1.0, 0.0, 0.0, dt, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, dt, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, dt],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)
B = np.array(
    [
        [0.5 * dt**2, 0.0, 0.0],
        [0.0, 0.5 * dt**2, 0.0],
        [0.0, 0.0, 0.5 * dt**2],
        [dt, 0.0, 0.0],
        [0.0, dt, 0.0],
        [0.0, 0.0, dt],
    ]
)
G = np.array([0.0, 0.0, -0.5 * g * dt**2, 0.0, 0.0, -g * dt])
xT = np.zeros((nx))

x = cp.Variable((nx, T + 1))
z = cp.Variable(T + 1)
u = cp.Variable((nu, T))
s = cp.Variable(T)

# Objective
obj = -z[T]

# IC and TC
con = [x[:, 0] == x0]
con += [x[:, T] == xT]
con += [z[0] == np.log(m0)]
con += [z[T] >= np.log(m_dry)]

# Dynamics
for k in range(T):
    con += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + G]
    con += [z[k + 1] == z[k] - a * s[k] * dt]

# State and Input Constraints
for k in range(T):
    z0 = np.log(m0 - (a * rho2 * k * dt))
    mu1 = rho1 * np.exp(-z0)
    mu2 = rho2 * np.exp(-z0)
    con += [cp.norm(u[:, k]) <= s[k]]
    con += [mu1 * (1.0 - (z[k] - z0) + 0.5 * (z[k] - z0) ** 2) <= s[k]]
    con += [s[k] <= mu2 * (1.0 - (z[k] - z0))]
    con += [np.log(m0 - a * rho2 * k * dt) <= z[k]]
    con += [z[k] <= np.log(m0 - a * rho1 * k * dt)]
    con += [u[2, k] >= s[k] * np.cos(tvc_max)]

prob = cp.Problem(cp.Minimize(obj), con)

# Set initial condition
x0.value = np.array(
        [
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(200, 400),
            0.0,
            0.0,
            0.0,
        ]
    )

# Generate code with CVXPYgen/QOCOGEN
cpg.generate_code(prob, code_dir='lcvx', solver='QOCO')

# Solve problem with CVXPY/QOCO
val = prob.solve(solver='QOCO', verbose=True)
t0 = time.time()
val = prob.solve(solver='QOCO', verbose=True)
t1 = time.time()
sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
sys.stdout.write('Objective function value: %.6f\n' % val)

# Solve problem with CVXPYgen/QOCO Custom
t0 = time.time()
val = prob.solve(method='CPG', verbose=True)
t1 = time.time()
sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
sys.stdout.write('Objective function value: %.6f\n' % val)