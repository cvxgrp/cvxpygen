import numpy as np

def pgd(
    sim: callable,
    theta_init: np.ndarray,
    theta_lower: np.ndarray = None,
    theta_upper: np.ndarray = None,
    stepsize: float = 0.1,
    n_iter: int = 25,
    eps: float = 1e-3,
):
    if theta_lower is not None:
        theta_lower += eps
        
    if theta_upper is not None:
        theta_upper -= eps
    
    def _propose(theta, stepsize, grad, compute_grad=True):
        theta_new = np.clip(theta - stepsize * grad, theta_lower, theta_upper)
        val_new, grad_new = sim(theta_new, compute_grad)
        return theta_new, val_new, grad_new
    
    theta = theta_init.copy()
    
    perf = np.zeros(n_iter + 1)
    perf[0], grad = sim(theta, True)

    stepsizes = np.zeros(n_iter + 1)
    stepsizes[0] = stepsize
    
    for it in range(n_iter):
        
        theta_new, perf[it+1], grad_new = _propose(theta, stepsize, grad)
        
        if perf[it+1] < perf[it]:
            theta = theta_new
            grad = grad_new
            stepsize *= 1.5
        else:
            while perf[it+1] >= perf[it]:
                stepsize /= 2
                theta_new, perf[it+1], _ = _propose(theta, stepsize, grad, compute_grad=False)
            theta, perf[it+1], grad = _propose(theta, stepsize, grad)
        
        stepsizes[it+1] = stepsize
                
    return theta, perf, stepsizes
