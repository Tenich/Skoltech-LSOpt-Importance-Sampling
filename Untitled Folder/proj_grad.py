from tqdm import tqdm_notebook
import numpy as np

def _dot(a, b):
    return sum(np.inner(x.ravel(), y.ravel()) for x, y in zip(a, b))

def step(old_theta, step_size, grad):
    return tuple(t - step_size * g for t, g in zip(old_theta, grad))


def projected_gradient(obj, f_class, projector, C, theta0, n_iters=1000):
    estimations = []
    thetas = [theta0]
    
    cumsum = 0
    theta = theta0
    for i in tqdm_notebook(range(1, n_iters + 1)):
        f = f_class(*theta)
        x = f.sample()
        
        cumsum += obj(x) / f.pdf(x)
        estimations.append(cumsum / i)
        
        mult = (obj(x) / f.pdf(x)) ** 2
        g = tuple(mult * (a - b) for a, b in zip(f.grad_A(), f.T(x)))        
        theta = projector(*step(theta, C / i ** 0.5, g))
        thetas.append(theta)
        
    return thetas, estimations



