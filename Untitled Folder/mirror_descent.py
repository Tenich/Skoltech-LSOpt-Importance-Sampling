import numpy as np
def mirror (x, g, alpha, dgf = 'xlogx'):
    n = len(x)
    updated_x = [x[i] * np.exp(- alpha * g[i]) for i in range(n)]
    return updated_x

def mirror_update(obj, f_class, mirror, C, theta0, n_iters=5000):
    estimations = []
    thetas = [theta0]
    
    cumsum = 0
    theta = theta0
    #print theta
    for i in range(1, n_iters + 1):
        f = f_class(*theta)
        x = f.sample()
        cumsum += obj(x) / f.pdf(x)
        estimations.append(cumsum / i)
        
        tmp = (obj(x) / f.pdf(x)) ** 2
        g = tuple(tmp * (a - b) for a, b in zip(f.grad_A(), f.T(x)))
        alpha = C / i ** 0.5
        theta = mirror(theta, g[0], alpha) #  - C / i ** 0.5 * 
        thetas.append((theta, ))
        
    return thetas, estimations
