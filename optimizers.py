from tqdm import tqdm_notebook
import numpy as np


####### Projected gradient method #######

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
        # print(obj(x))
        cumsum += obj(x) / f.pdf(x)
        estimations.append(cumsum / i)

        mult = (obj(x) / f.pdf(x)) ** 2
        g = tuple(mult * (a - b) for a, b in zip(f.grad_A(), f.T(x)))
        theta = projector(*step(theta, C, g))
        thetas.append(theta)

    return thetas, estimations


####### Mirror descent method #######

def mirror(x, g, alpha, dgf='xlogx'):
    n = len(x)
    updated_x = [x[i] * np.exp(- alpha * g[i]) for i in range(n)]
    return updated_x


def mirror_update(obj, f_class, mirror, C, theta0, n_iters=5000):
    estimations = []
    thetas = [theta0]

    cumsum = 0
    theta = theta0
    # print theta
    for i in tqdm_notebook(range(1, n_iters + 1)):
        f = f_class(*theta)
        x = f.sample()
        cumsum += obj(x) / f.pdf(x)
        estimations.append(cumsum / i)

        tmp = (obj(x) / f.pdf(x)) ** 2
        g = tuple(tmp * (a - b) for a, b in zip(f.grad_A(), f.T(x)))
        alpha = C / i ** 0.5
        theta = mirror(theta, g[0], alpha)  # - C / i ** 0.5 *
        thetas.append((theta, ))

    return thetas, estimations


####### ADMM method #######
import scipy as sp


def __ravel(x):
    return np.concatenate([c.ravel() for c in x])


def __reshape(x, shapes):
    result = []
    shift = 0
    for shape in shapes:
        result.append(x[shift: shift + np.sum(shape)].reshape(shape))
        shift += np.sum(shape)
    return tuple(result)


def admm(obj, f_class, projector, r, theta0, n_iters=1000):
    raise NotImplemented

    estimations = []
    thetas = [theta0]

    cumsum = 0
    theta = theta0
    theta_shape = tuple([x.shape for x in theta])
    for i in tqdm_notebook(range(1, n_iters + 1)):
        f = f_class(*theta)
        x = f.sample()

        cumsum += obj(x) / f.pdf(x)
        estimations.append(cumsum / i)

        def subtask_obj(theta):
            theta = __reshape(theta, theta_shape)
            return obj(theta) ** 2 / f.pdf(theta)

        res = scipy.optimize.minimize(subtask_obj, )

        mult = (obj(x) / f.pdf(x)) ** 2
        g = tuple(mult * (a - b) for a, b in zip(f.grad_A(), f.T(x)))
        theta = projector(*step(theta, C / i ** 0.5, g))
        thetas.append(theta)

    return thetas, estimations
