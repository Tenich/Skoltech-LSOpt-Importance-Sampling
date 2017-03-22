import numpy as np


def _dot(a, b):
    return sum(np.inner(x.ravel(), y.ravel()) for x, y in zip(a, b))


class ExponentialFamilyDistribution(object):
    def __init__(self, *theta):
        self.theta = theta

    def A(self):
        return NotImplemented

    def T(self, x):
        return NotImplemented

    def h(self, x):
        return NotImplemented

    def pdf(self, x):
        return self.h(x) * np.exp(_dot(self.theta, self.T(x)) - self.A())

    def sample(self):
        raise NotImplemented


class GaussianDistribution(ExponentialFamilyDistribution):
    def __init__(self, *theta):
        super(GaussianDistribution, self).__init__(*theta)

        m, S = theta
        self.m = m
        self.S = S
        self.cov = np.linalg.pinv(S)
        self.mu = self.cov.dot(m)
        self.k = m.shape[0]

    def A(self):
        return 0.5 * (np.inner(self.m, self.cov.dot(self.m)) - np.log(np.linalg.det(self.S)))

    def grad_A(self):
        # \nabla_S - log(det(S)) = - S^{-1} = - cov
        # \nabla_S^{-1} m^T S^{-1} m =  mm^T
        # \nabla_S m^T S^{-1} m =  -S^{-1} mm^T S^{-1}

        t = self.cov.dot(self.m)
        return (t, -0.5 * (np.outer(t, t) + self.cov))

    def T(self, x):
        return (x, -0.5 * np.outer(x, x))

    def h(self, x):
        return (2 * np.pi) ** (-self.k / 2.)

    def sample(self):
        return np.random.multivariate_normal(mean=self.mu, cov=self.cov)
