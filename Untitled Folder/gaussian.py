from exp_family import *
import numpy as np

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
        return (self.mu, -0.5 * (np.outer(self.mu, self.mu) + self.cov))
    
    def T(self, x):
        return (x, -0.5 * np.outer(x, x))
    
    def h(self, x):
        return (2 * np.pi) ** (-self.k / 2.)
        
    def sample(self):
        return np.random.multivariate_normal(mean=self.mu, cov=self.cov)
