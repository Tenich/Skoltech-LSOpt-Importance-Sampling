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
