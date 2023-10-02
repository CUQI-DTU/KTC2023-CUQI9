import numpy as np

class CompleteElectrodeModel:
    def __init__(self, injection, **kwargs):
        self.injection = injection
        self.sigmamin = kwargs.get("sigmamin", 1e-9)
        self.sigmamax = kwargs.get("sigmamax", 1e9)
        self.zmin = kwargs.get("zmin", 1e-6)
        self.zmax = kwargs.get("zmax", 1e6)

    def solve(self, sigma, z):
        sigma = np.clip(sigma, self.sigmamin, self.sigmamax)
        z = np.clip(z, self.zmin, self.zmax)

    def jacobian(self, sigma, z):
        pass
