import abc
import typing

import numpy as np
from numpy.typing import ArrayLike


class ForwardModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "jacobian")
            and callable(subclass.jacobian)
            and hasattr(subclass, "neumann_to_dirichlet")
            and callable(subclass.neumann_to_dirichlet)
            and hasattr(subclass, "jacobian")
            and callable(subclass.jacobian)
            or NotImplemented
        )

    @abc.abstractmethod
    def neumann_to_dirichlet(self, current_injection: ArrayLike) -> ArrayLike:
        """Compute Neumann to Dirichlet map"""
        raise NotImplementedError

    @abc.abstractmethod
    def jacobian(self, current_injection: ArrayLike) -> ArrayLike:
        """Return Jacobian"""
        raise NotImplementedError

    @abc.abstractmethod
    def poisson(self, pertubation: ArrayLike, current_injection: ArrayLike) -> ArrayLike:
        """Return solution to generalized Poisson problem"""
        raise NotImplementedError

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
