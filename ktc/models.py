import abc
import meshio

import numpy as np
from numpy.typing import ArrayLike


class ForwardModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "frechet_derivative")
            and callable(subclass.frechet_derivative)
            and hasattr(subclass, "neumann_to_dirichlet")
            and callable(subclass.neumann_to_dirichlet)
            and hasattr(subclass, "frechet_derivative")
            and callable(subclass.frechet_derivative)
            or NotImplemented
        )

    @abc.abstractmethod
    def neumann_to_dirichlet(self, current_injection: ArrayLike) -> ArrayLike:
        """Compute Neumann to Dirichlet map"""
        raise NotImplementedError

    @abc.abstractmethod
    def frechet_derivative(self, current_injection: ArrayLike, mesh) -> ArrayLike:
        """Return frechet_derivative on mesh"""
        raise NotImplementedError

    @abc.abstractmethod
    def poisson(self, pertubation: ArrayLike, current_injection: ArrayLike) -> ArrayLike:
        """Return solution to generalized Poisson problem"""
        raise NotImplementedError


class FenicsForwardModel(ForwardModel):
    def neumann_to_dirichlet(self, current_injection: ArrayLike) -> ArrayLike:
        """Compute Neumann to Dirichlet map"""
        pass

    def frechet_derivative(self, current_injection: ArrayLike) -> ArrayLike:
        """Return frechet_derivative"""
        pass


    def poisson(self, pertubation: ArrayLike, current_injection: ArrayLike) -> ArrayLike:
        """Return solution to generalized Poisson problem"""
        pass


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

    def frechet_derivative(self, sigma, z):
        pass
