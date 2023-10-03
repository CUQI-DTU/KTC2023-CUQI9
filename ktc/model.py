import abc
import meshio

import numpy as np
from numpy.typing import ArrayLike


class ForwardModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "jacobian")
            and callable(subclass.jacobian)
            and hasattr(subclass, "solve")
            and callable(subclass.solve)
            and hasattr(subclass, "jacobian")
            and callable(subclass.jacobian)
            or NotImplemented
        )

    @abc.abstractmethod
    def solve(self, current_injection: ArrayLike) -> (ArrayLike, ArrayLike):
        """Compute Neumann to Dirichlet map"""
        raise NotImplementedError

    @abc.abstractmethod
    def jacobian(self, mesh) -> ArrayLike:
        """Return jacobian on mesh"""
        raise NotImplementedError

    @abc.abstractmethod
    def poisson(self, pertubation: ArrayLike, u: ArrayLike) -> ArrayLike:
        """Return solution to generalized Poisson problem"""
        raise NotImplementedError


class FenicsForwardModel(ForwardModel):
    def __init__(self) -> None:
        super().__init__()

    def solve(self, current_injection: ArrayLike) -> ArrayLike:
        """Compute Neumann to Dirichlet map"""
        pass

    def jacobian(self, current_injection: ArrayLike) -> ArrayLike:
        """Return jacobian"""
        pass

    def poisson(
        self, pertubation: ArrayLike, current_injection: ArrayLike
    ) -> ArrayLike:
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

    def jacobian(self, sigma, z):
        pass
