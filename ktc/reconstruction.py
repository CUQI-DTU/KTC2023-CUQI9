import numpy as np
import pygmsh

from model import ForwardModel
from numpy.typing import ArrayLike
class SeriesReversion:
    def __init__(self, model: ForwardModel, mesh, **kwargs):
        self.model = model
        self.mesh = mesh
        self.sigma_ref = kwargs.get("sigma", 1)
        self.data_ref = kwargs.get("data", 0)  # Compute default using fwd model
        self.rec_mesh = self.generate_reconstruction_mesh(boundary_gap = 0.1, mesh_size = 0.2)
        
    def frame_operator(self, current_injections):
        S = current_injections @ current_injections.T
        return S
        
    def reconstruct(self, current_injections: ArrayLike, data: ArrayLike):
        J = self.model.jacobian(self.sigma_ref, self.mesh)
    
        F1 = np.linalg.solve(J, (data - self.data_ref))

        # u, _ = self.model.solve(current_injections)

        # h = -self.model.poisson(F1, current_injections, u)
        # v = self.model.poisson(F1, current_injections, h)
        # F2 = np.linalg.solve(J, v)

        return F1
