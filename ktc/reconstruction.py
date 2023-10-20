import numpy as np
import pygmsh

from ktc.model import ForwardModel
from numpy.typing import ArrayLike

class SeriesReversion:
    def __init__(self, model: ForwardModel, **kwargs):
        self.model = model
        self.sigma_ref = kwargs.get("sigma", 1)
        self.data_ref = kwargs.get("data", 0)  # Compute default using fwd model
        self.rec_mesh = self.generate_reconstruction_mesh(boundary_gap = 0.1, mesh_size = 0.2)
        
    def generate_reconstruction_mesh(self, boundary_gap, mesh_size):
        radius = 1 - boundary_gap
        with pygmsh.geo.Geometry() as geom:
            geom.add_circle([0.0, 0.0], radius, mesh_size=mesh_size)
            return geom.generate_mesh()


    def reconstruct(self, current_injections: ArrayLike, data: ArrayLike):
        J = self.model.jacobian(self.sigma_ref, self.mesh)

        F1 = np.linalg.solve(J, (data - self.data_ref))

        # u, _ = self.model.solve(current_injections)

        # h = -self.model.poisson(F1, current_injections, u)
        # v = self.model.poisson(F1, current_injections, h)
        # F2 = np.linalg.solve(J, v)

        return F1
