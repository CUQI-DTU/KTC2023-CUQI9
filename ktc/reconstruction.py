import numpy as np
from itertools import product
from dolfin import Function, plot
import h5py

from ktc.model import FenicsForwardModel
from ktc.smprior import SMPrior

class SeriesReversion:
    # Current injections: each row is a unique injection pattern
    def __init__(self, model, recon_mesh, current_injections, W, voltages_ref):
        self.model = model
        self.recon_mesh = recon_mesh
        self.voltages_ref = voltages_ref

        self.u = []
        self.U = []
        for current_injection in current_injections:
            ui, Ui = self.model.solve_forward(current_injection)
            self.u.append(ui)
            self.U.append(Ui)

        self.W = W
        
        grad_cache_filename = "cache/gradient_cache.hdf5"
        grad_cache = h5py.File(grad_cache_filename, "r")
        grad_id = str(hash(recon_mesh))
        print(grad_id)
        if grad_id in grad_cache.keys():
            print("Read gradient from cache.")
            self.gradient = grad_cache.get(grad_id)
        else:
            print("No gradient found. Generating new gradient.")
            self.gradient = self._gradient()
            grad_cache.close()
            grad_cache = h5py.File(grad_cache_filename, "w")
            grad_cache.create_dataset(grad_id, data = self.gradient)
        grad_cache.close()

    def _gradient(self):
        N = self.recon_mesh.num_cells()

        blocks = []
        for n in range(N):
            P_list = []
            chi = self.model.basis(n, self.W)
            for ui in self.u:
                _, P = self.model.solve_pertubation(chi, ui)
                P_list.append(P)
            blocks.append(P_list)

        return np.block(blocks).T

    def _smoothingPrior(self):
        sigma0 = np.ones((self.recon_mesh.num_vertices(), 1))  # linearization point
        corrlength = 1 * 0.115  # used in the prior
        var_sigma = 0.05**2  # prior variance
        mean_sigma = sigma0

        smprior = SMPrior(self.recon_mesh, corrlength, var_sigma, mean_sigma)
        return smprior.L

    def reconstruct(self, voltages):
        J = self.gradient

        L = self._smoothingPrior()
        # F1, _, _, _ = np.linalg.lstsq(J, (voltages - self.U).flatten(order = "F"))
        F1 = np.linalg.solve(
            J.T @ J + L.T @ L, J.T @ (voltages - self.voltages_ref).flatten(order="F")
        )
        # print(J)
        # u, _ = self.model.solve(current_injections)

        # h = -self.model.poisson(F1, current_injections, u)
        # v = self.model.poisson(F1, current_injections, h)
        # F2 = np.linalg.solve(J, v)

        return F1

    def solution_plot(self, pertubation):
        f = Function(self.W)
        f.vector().set_local(pertubation)
        return plot(f)
