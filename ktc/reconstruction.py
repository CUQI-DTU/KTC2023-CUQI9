import numpy as np
from itertools import product

from ktc.model import FenicsForwardModel
from dolfin import Function, plot
class SeriesReversion:
    # Current injections: each row is a unique injection pattern
    def __init__(self, model, recon_mesh, current_injections, W):
        self.model = model
        self.recon_mesh = recon_mesh
        
        self.u = []
        self.U = []
        for current_injection in current_injections:
           ui, Ui = self.model.solve_forward(current_injection)
           self.u.append(ui)
           self.U.append(Ui)
            
        # Construct conoical dual frame
        # TODO: Solve using lsq (and verify it is correct)
        self.dual_frame = np.linalg.solve(current_injections @ current_injections.T,current_injections)
        self.W = W
        self.gradient = self._gradient()
        
    def _gradient(self):
        N = self.recon_mesh.num_cells()
        
        blocks = []
        for n in range(N):
            P_list = []
            chi = self.model.basis(n,self.W)
            for ui in self.u:
                _, P = self.model.solve_pertubation(chi, ui)
                P_list.append(P)
            blocks.append(P_list)

        return np.block(blocks).T
        
    def reconstruct(self, voltages):
        J = self.gradient
    
        F1, _, _, _ = np.linalg.lstsq(J, (voltages - self.U).flatten(order = "F"))

        # u, _ = self.model.solve(current_injections)

        # h = -self.model.poisson(F1, current_injections, u)
        # v = self.model.poisson(F1, current_injections, h)
        # F2 = np.linalg.solve(J, v)

        return F1
    
    def solution_plot(self, pertubation):
        f = Function(self.W)
        f.vector().set_local(pertubation)
        plot(f)
