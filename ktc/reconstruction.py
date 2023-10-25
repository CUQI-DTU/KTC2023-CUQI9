import numpy as np

from ktc.model import FenicsForwardModel
class SeriesReversion:
    # Current injections: each row is a unique injection pattern
    def __init__(self, model, mesh, current_injections, W):
        self.model = model
        self.mesh = mesh
        
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
        
    def reconstruct(self, data):
        J = self.model.gradient(self.sigma_ref, self.mesh)
    
        F1 = np.linalg.solve(J, (data - self.data_ref))

        # u, _ = self.model.solve(current_injections)

        # h = -self.model.poisson(F1, current_injections, u)
        # v = self.model.poisson(F1, current_injections, h)
        # F2 = np.linalg.solve(J, v)

        return F1

    def _gradient():
        pass