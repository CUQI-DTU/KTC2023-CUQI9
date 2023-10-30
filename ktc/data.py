import numpy as np
from scipy.linalg import solve


def _mean_free_potential(mpat, uel):
    _, n = mpat.shape
    _, m = uel.shape
    a = np.vstack((np.ones((1, n)), mpat))
    b = np.vstack((np.zeros((1, m)), uel))
    return solve(a, b)


class Data:
    def __init__(self, mat):
        # Load current patterns and injections
        mpat = mat["Mpat"]
        inj = mat["Inj"]
        m, n = inj.shape

        # Based on the dimensions of the current injections, reshape potentials
        uel = np.reshape(mat["Uel"], (m - 1, n), order="F")

        self.measurement_pattern = mpat.T
        self.current_injection = inj.T
        self.electrode_potential = _mean_free_potential(mpat.T, uel).T
        self.electrode_count = m
        self.injection_count = n


class Reference(Data):
    def __init__(self, mat):
        # Load current patterns and injections
        mpat = mat["Mpat"]
        injref = mat["Injref"]
        m, n = injref.shape

        # Based on the dimensions of the current injections, reshape potentials
        uelref = np.reshape(mat["Uelref"], (m - 1, n), order="F")

        self.measurement_pattern = mpat.T
        self.current_injection = injref.T
        self.electrode_potential = _mean_free_potential(mpat.T, uelref).T
        self.electrode_count = m
        self.injection_count = n
