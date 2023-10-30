import numpy as np
from scipy.linalg import solve


def _mean_free_potential(mpat, uel):
    _, n = mpat.shape
    _, m = uel.shape
    a = np.vstack((np.ones((1, n)), mpat))
    b = np.vstack((np.zeros((1, m)), uel))
    return solve(a, b)


def _electrode_mask(inj, drop):
    _, n = inj.shape
    mask = np.ones(inj.shape, dtype=bool)
    for ii in range(n):
        for jj in drop:
            if inj[jj, ii] != 0:
                mask[:, ii] = 0
            mask[jj, :] = 0

    return mask


class Data:
    def __init__(self, mat, drop = []):
        # Load current patterns and injections
        mpat = mat["Mpat"]
        inj = mat["Inj"]
        m, n = inj.shape

        # Based on the dimensions of the current injections, reshape potentials
        uel = np.reshape(mat["Uel"], (m - 1, n), order="F")

        self.mask = _electrode_mask(inj, drop)
        self.measurement_pattern = mpat.T
        self.current_injection = inj.T
        self.electrode_potential = _mean_free_potential(mpat.T, uel).T
        self.electrode_count = m
        self.injection_count = n


class Reference(Data):
    def __init__(self, mat, drop = []):
        # Load current patterns and injections
        mpat = mat["Mpat"]
        injref = mat["Injref"]
        m, n = injref.shape

        # Based on the dimensions of the current injections, reshape potentials
        uelref = np.reshape(mat["Uelref"], (m - 1, n), order="F")

        self.mask = _electrode_mask(injref, drop)
        self.measurement_pattern = mpat.T
        self.current_injection = injref.T
        self.electrode_potential = _mean_free_potential(mpat.T, uelref).T
        self.electrode_count = m
        self.injection_count = n
