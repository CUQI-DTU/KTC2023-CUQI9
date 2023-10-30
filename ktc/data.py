import numpy as np


def _mean_free_potential(potential_difference):
    m, n = potential_difference.shape
    cumsum = np.cumsum(potential_difference, axis=1)
    potential = np.hstack([np.zeros((m, 1)), cumsum])
    return potential - np.mean(potential, axis=1, keepdims=True)


class Data:
    def __init__(self, mat):
        # Load current patterns and injections
        injection_pattern = mat["Mpat"]
        current_injection = mat["Inj"]
        m, n = current_injection.shape

        # Based on the dimensions of the current injections, reshape potentials
        potential_difference = np.reshape(mat["Uel"], (m - 1, n), order="F")

        self.injection_pattern = injection_pattern.T
        self.current_injection = current_injection.T
        self.electrode_potential = _mean_free_potential(potential_difference.T)
        self.electrode_count = m
        self.injection_count = n


class Reference(Data):
    def __init__(self, mat):
        # Load current patterns and injections
        injection_pattern = mat["Mpat"]
        current_injection = mat["Injref"]
        m, n = current_injection.shape

        # Based on the dimensions of the current injections, reshape potentials
        potential_difference = np.reshape(mat["Uelref"], (m - 1, n), order="F")

        self.injection_pattern = injection_pattern.T
        self.current_injection = current_injection.T
        self.electrode_potential = _mean_free_potential(potential_difference.T)
        self.electrode_count = m
        self.injection_count = n
