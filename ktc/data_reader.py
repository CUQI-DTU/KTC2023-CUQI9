import scipy as sp
import numpy as np

CURRENT_INJECTION_KEY = "Inj"
VOLTAGE_DIFFERENCE_KEY = "Uel"

class DataReader:
    def __init__(self, matdict):
        self.current_injections = matdict[CURRENT_INJECTION_KEY].T
        self.injection_count, self.electrode_count = self.current_injections.shape

        diff_shape = (self.electrode_count-1, self.injection_count)
        voltage_diff = np.reshape(matdict[VOLTAGE_DIFFERENCE_KEY],diff_shape,order="F").T
        self.voltages = self._undifference(voltage_diff)
        
    def _undifference(self, voltage_differences):
        cumsum = np.cumsum(voltage_differences, axis=1)
        u = np.hstack([np.zeros((self.injection_count,1)), cumsum])
        u = u - np.mean(u, axis=1)[:,np.newaxis]
        return u