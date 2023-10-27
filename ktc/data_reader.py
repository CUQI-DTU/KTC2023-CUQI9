import scipy as sp
import numpy as np

CURRENT_INJECTION_KEY = "Inj"
CURRENT_INJECTION_KEY_REF = "Injref"
VOLTAGE_DIFFERENCE_KEY = "Uel"
VOLTAGE_DIFFERENCE_KEY_REF = "Uelref"


class DataReader:
    def __init__(self, matdict):
        Uel = None
        try:
            self.current_injections = matdict[CURRENT_INJECTION_KEY].T
            Uel = matdict[VOLTAGE_DIFFERENCE_KEY]
        except:
            self.current_injections = matdict[CURRENT_INJECTION_KEY_REF].T
            Uel = matdict[VOLTAGE_DIFFERENCE_KEY_REF]

        self.injection_count, self.electrode_count = self.current_injections.shape

        diff_shape = (self.electrode_count - 1, self.injection_count)
        voltage_diff = np.reshape(Uel, diff_shape, order="F").T
        self.voltages = self._undifference(voltage_diff)

    def _undifference(self, voltage_differences):
        cumsum = -np.cumsum(voltage_differences, axis=1)
        u = np.hstack([np.zeros((self.injection_count, 1)), cumsum])
        u = u - np.mean(u, axis=1)[:, np.newaxis]
        return u
