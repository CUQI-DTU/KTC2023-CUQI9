from reconstruction import SeriesReversion
from model import FenicsForwardModel

fenics_model = FenicsForwardModel(number_of_electrodes=32)

SR = SeriesReversion(model = fenics_model)

print("")
