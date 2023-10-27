import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from datetime import datetime
import glob


from dolfin import plot, project, interpolate, FunctionSpace

from ktc.model import create_disk_mesh, FenicsForwardModel
from ktc.reconstruction import SeriesReversion
from ktc.data_reader import DataReader

from fenics import set_log_level

WARNING = 30
set_log_level(WARNING)

REFERENCE = sp.io.loadmat("data/TrainingData/ref.mat")
CURRENT_INJECTIONS = REFERENCE["Injref"].T
DATA_FILES = sorted(glob.glob("data/TrainingData/data1.mat"))
TRUTH_FILES = sorted(glob.glob("data/GroundTruths/true*.mat"))

for idx, fileName in enumerate(DATA_FILES):
    matdict = sp.io.loadmat(fileName)
    data = DataReader(matdict)
    # TRUTH = sp.io.loadmat(    "data/GroundTruths/true" + idx + ".mat")
    
    background_conductivity = 0.8
    electrode_count = 32
    impedance = np.full(electrode_count, 1e-6)
    radius = 1
    
    mesh, subdomains = create_disk_mesh(radius, electrode_count, polygons=200, fineness=20)
    forward_model = FenicsForwardModel(mesh, subdomains,electrode_count, impedance, background_conductivity)
    
    boundary_gap = 0.1
    reconstruction_mesh, _ = create_disk_mesh(radius - boundary_gap, electrode_count, polygons=50, fineness=7)

    W = FunctionSpace(reconstruction_mesh, "DG", 0)
    series_reversion = SeriesReversion(forward_model, reconstruction_mesh, CURRENT_INJECTIONS, W)
 
    F = series_reversion.reconstruct(data.voltages)
    series_reversion.solution_plot(F)
    now = datetime.now()
    file_name = now.strftime("%d_time_%H_%M.png")
    plt.savefig(file_name)
    
