import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import glob


from dolfin import plot, project, interpolate, FunctionSpace

from ktc.model import create_disk_mesh, FenicsForwardModel
from ktc.reconstruction import SeriesReversion

REFERENCE = sp.io.loadmat("data/TrainingData/ref.mat")
CURRENT_INJECTIONS = REFERENCE["Injref"].T
DATA_FILES = sorted(glob.glob("data/TrainingData/data1.mat"))
TRUTH_FILES = sorted(glob.glob("data/GroundTruths/true*.mat"))

for idx, fileName in enumerate(DATA_FILES):
    matdict = sp.io.loadmat(fileName)
    uel = matdict.get("Uel")
    # TRUTH = sp.io.loadmat("data/GroundTruths/true" + idx + ".mat")
    
    background_conductivity = 0.8
    electrode_count = 32
    impedance = np.full(electrode_count, 1e-6)
    radius = 1
    
    mesh, subdomains = create_disk_mesh(radius, electrode_count, polygons=300, fineness=50)
    forward_model = FenicsForwardModel(mesh, subdomains,electrode_count, impedance, background_conductivity)
    plot(mesh)
    plt.show()
    
    boundary_gap = 0.1
    reconstruction_mesh, _ = create_disk_mesh(radius - boundary_gap, electrode_count, polygons=5, fineness=1)
    
    W = FunctionSpace(mesh, "DG", 0)
    series_reversion = SeriesReversion(forward_model, reconstruction_mesh, CURRENT_INJECTIONS, W)
    

    # H = forward_model._interior_potential_space()
    # coeffs = forward_model._compute_coefficients(reconstruction_mesh, series_reversion.u)
    # print(coeffs)
    # chi = forward_model._basis(5,W,H)
    # print(chi)
    # plot(chi)
    
    # chi = forward_model._characteristic_function(reconstruction_mesh, 0)
    # H = forward_model._interior_potential_space()
    
    
    # u, U = forward_model.solve_forward(CURRENT_INJECTIONS[:,0])
    # plot(reconstruction_mesh)
    # # print((reconstruction_mesh.num_cells()))
    # # print((reconstruction_mesh.num_vertices()))
    # print(reconstruction_mesh.coordinates())
    # print(len(reconstruction_mesh.coordinates()))
    # # plot(u)

    # plt.show()
    
    
