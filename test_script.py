import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import glob


from dolfin import plot

from ktc.model import create_disk_mesh, FenicsForwardModel

REFERENCE = sp.io.loadmat("data/TrainingData/ref.mat")
CURRENT_INJECTIONS = REFERENCE["Injref"]
DATA_FILES = sorted(glob.glob("data/TrainingData/data1.mat"))
TRUTH_FILES = sorted(glob.glob("data/GroundTruths/true*.mat"))

for idx, fileName in enumerate(DATA_FILES):
    print(idx)
    print(fileName)
    matdict = sp.io.loadmat(fileName)
    uel = matdict.get("Uel")
    # TRUTH = sp.io.loadmat("data/GroundTruths/true" + idx + ".mat")
    
    background_conductivity = 0.8
    electrode_count = 32
    impedance = np.full(32, 1e-6)
    radius = 1
    mesh, subdomains = create_disk_mesh(radius, electrode_count)
    
    forward_model = FenicsForwardModel(mesh, subdomains,electrode_count, impedance, background_conductivity)
    
    u, U = forward_model.solve_forward(CURRENT_INJECTIONS[:,0])
    plot(u)
    plt.show()
    
    
