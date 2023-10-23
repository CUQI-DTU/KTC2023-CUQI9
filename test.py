import glob
import numpy as np
import scipy as sp
from ktc.model import FenicsForwardModel, create_disk_mesh

DATA = "data/"

radius = 1
electrode_count = 32
phase = np.pi / 2
impedance = np.full(electrode_count, 1e-6)
#model = FenicsForwardModel(radius, electrode_count)

mesh = create_disk_mesh(radius, 300, 50)
#mesh, subdomains = create_disk_mesh(radius, electrode_count, phase)

model = FenicsForwardModel(electrode_count)

num_inj_tested = 5
Imatr = sp.io.loadmat(DATA + "ref.mat")["Injref"]
print(Imatr)
Uel_sim, Q, q_list = model.solve_forward(injection_patterns=Imatr, num_inj_tested = num_inj_tested)
