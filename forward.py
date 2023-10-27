from ktc.model import *
from ktc.data_reader import *
import matplotlib.pyplot as plt

fileName = "data/TrainingData/data1.mat"
matdict_true = sp.io.loadmat("data/GroundTruths/true1.mat", struct_as_record=False, squeeze_me=True)
matdict_ref = sp.io.loadmat("data/TrainingData/ref.mat", struct_as_record=False, squeeze_me=True)
matdict = sp.io.loadmat(fileName, struct_as_record=False, squeeze_me=True)
data = DataReader(matdict)
ref = DataReader(matdict_ref)
true = matdict_true["truth"]

background_conductivity = 0.8
electrode_count = 32
impedance = np.full(electrode_count, 1e-6)
radius = 1

mesh, subdomains = create_disk_mesh(radius, electrode_count, polygons=200, fineness=20)
forward_model = FenicsForwardModel(mesh, subdomains, electrode_count, impedance, background_conductivity)

u, U1 = forward_model.solve_forward(data.current_injections)
u, U2 = forward_model.solve_forward(ref.current_injections)
plt.figure()
plt.plot(U1)
plt.plot(U2)
plt.plot(true.voltages)