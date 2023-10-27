# %%
from dolfin import *
from matplotlib import pyplot as plt
from utils import *
import scipy.io as io
import nlopt

# %% set up data
case_name = 'case1'  # 'case1' , case3', 'case4', 'case_ref'
KTC23_dir = './fwd_CEM_eltved_christensen/KTC23_data/'

Imatr = io.loadmat(KTC23_dir+"ref.mat")["Injref"]

if case_name == 'case1':
    phantom_file_name = KTC23_dir+"true1.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(KTC23_dir+"data1.mat")["Uel"].flatten()

elif case_name == 'case2':
    phantom_file_name = KTC23_dir+"true2.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(KTC23_dir+"data2.mat")["Uel"].flatten()

elif case_name == 'case3':
    phantom_file_name = KTC23_dir+"true3.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(KTC23_dir+"data3.mat")["Uel"].flatten()

elif case_name == 'case4':
    phantom_file_name = KTC23_dir+"true4.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(KTC23_dir+"data4.mat")["Uel"].flatten()

elif case_name == 'case_ref':
    phantom_file_name = KTC23_dir+"true1.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    phantom[:] = 0
    Uel_ref = io.loadmat(KTC23_dir+"ref.mat")["Uelref"].flatten()

else:
    raise Exception("unknown case")

# Update phantom to make it of float type with correct conductivity values 
high_conductivity = 1e1
low_conductivity = 1e-2
background_conductivity = 0.8
# Define conductivity
phantom_float = np.zeros(phantom.shape)
phantom_float[phantom == 0] = background_conductivity
phantom_float[np.isclose(phantom, 1, rtol=0.01)] = low_conductivity
phantom_float[phantom == 2] = high_conductivity

# %% build eit-fenics model
L = 32
F = 30
myeit = EITFenics(L, F, background_conductivity=background_conductivity)
# #%%
# mysigma = interpolate(myeit.inclusion, myeit.H_sigma)
# sigma_values = mysigma.vector()[:]
# sigma_values = np.ones(myeit.mesh.num_cells())
# myeit.evalute_target_external(Imatr, sigma_values, Uel_ref)



# %%
opt = nlopt.opt(nlopt.LD_MMA, myeit.mesh.num_cells())
opt.set_lower_bounds(1e-5*np.ones(myeit.mesh.num_cells()))
opt.set_upper_bounds(1e1*np.ones(myeit.mesh.num_cells()))
opt.set_min_objective(lambda x, grad: myeit.evalute_target_external(Imatr, x, Uel_ref, grad))
opt.set_xtol_rel(1e-4)
opt.set_maxeval(100)
x0 = 0.8*np.ones(myeit.mesh.num_cells())
x = opt.optimize(x0)
minf = opt.last_optimum_value()
print('optimum at ', x)
print('minimum value = ', minf)
print('result code = ', opt.last_optimize_result())
print('nevals = ', opt.get_numevals())
print('initial step =', opt.get_initial_step(x0))
# %%
myeit.inclusion.vector()[:] = x
im = plot(myeit.inclusion)
plt.colorbar(im)
# %%
