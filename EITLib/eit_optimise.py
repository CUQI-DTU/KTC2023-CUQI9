# %%
from dolfin import *
from matplotlib import pyplot as plt
from utils import *
import scipy.io as io
import nlopt
from KTCRegularization import SMPrior
import pickle

# %% set up data
case_name = 'case3'  # 'case1' , case3', 'case4', 'case_ref'
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
mesh = Mesh()
with XDMFFile("mesh_file_32_300.xdmf") as infile:
    infile.read(mesh)
myeit = EITFenics(mesh=mesh, L=L, background_conductivity=background_conductivity)
# #%%
# mysigma = interpolate(myeit.inclusion, myeit.H_sigma)
# sigma_values = mysigma.vector()[:]
# sigma_values = np.ones(myeit.mesh.num_cells())
# myeit.evalute_target_external(Imatr, sigma_values, Uel_ref)

# %% Prepare simulated/fake data
Uel_sim, Q, q_list = myeit.solve_forward(Imatr, phantom_float, 76)

# %%
Uel_data = Uel_sim # or Uel_ref
#%% OBJECTIVE FUNCTION 
# load smprior object
file = open("smprior_32_300.p", 'rb')
smprior = pickle.load(file)

# %%
eva_count = 0

def obj(x, grad):
    global eva_count
    compute_grad = False
    if grad.size >0:
        compute_grad=True
    v1, g1 =  myeit.evaluate_target_external(Imatr, x, Uel_data, compute_grad=compute_grad)

    v2, g2 = smprior.evaluate_target_external(x, compute_grad=compute_grad)
    if grad.size >0:
        grad[:] = g1.flatten()+g2.flatten()
    print("[",eva_count,"]:", v1+v2, "(", v1, "+", v2, ")")
    eva_count += 1

    plt.figure()
    im = plot(myeit.inclusion)
    plt.colorbar(im)
    plt.title("sigma")
    plt.show()

    if (compute_grad):
        g1_fenics = Function(myeit.H_sigma)
        g1_fenics.vector()[:] = g1.flatten()
        g2_fenics = Function(myeit.H_sigma)
        g2_fenics.vector()[:] = g2.flatten()
        g_fenics = Function(myeit.H_sigma)
        g_fenics.vector()[:] = grad
        plt.figure()
        im = plot(g1_fenics)
        plt.colorbar(im)
        plt.title("grad 1")
        plt.show()
        plt.figure()
        im = plot(g2_fenics)
        plt.colorbar(im)
        plt.title("grad 2")
        plt.show()
        plt.figure()
        im = plot(g_fenics)
        plt.colorbar(im)
        plt.title("grad (1 + 2)")
        plt.show()

    return v1+v2

# %%
opt = nlopt.opt(nlopt.LD_SLSQP, myeit.H_sigma.dim())
opt.set_lower_bounds(1e-5*np.ones(myeit.H_sigma.dim()))
opt.set_upper_bounds(1e2*np.ones(myeit.H_sigma.dim()))
opt.set_min_objective(obj)
opt.set_xtol_rel(1e-4)
opt.set_maxeval(100)
x0 = 0.8*np.ones(myeit.H_sigma.dim())
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
