# %%
from dolfin import *
from matplotlib import pyplot as plt
from utils import *
import scipy.io as io
import nlopt
from KTCRegularization import SMPrior
import pickle

#  set up parameters
high_conductivity = 1e1
low_conductivity = 1e-2
background_conductivity = 0.8

# %% set up data
case_name = 'case1'  # 'case1' , case3', 'case4', 'case_ref'
KTC23_dir = './fwd_CEM_eltved_christensen/KTC23_data/'

Imatr = io.loadmat(KTC23_dir+"ref.mat")["Injref"]

background_phantom_file_name = KTC23_dir+"true1.mat"
background_phantom = io.loadmat(background_phantom_file_name)["truth"]
background_phantom[:] = 0
background_Uel_ref = io.loadmat(KTC23_dir+"ref.mat")["Uelref"].flatten()
background_phantom_float = np.zeros(background_phantom.shape)
background_phantom_float[:] = background_conductivity

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

    phantom = background_phantom
    Uel_ref = background_Uel_ref


else:
    raise Exception("unknown case")



# # Update phantom to make it of float type with correct conductivity values 
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


# %% Voltage with background phantom
background_Uel_sim, background_Q, background_q_list = myeit.solve_forward(Imatr, background_phantom_float, 76)

# %% Prepare simulated/fake data
Uel_sim, Q, q_list = myeit.solve_forward(Imatr, phantom_float, 76)
myeit.add_background_sol_info(background_Uel_sim, background_Uel_ref, background_q_list)
myeit.SetInvGamma( 0.05, 0.01, meas_data=Uel_ref- background_Uel_ref)

# %%
Uel_data =  Uel_ref
#%% OBJECTIVE FUNCTION 
# load smprior object
file = open("smprior_32_300.p", 'rb')
smprior = pickle.load(file)

# %%
eva_count = 0

class Target:
    def __init__(self, myeit, x0, delta) -> None:
        self.myeit = myeit
        self.x0 = x0
        q0 = Function(myeit.H_sigma)
        q0.vector().set_local(x0)
        self.tv_penalty = MyTV(q0, myeit.mesh, delta)
    def eval(self, x, grad):
        global eva_count
        compute_grad = False
        if grad.size >0:
            compute_grad=True
        v1, g1 =  self.myeit.evaluate_target_external(Imatr, x, Uel_data, compute_grad=compute_grad)

        # v2, g2 = smprior.evaluate_target_external(x, compute_grad=compute_grad) # replace this with tv
        qk = Function(self.myeit.H_sigma)
        qk.vector().set_local(x)
        v2 = self.tv_penalty.eval_TV(qk)
        if compute_grad:
            g2 = self.tv_penalty.eval_grad(qk).get_local()
        if grad.size >0:
            grad[:] = g1.flatten()+g2.flatten()
        print("[",eva_count,"]:", v1+v2, "(", v1, "+", v2, ")")
        eva_count += 1


        plt.figure()
        im = plot(self.myeit.inclusion)
        plt.colorbar(im)
        plt.title("sigma")
        plt.show()

        if (compute_grad):
            g1_fenics = Function(self.myeit.H_sigma)
            g1_fenics.vector()[:] = g1.flatten()
            g2_fenics = Function(self.myeit.H_sigma)
            g2_fenics.vector()[:] = g2.flatten()
            g_fenics = Function(self.myeit.H_sigma)
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
delta = 1e-3
# tv_penalty = MyTV(myeit.inclusion, myeit.mesh,delta)



# %%
opt = nlopt.opt(nlopt.LD_SLSQP, myeit.H_sigma.dim())
opt.set_lower_bounds(1e-5*np.ones(myeit.H_sigma.dim()))
opt.set_upper_bounds(1e2*np.ones(myeit.H_sigma.dim()))


#x0= 10*np.ones(myeit.H_sigma.dim())


# %%
#my_target = Target(myeit, x0, delta)

#opt.set_min_objective(my_target.eval)
#opt.set_xtol_rel(1e-4)
#opt.set_maxeval(100)



#x = opt.optimize(x0)
#minf = opt.last_optimum_value()
#print('optimum at ', x)
#print('minimum value = ', minf)
#print('result code = ', opt.last_optimize_result())
#print('nevals = ', opt.get_numevals())
#print('initial step =', opt.get_initial_step(x0))
# %%
#myeit.inclusion.vector()[:] = x
#im = plot(myeit.inclusion)
#plt.colorbar(im)
# %%

# optimise using scipy

# %%

from scipy.optimize import minimize
 
class Target_scipy:
    def __init__(self, myeit, smprior, Imatr, Uel_data, factor=1) -> None:
        self.myeit = myeit
        self.smprior = smprior
        self.Imatr = Imatr
        self.Uel_data = Uel_data
        self.v1 = None
        self.v2 = None
        self.g1 = None
        self.g2 = None
        self.factor = factor
        self.counter = 0
    def obj_scipy(self,x):
        self.counter +=1
        self.v1, self.g1 =  self.myeit.evaluate_target_external(self.Imatr, x, self.Uel_data, compute_grad=True)
        factor = self.factor
        self.v2, self.g2 = self.smprior.evaluate_target_external(x,  compute_grad=True)
        
        print(self.counter)
        if self.counter % 5 == 0:
            plt.figure()
            im = plot(self.myeit.inclusion)
            plt.colorbar(im)
            plt.title("sigma")
            plt.show()
     
     
        print(self.v1+factor*self.v2, "(", self.v1, "+",factor, "*", self.v2,  ")")
     
        return self.v1+factor*self.v2
 
    def obj_scipy_grad(self, x):
        g1 = self.g1
        g2 = self.g2
        factor = self.factor

        if self.counter % 20 == 0:
          g1_fenics = Function(self.myeit.H_sigma)
          g1_fenics.vector()[:] = g1.flatten()
          g2_fenics = Function(self.myeit.H_sigma)
          g2_fenics.vector()[:] = g2.flatten()
          g_fenics = Function(self.myeit.H_sigma)
          g_fenics.vector()[:] = g1.flatten()+factor*g2.flatten()
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
          plt.title("grad (1 + factor*2)")
          plt.show()
        # fig, axs = plt.subplots(1, 4)
        # axs[0].plot(myeit.inclusion)
        # axs[0].set_title("sigma")
        # axs[1].plot(g1_fenics)
        # axs[1].set_title("grad 1")
        # axs[2].plot(g2_fenics)
        # axs[2].set_title("grad 2")
        # axs[3].plot(g_fenics)
        # axs[3].set_title("grad = grad 1 + factor * grad 2")
     
        return g1.flatten()+factor*g2.flatten()
 
target_scipy = Target_scipy( myeit, smprior, Imatr, Uel_data, factor=1)
x0 = 0.8*np.ones(myeit.H_sigma.dim())
# fenics function with circular inclusion
#myexp = Expression("x[0]*x[0] + x[1]*x[1] < r*r ? 0.8 : 0.8", r=0.115, degree=1)
#my_x0 = interpolate(myexp, myeit.H_sigma)
#x0 = my_x0.vector().get_local()
#plt.figure()
#im = plot(my_x0)
bounds = [(1e-5,100)]*myeit.H_sigma.dim()
res = minimize(target_scipy.obj_scipy, x0, method='L-BFGS-B', jac=target_scipy.obj_scipy_grad, options={'disp': True, 'maxiter':500} , bounds=bounds)
# %%
#res_fenics = Function(myeit.H_sigma)
#res_fenics.vector().set_local( res['x'])
res_fenics = target_scipy.myeit.inclusion
plt.figure()
im = plot(res_fenics)
plt.colorbar(im)
# %%
#project and segment
X, Y = np.meshgrid(np.linspace(-1,1,256),np.linspace(-1,1,256) )
Z = np.zeros_like(X)

# interpolate to the grid:
for i in range(256):
    for j in range(256):
        try:
            Z[i,j] = res_fenics(X[i,j], Y[i,j])
        except:
            Z[i,j] = background_conductivity

#%%

from KTCScoring import Otsu2
deltareco_pixgrid = np.flipud(Z)
level, x = Otsu2(deltareco_pixgrid.flatten(), 256, 7)




deltareco_pixgrid_segmented = np.zeros_like(deltareco_pixgrid)
ind0 = deltareco_pixgrid < x[level[0]]
ind1 = np.logical_and(deltareco_pixgrid >= x[level[0]],deltareco_pixgrid <= x[level[1]])
ind2 = deltareco_pixgrid > x[level[1]]
inds = [np.count_nonzero(ind0),np.count_nonzero(ind1),np.count_nonzero(ind2)]
bgclass = inds.index(max(inds)) #background clas
match bgclass:
    case 0:
        deltareco_pixgrid_segmented[ind1] = 2
        deltareco_pixgrid_segmented[ind2] = 2
    case 1:
        deltareco_pixgrid_segmented[ind0] = 1
        deltareco_pixgrid_segmented[ind2] = 2
    case 2:
        deltareco_pixgrid_segmented[ind0] = 1
        deltareco_pixgrid_segmented[ind1] = 1
# fig, ax = plt.subplots()
# cax = ax.imshow(deltareco_pixgrid_segmented, cmap='gray')
# plt.colorbar(cax)
# plt.axis('image')
# plt.title('segmented linear difference reconstruction')
#%%
fig, ax = plt.subplots()
cax = ax.imshow(deltareco_pixgrid_segmented, cmap='gray')
plt.colorbar(cax)
plt.axis('image')
plt.figure()
reconstruction = deltareco_pixgrid_segmented
cax = plt.imshow(phantom, cmap='gray')
plt.colorbar(cax)
# %%
