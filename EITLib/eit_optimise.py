# %%
from dolfin import *
from matplotlib import pyplot as plt
from utils import *
import scipy.io as io
import nlopt
from KTCRegularization import SMPrior
import pickle
from scipy.ndimage import gaussian_filter
from segmentation import cv, scoring_function

#  set up parameters
high_conductivity = 1e1
low_conductivity = 1e-2
background_conductivity = 0.8
radius = 0.115
difficulty_level = 1

# %% set up data
case_name = 'case1'  # 'case1' , case3', 'case4', 'case_ref'
input_dir = './input/'
output_dir = './output/'

# %% Call KTC challange code to make reconstructions
#import sys
#sys.path.append("/Users/amal/Documents/research_code/CUQI-DTU/Collab-KTC2023/KTC_code")

#from main import main

#main(input_dir,output_dir,difficulty_level)
 

Imatr = io.loadmat(input_dir+"ref.mat")["Injref"]

background_phantom_file_name = input_dir+"true1.mat"
background_phantom = io.loadmat(background_phantom_file_name)["truth"]
background_phantom[:] = 0
background_Uel_ref = io.loadmat(input_dir+"ref.mat")["Uelref"].flatten()
background_phantom_float = np.zeros(background_phantom.shape)
background_phantom_float[:] = background_conductivity

if case_name == 'case1':
    phantom_file_name = input_dir+"true1.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(input_dir+"data1.mat")["Uel"].flatten()
    recon_KTC_file = output_dir+"1.mat"
    

elif case_name == 'case2':
    phantom_file_name = input_dir+"true2.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(input_dir+"data2.mat")["Uel"].flatten()

elif case_name == 'case3':
    phantom_file_name = input_dir+"true3.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(input_dir+"data3.mat")["Uel"].flatten()

elif case_name == 'case4':
    phantom_file_name = input_dir+"true4.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(input_dir+"data4.mat")["Uel"].flatten()

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
            plt.title("sigma "+str(self.counter))
            plt.show()

        if self.counter == 30:
            self.factor = 1
     
     
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
 
target_scipy = Target_scipy( myeit, smprior, Imatr, Uel_data, factor=1e-4)

# Class Target_scipy_TV for TV regularization that uses TV_reg

class Target_scipy_TV:
    def __init__(self, myeit, tv_reg, smprior, Imatr, Uel_data, factor=1, factor_sm=1) -> None:
        self.myeit = myeit
        self.tv_reg = tv_reg
        self.Imatr = Imatr
        self.Uel_data = Uel_data
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.g1 = None
        self.g2 = None
        self.g3 = None
        self.factor = factor
        self.factor_sm = factor_sm
        self.smprior = smprior
        self.counter = 0
        self.list_v1 = []
        self.list_v2 = []
        self.list_v3 = []


    def obj_scipy(self,x):
        x_fun = Function(self.myeit.H_sigma)
        x_fun.vector().set_local(x)

        self.counter +=1
        self.v1, self.g1 =  self.myeit.evaluate_target_external(self.Imatr, x, self.Uel_data, compute_grad=True)
        self.v3, self.g3 = self.smprior.evaluate_target_external(x,  compute_grad=True)
        factor = self.factor
        self.v2 = self.tv_reg.cost_reg(x_fun )
        #self.factor = 0.6*((self.v1/2)/self.v2)
        #self.factor_sm = 0.6*((self.v1/2)/self.v3)
        self.list_v1.append(np.log(self.v1))
        self.list_v2.append(np.log(self.factor*self.v2))
        self.list_v3.append(np.log(self.factor_sm*self.v3))



        
        print(self.counter)
        if self.counter % 5 == 0:
            plt.figure()
            im = plot(self.myeit.inclusion)
            plt.colorbar(im)
            plt.title("sigma "+str(self.counter))
            plt.show()
            plt.figure()
            plt.plot(self.list_v1)
            plt.title("v1")
            plt.show()
            plt.figure()
            plt.plot(self.list_v2)
            plt.title("v2")
            plt.show()
            plt.figure()
            plt.plot(self.list_v3)
            plt.title("v3")
            plt.show()

            
        

         
        print(self.v1+factor*self.v2, "(", self.v1, "+", factor*self.v2,  "+", self.factor_sm*self.v3, ")")
     
        return self.v1+factor*self.v2+self.factor_sm*self.v3
 
    def obj_scipy_grad(self, x):
        x_fun = Function(self.myeit.H_sigma)
        x_fun.vector().set_local(x)
        g1 = self.g1
        g2 = self.tv_reg.grad_reg(x_fun).get_local()
        self.g2 = g2
        g3 = self.g3
        
        factor = self.factor
        factor_sm = self.factor_sm

        if self.counter % 20 == 0:
          g1_fenics = Function(self.myeit.H_sigma)
          g1_fenics.vector()[:] = g1.flatten()
          g2_fenics = Function(self.myeit.H_sigma)
          g2_fenics.vector()[:] = g2.flatten()
          g3_fenics = Function(self.myeit.H_sigma)
          g3_fenics.vector()[:] = g3.flatten()
          g_fenics = Function(self.myeit.H_sigma)
          g_fenics.vector()[:] = g1.flatten()+factor*g2.flatten()+factor_sm*g3.flatten()
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
          im = plot(g3_fenics)
          plt.colorbar(im)
          plt.title("grad 3")
          plt.figure()
          im = plot(g_fenics)
          plt.colorbar(im)
          plt.title("grad (g1 + factor*g2 + factor_sm*g3)")
          plt.show()


        return g1.flatten()+factor*g2.flatten()+factor_sm*g3.flatten()

          
        



#%%

# Create initial guess x0

recon_background_flag = False

if recon_background_flag:
    recon_KTC = io.loadmat(recon_KTC_file)["reconstruction"]
    recon_KTC_float = np.zeros_like(recon_KTC)
    recon_KTC_float[:] = recon_KTC
    recon_KTC_float[recon_KTC_float == 0] = background_conductivity
    recon_KTC_float[recon_KTC_float == 1] = low_conductivity
    recon_KTC_float[recon_KTC_float == 2] = high_conductivity
    
    im = plt.imshow(np.flipud(recon_KTC_float))
    plt.title('KTC reconstruction, flipped for interpolation')
    plt.colorbar(im)

    recon_KTC_float_smoothed = gaussian_filter(recon_KTC_float, sigma=30)
    plt.figure()
    im = plt.imshow(recon_KTC_float_smoothed)
    plt.colorbar(im)
    
    
    x0_exp = Inclusion(np.fliplr(recon_KTC_float_smoothed.T), radius, degree=1)
    x0_fun = interpolate(x0_exp, myeit.H_sigma)
    plot(x0_fun)
    x0 = x0_fun.vector().get_local()
else:
    x0 = 0.8 * np.ones(myeit.H_sigma.dim())


tv_reg = TV_reg(myeit.H_sigma, None, 1, 1e-4)
target_scipy_TV = Target_scipy_TV( myeit, tv_reg, smprior=smprior, Imatr=Imatr, Uel_data=Uel_data, factor=5e6, factor_sm=0.6)
#%%

# fenics function with circular inclusion
#myexp = Expression("x[0]*x[0] + x[1]*x[1] < r*r ? 0.8 : 0.8", r=0.115, degree=1)
#my_x0 = interpolate(myexp, myeit.H_sigma)
#x0 = my_x0.vector().get_local()
#plt.figure()
#im = plot(my_x0)
# time:
import time
start = time.time()
bounds = [(1e-5,100)]*myeit.H_sigma.dim()
res = minimize(target_scipy_TV.obj_scipy, x0, method='L-BFGS-B', jac=target_scipy_TV.obj_scipy_grad, options={'disp': True, 'maxiter':50} , bounds=bounds)
end = time.time()
print("time elapsed: ", end-start)
print("time elapsed in minutes: ", (end-start)/60)
# %%
#res_fenics = Function(myeit.H_sigma)
#res_fenics.vector().set_local( res['x'])
res_fenics = target_scipy.myeit.inclusion
plt.figure()
im = plot(res_fenics)
plt.colorbar(im)
# %%
#project and segment

X, Y = np.meshgrid(np.linspace(-radius,radius,256),np.linspace(-radius,radius,256) )
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
plt.figure()
cax = plt.imshow(deltareco_pixgrid_segmented)


# plot circle of radius 0.115
theta = np.linspace(0, 2*np.pi, 100)
cir_rad = 256/2
x = cir_rad*np.cos(theta)
y = cir_rad*np.sin(theta)
plt.plot(x+cir_rad, y+cir_rad, color='red', linewidth=2)
plt.title('segmented reconstruction')
plt.axis('image')
plt.colorbar(cax)

#%% plot chan vesa segmentation
cv_seg = cv( np.log(deltareco_pixgrid) + deltareco_pixgrid, mu=.1, lambda1=0.5, lambda2=0.5, init_level_set='checkerboard')

plt.figure()
im = plt.imshow(cv_seg, cmap='gray')
plt.colorbar(im)
plt.title('Chan Vese segmentation')

#%%
plt.figure()
cax = plt.imshow(phantom, cmap='gray')
plt.title('phantom')
plt.colorbar(cax)
# %%
#print("KTC score: ", scoring_function(recon_KTC,phantom))
print("CV score: ", scoring_function(cv_seg,phantom))
print("Otsu score: ", scoring_function(deltareco_pixgrid_segmented,phantom))



# %%
