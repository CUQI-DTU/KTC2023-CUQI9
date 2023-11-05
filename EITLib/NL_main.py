
# %%

from dolfin import *
from matplotlib import pyplot as plt
from .utils import *
import scipy.io as io
from .KTCRegularization_NLOpt import SMPrior
import pickle
from scipy.ndimage import gaussian_filter
from .segmentation import cv, scoring_function
from scipy.optimize import minimize

def NL_main(Uel_ref, background_Uel_ref, Imatr, difficulty_level, niter=50, output_dir_name=None):
#  set up parameters
    high_conductivity = 1e1
    low_conductivity = 1e-2
    background_conductivity = 0.8
    radius = 0.115
    Uel_data =  Uel_ref.flatten()
    background_Uel_ref = background_Uel_ref.flatten()

    # %% build eit-fenics model
    L = 32
    mesh = Mesh()
    with XDMFFile("./EITLib/mesh_file_32_300.xdmf") as infile:
        infile.read(mesh)
    myeit = EITFenics(mesh=mesh, L=L, background_conductivity=background_conductivity)

    
    # %% Voltage with background phantom
    background_phantom_float = np.zeros((256,256)) + background_conductivity
    background_Uel_sim, background_Q, background_q_list = myeit.solve_forward(Imatr, background_phantom_float, 76)
    
    # %% Prepare simulated/fake data
    myeit.add_background_sol_info(background_Uel_sim, background_Uel_ref, background_q_list)
    myeit.SetInvGamma( 0.05, 0.01, meas_data=Uel_data- background_Uel_ref)

    
    #%% OBJECTIVE FUNCTION 
    # load smprior object
    file = open("./EITLib/smprior_32_300.p", 'rb')

    smprior = pickle.load(file)
    
    
    # optimise using scipy

    
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
            plot_flag = False
            if self.counter % 5 == 0 and plot_flag:
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
            plot_flag = False
            if self.counter % 20 == 0 and plot_flag:
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
    # time:
    import time
    start = time.time()
    bounds = [(1e-5,100)]*myeit.H_sigma.dim()
    res = minimize(target_scipy_TV.obj_scipy, x0, method='L-BFGS-B', jac=target_scipy_TV.obj_scipy_grad, options={'disp': True, 'maxiter':niter} , bounds=bounds)
    end = time.time()
    print("time elapsed: ", end-start)
    print("time elapsed in minutes: ", (end-start)/60)
    # save v1_list, v2_list and v3_list
    v1_list = np.array(target_scipy_TV.list_v1)
    v2_list = np.array(target_scipy_TV.list_v2)
    v3_list = np.array(target_scipy_TV.list_v3)
    np.savez(output_dir_name+"/v_list.npz", v1_list=v1_list, v2_list=v2_list, v3_list=v3_list)
    # %%
    res_fenics = Function(myeit.H_sigma)
    res_fenics.vector().set_local( res['x'])
    #res_fenics = target_scipy.myeit.inclusion
    plot_flag = False
    if plot_flag:
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
    deltareco_pixgrid = np.flipud(Z)
    
    return deltareco_pixgrid
    