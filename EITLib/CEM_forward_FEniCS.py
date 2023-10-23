# %%
"""
Note by Amal Alghamdi: This code is copied from the project report: Depth 
Dependency in Electrical Impedance Tomography with the Complete 
Electrode Model by Anders Eltved and Nikolaj Vestbjerg Christensen (Appendix D.5). Some 
modifications are made

Created on Wed May 13 0 8 : 1 8 : 4 7 2015

@author : ec0di
"""
from dolfin import *
from matplotlib import pyplot as plt
from utils import *
import scipy.io as io

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
F = 50
myeit = EITFenics(L, F, background_conductivity=background_conductivity)
num_inj_tested = 76
z = 1e-6 
Uel_sim, Q, q_list = myeit.solve_forward(Imatr, phantom_float, num_inj_tested)

#%%
v_list = myeit.solve_adjoint(q_list, phantom_float, Uel_ref)

#%%
H1 = FunctionSpace(myeit.mesh, 'CG', 1)
h_func = Function(H1)
plot_v = False
if plot_v:
    for i, v in enumerate(v_list):
         plt.figure()
         h_func.vector().set_local(v.vector().get_local()[L:-1])
         # h_func.vector().get_local()
         im = plot(h_func)
         plt.colorbar(im)

#%% Compute gradient 
grad = myeit.evaluate_gradient(q_list, v_list)

#%%
grad.vector().set_local(grad.vector().get_local())
im = plot(grad)
plt.colorbar(im)



#%%
H = FunctionSpace(myeit.mesh, 'CG', 1)
inc_pert_1 = interpolate(myeit.inclusion, H)
sigma_perturb = Function(H)
sigma_perturb.vector().set_local(inc_pert_1.vector().get_local() - 0.8)



#project(myeit.inclusion - 0.8, myeit.V[L])
#%%
print("Solving P")
sigma_background = Function(H)
sigma_background.vector().set_local(inc_pert_1.vector().get_local()*0 + 0.8)

w_list = myeit.solve_P(q_list, sigma_perturb)

#%%
# Solve forward for background phantom A
phantomA = np.copy(phantom_float)
phantomA[:] = 0.8
Uel_sim_A, Q_A, q_list_A = myeit.solve_forward( Imatr, phantomA, num_inj_tested)

#%%
# Solve forward for background phantom AC
phantomAC = np.copy(phantom_float)
Uel_sim_AC, Q_AC, q_list_AC = myeit.solve_forward( Imatr, phantomAC, num_inj_tested)



# %% Plot potential solution for each injection pattern
H1 = FunctionSpace(myeit.mesh, 'CG', 1)
h_func = Function(H1)
for i, w in enumerate(w_list):
     plt.figure()
     h_func.vector().set_local(w.vector().get_local()[L:-1])
     # h_func.vector().get_local()
     im = plot(h_func)
     plt.colorbar(im)
#     plt.savefig(case_name+"_q_L_"+str(L)+"_inj_"+str(i)+".png")

#%%

mismatch_norm_list = []
rel_error_list = []
plot_diffs = False
for j, q in enumerate(q_list):
    # Compute the difference between the rhs and lhs in the paper, right before eq 5.1
    lhs_diff = q_list_A[j].vector().get_local()[L:-1] - q_list_AC[j].vector().get_local()[L:-1]
    
    rhs_ = w_list[j].vector().get_local()[L:-1]
    


    rhs_diff_func = Function(H)
    rhs_diff_func.vector().set_local(rhs_)
    if plot_diffs:
       plt.figure()
       im = plot(rhs_diff_func)
       plt.colorbar(im)
       

    lhs_diff_func = Function(H)
    lhs_diff_func.vector().set_local(-lhs_diff)
    if plot_diffs:       
       plt.figure()
       im = plot(lhs_diff_func)
       plt.colorbar(im)
       

    mismatch_func = Function(H)
    mismatch_func.vector().set_local(rhs_-(-lhs_diff))
    if plot_diffs:     
       plt.figure()
       im = plot(mismatch_func)
       plt.colorbar(im)
       
    
    print(norm(lhs_diff_func))
    print(norm(rhs_diff_func))
    print(norm(mismatch_func))
    mismatch_norm_list.append(norm(mismatch_func))
    rel_error_list.append(norm(mismatch_func)/norm(rhs_diff_func))


# %% postprocess
plt.figure()
plt.plot(Q.flatten(order='F'), label="Q")
plt.legend()
img_title = case_name+"_Q_L_"+str(L) +"_F_"+str(F)
plt.title(img_title)
plt.savefig(img_title+".png")

plt.figure()
plt.plot(Uel_sim[:num_inj_tested*31], label="Uel_sim")
plt.legend()
img_title = case_name+"_Uel_sim_L_"+str(L) +"_F_"+str(F)
plt.title(img_title)
plt.savefig(img_title+".png")

plt.figure()
plt.plot(Uel_ref[:num_inj_tested*31], label="Uel_ref")
plt.legend()
img_title = case_name+"_Uel_ref_L_"+str(L) +"_F_"+str(F)
plt.title(img_title)
plt.savefig(img_title+".png")

plt.figure()
plt.plot(Uel_ref[:num_inj_tested*31] -
         Uel_sim[:num_inj_tested*31], label="Uel_ref - Uel_sim")
plt.legend()
img_title = case_name+"_error_L_"+str(L) +"_F_"+str(F)
plt.title(img_title)
plt.savefig(img_title+".png")

plt.figure()
plt.plot(Uel_sim[:num_inj_tested*31], label='U_sim')
plt.plot(Uel_ref[:num_inj_tested*31] -
         Uel_sim[:num_inj_tested*31], label='Uel_ref - Uel_sim')
plt.legend()
img_title = case_name+"_sim_and_error_L_"+str(L) +"_F_"+str(F)
plt.title(img_title)
plt.savefig(img_title+".png")

# %%
