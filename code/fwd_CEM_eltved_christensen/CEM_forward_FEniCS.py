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
KTC23_dir = './KTC23_data/'

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

# %% build eit-fenics model
L = 32
F = 50
myeit = EITFenics(L, F)
myeit.create_inclusion(phantom)
num_inj_tested = 76
Uel_sim, Q, q_list = myeit.solve_forward(Imatr, num_inj_tested, 1e-6)

# # %% Plot potential solution for each injection pattern
# H1 = FunctionSpace(myeit.mesh, 'CG', 1)
# h_func = Function(H1)
# for i, q in enumerate(q_list):
#     plt.figure()
#     h_func.vector().set_local(q.vector().get_local()[L:-1])
#     # h_func.vector().get_local()
#     im = plot(h_func)
#     plt.colorbar(im)
#     plt.savefig(case_name+"_q_L_"+str(L)+"_inj_"+str(i)+".png")

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
