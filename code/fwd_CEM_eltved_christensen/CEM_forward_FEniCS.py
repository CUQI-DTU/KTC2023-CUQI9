#%%
"""
Note by Amal Alghamdi: This code is copied from the project report: Depth 
Dependency in Electrical Impedance Tomography with the Complete 
Electrode Model by Anders Eltved and Nikolaj Vestbjerg Christensen (Appendix D.5). Some 
modifications are made

Created on Wed May 13 0 8 : 1 8 : 4 7 2015

@author : ec0di
"""

from mshr import *
from dolfin import *
from matplotlib import pyplot as plt
from utils import *
import scipy.io as io
from scipy.interpolate import RegularGridInterpolator

case_name = 'case2' #'case1' , case3', 'case4', 'case_ref'
KTC23_dir = './KTC23_data/'

L = 32
# Define vector of contact impedance
z =  10e-6##0.1 # Impedence 
Z = []
for i in range (L):
    Z.append (z)

# Define domain
R = 1 # radius of circle
n = 300#300 # 300 # number o f polygons to approximate circle
F = 50  # 50 # fineness of mesh
mesh = generate_mesh(Circle(Point( 0, 0), R, n) ,F) # generate mesh
N = mesh.num_entities( 2 )
subdomains = build_subdomains(L, mesh)

xdmf = XDMFFile(case_name+"file_sub.xdmf")
xdmf.write(subdomains)
V, dS = build_spaces(mesh, L, subdomains)

high_conductivity = 1e1
low_conductivity =  1e-2
background_conductivity = 0.8

Imatr = io.loadmat(KTC23_dir+"ref.mat")["Injref"]

if case_name == 'case1':
    phantom_file_name = KTC23_dir+"true1.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(KTC23_dir+"data1.mat")["Uel"]

elif case_name == 'case2':
    phantom_file_name = KTC23_dir+"true2.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(KTC23_dir+"data2.mat")["Uel"]

elif case_name == 'case3':
    phantom_file_name = KTC23_dir+"true3.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(KTC23_dir+"data3.mat")["Uel"]

elif case_name == 'case4':
    phantom_file_name = KTC23_dir+"true4.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat(KTC23_dir+"data4.mat")["Uel"]

elif case_name == 'case_ref':
    phantom_file_name = KTC23_dir+"true1.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    phantom[:] = 0
    Uel_ref = io.loadmat(KTC23_dir+"ref.mat")["Uelref"]

else:
    raise Exception("unknown case")

#%%
# Define conductivity
class inclusion( UserExpression ):
    def __init__(self, phantom , **kwargs):
        super().__init__(**kwargs)
        x_grid = np.linspace(-1, 1, 256)
        y_grid = np.linspace(-1, 1, 256)
        self._interpolater = RegularGridInterpolator((x_grid, y_grid), phantom, method="nearest") 

    def eval( self , values , x ) :
        values[ 0 ] = self._interpolater([ x [ 0 ] , x [ 1 ] ])

phantom_float = np.zeros(phantom.shape)
phantom_float[phantom == 0] = background_conductivity
phantom_float[np.isclose(phantom, 1, rtol = 0.01)] = low_conductivity 
phantom_float[phantom == 2] = high_conductivity

plt.figure()
im = plt.imshow(phantom_float)
plt.colorbar(im) #norm= 'log'
plt.savefig(case_name+"_ph_mod.png")

my_inclusion = inclusion(phantom_float, degree= 1) 
#%%

# Define H1 room
H1=FunctionSpace(mesh ,'CG' , 1)

# Loop over current patterns 
num_inj = 76 # Number of injection pattern
num_inj_tested = 76

B = build_b(my_inclusion, Z, V, dS, L)
Q = np.zeros((L,num_inj))
Diff= np.zeros((L-1,num_inj)) 
q_list = []

for i in range(num_inj)[:num_inj_tested]:
    print("injection pattern"+str(i))
    Q_i , q = solver(Imatr[:,i], B, V, dS, L)
    q_list.append(q)

    Q[:, i] = Q_i
    Diff[:, i] = np.diff(Q_i)

#%% Plot potential solution for each injection pattern
for i, q in enumerate(q_list):
    plt.figure()
    h_func = Function(H1)
    h_func.vector().set_local(q.vector().get_local()[L:-1])
    h_func.vector().get_local()
    im = plot(h_func)
    plt.colorbar(im)
    plt.savefig(case_name+"new_q2_"+str(L)+"_"+str(i)+".png")

# %%
plt.figure()
plt.plot(Q.flatten(order='F'))
plt.savefig(case_name+"Q_flattened"+str(L)+"_"+str(i)+".png")

plt.figure()
plt.plot(Diff.flatten(order='F')[:num_inj_tested*31])
plt.savefig(case_name+"Diff_flattened"+str(L)+"_"+str(i)+".png")

plt.figure()
plt.plot(-Uel_ref[:num_inj_tested*31])
plt.savefig(case_name+"data_flattened"+str(L)+"_"+str(i)+".png")

plt.figure()
plt.plot(-Uel_ref[:num_inj_tested*31].flatten()- Diff.flatten(order='F')[:num_inj_tested*31])
plt.savefig(case_name+"data_model_diff"+str(L)+"_"+str(i)+".png")
# %% Error and model data 
plt.figure()
plt.plot(-Uel_ref[:num_inj_tested*31].flatten()- Diff.flatten(order='F')[:num_inj_tested*31], label= 'Error')
plt.plot(Diff.flatten(order='F')[:num_inj_tested*31], label= 'Model output')
plt.legend()
plt.savefig(case_name+"data_diff_and_model.png"+str(L)+"_"+str(i)+".png")
