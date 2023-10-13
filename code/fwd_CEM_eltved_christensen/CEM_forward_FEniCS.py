# - - coding : utf-8 - -
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

L = 32
case_name = 'case_ref' #'case1' , case3', 'case4', 'case_ref'
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
print N

xdmf = XDMFFile("file_sub.xdmf")
xdmf.write(subdomains)
V, dS = build_spaces(mesh, L, subdomains)

high_conductivity = 1e1
low_conductivity =  1e-2
background_conductivity = 0.8

Imatr = io.loadmat("ref.mat")["Injref"]

if case_name == 'case1':
    phantom_file_name = "true1.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat("data1.mat")["Uel"]

elif case_name == 'case2':
    phantom_file_name = "true2.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat("data2.mat")["Uel"]

elif case_name == 'case3':
    phantom_file_name = "true3.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat("data3.mat")["Uel"]

elif case_name == 'case4':
    phantom_file_name = "true4.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    Uel_ref = io.loadmat("data4.mat")["Uel"]

elif case_name == 'case_ref':
    phantom_file_name = "true1.mat"
    phantom = io.loadmat(phantom_file_name)["truth"]
    phantom[:] = 0
    Uel_ref = io.loadmat("ref.mat")["Uelref"]

else:
    raise Exception("unknown case")

#%%
# Define conductivity
class sigma_fun( Expression ):
    def eval( self , values , x ):
        values[ 0 ] = 1


# Define h
# h is defined from file h_functions
def h_fun(x , y ) :
    return h_geometry (x , y )

class sigma_h_fun( Expression ):
    def eval( self , values , x ) :
        values[ 0 ] = sigma ( x )+h_fun ( x [ 0 ] , x [ 1 ] )

class inclusion( Expression ):
    def __init__(self, phantom, **kwargs ):
        x_grid = np.linspace(-1, 1, 256)
        y_grid = np.linspace(-1, 1, 256)
        self.interpolater = RegularGridInterpolator((x_grid, y_grid), phantom, method="nearest") 
         

    def eval( self , values , x ) :
        values[ 0 ] = self.interpolater([ x [ 0 ] , x [ 1 ] ])


ph1_float = np.zeros(phantom.shape)

ph1_float[phantom == 0] = background_conductivity
ph1_float[np.isclose(phantom, 1, rtol = 0.01)] = low_conductivity 
ph1_float[phantom == 2] = high_conductivity

plt.figure()
im = plt.imshow(np.log(ph1_float))
plt.colorbar(im) #norm= 'log'
plt.savefig(case_name+"ph_mod.png")

my_inclusion = inclusion(ph1_float, degree= 1) 
#%%

# Define string names for later print
sigmastr = "sigma=1"
hstr = "h=0.3_geometry"

# Define H1 room
H1=FunctionSpace(mesh ,'CG' , 1)

# Initiate functions
sigma = sigma_fun( element=H1.ufl_element() )
sigma_h = sigma_h_fun( element=H1.ufl_element() )

# AMAL: which sigma to use?
sigma_fenics_fun = interpolate(sigma, H1)

# Loop over current patterns 
num_inj = 76 # Number of injection pattern
num_inj_tested = 76



B = build_b(my_inclusion, Z, V, dS, L)

Q = np.zeros((L,num_inj))
Diff= np.zeros((L-1,num_inj)) 
q_list = []
for i in range(num_inj)[:num_inj_tested]:
    print "injection pattern", i
    Q_i , q = solver(Imatr[:,i], B, V, dS, L)
    q_list.append(q)

    Q[:, i] = Q_i
    Diff[:, i] = np.diff(Q_i)



#Q , q, subdomains = solver_4_quarters(sigma_fenics_fun, I, Z, mesh)

#Q = sol.split()
#print Q[0].compute_vertex_values()
#plot(Q[1])
#plt.savefig("q1.png")
#plt.figure()
#im = plot(q[2])
#plt.colorbar(im)
#plt.savefig("q2_"+str(L)+".png")

#%%
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



# %%
