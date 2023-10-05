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

L = 32

# Define vector of contact impedance
z = 0.1
Z = []
for i in range (L):
    Z.append (z)

# Define domain
R = 1 # radius of circle
n = 300#300 # 300 # number o f polygons to approximate circle
F = 20  # 50 # fineness of mesh
mesh = generate_mesh(Circle(Point( 0, 0), R, n) ,F) # generate mesh
N = mesh.num_entities( 2 )
subdomains = build_subdomains(L, mesh)
print N

xdmf = XDMFFile("file_sub.xdmf")
xdmf.write(subdomains)




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

mat_file = io.loadmat("ref.mat")
Imatr = mat_file["Injref"]

Q = np.zeros((L,num_inj))
q_list = []
for i in range(num_inj):
    Q_i , q = solver(sigma_fenics_fun, L, Imatr[:,i], Z, mesh, subdomains)
    q_list.append(q)
    Q[:, i] = Q_i


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
    plt.savefig("new_q2_"+str(L)+"_"+str(i)+".png")

# %%
