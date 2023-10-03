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

L = 20

# Define vector of contact impedance
z = 0.1
Z = []
for i in range (L):
    Z.append (z)

# Define domain
R = 1 # radius of circle
n = 300 # number o f polygons to approximate circle
F = 50 # fineness of mesh
mesh = generate_mesh(Circle(Point( 0, 0), R, n) ,F) # generate mesh
N = mesh.num_entities( 2 )
print N

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

I = np.zeros(L)
I[0] = 1
I[1] = -1
sol = solver(sigma_fenics_fun, L, I, Z, mesh)

Q = sol.split()
print Q[0].compute_vertex_values()
plot(Q[1])
plt.savefig("q1.png")
