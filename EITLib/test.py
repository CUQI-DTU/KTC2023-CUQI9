#%%
from utils import *
from dolfin import *

#eit_fine = EITFenics(32, 50, 300)
#eit_coarse = EITFenics(32, 5, 200)
eit_fine_mesh = create_disk_mesh(1, 300, 50)
eit_coarse_mesh = create_disk_mesh(1, 50, 5)

plot(eit_fine_mesh)
plt.figure()
plot(eit_coarse_mesh)

#%% Define DG on the coarse mesh
V_coarse = FunctionSpace(eit_coarse_mesh, 'DG', 0)

# Define DG on the fine mesh
V_fine = FunctionSpace(eit_fine_mesh, 'CG', 1)

# Create sine expression on the coarse mesh
f_coarse = Expression('sin(x[0]*pi)*sin(x[1]*pi)', degree=2)

# Create a function on the coarse mesh
u_coarse = interpolate(f_coarse, V_coarse)

# Plot the coarse function
plt.figure()
plot(u_coarse)


# %%
# Interpolate the coarse function on the fine mesh
u_fine = project(u_coarse, V_fine)
plot(u_fine)

#%%
#dx = Measure('dx', domain=eit_coarse_mesh)
#dy = Measure('dx', domain=eit_fine_mesh)

#from dolfin import dx as dy
sigma_test = TestFunction(V_coarse)
f = sigma_test*inner(nabla_grad(u_fine), nabla_grad(u_fine))*dx

# %%
