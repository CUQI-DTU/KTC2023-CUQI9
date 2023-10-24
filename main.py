import glob
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mshr import *
from dolfin import Point, SubDomain, MeshFunction, XDMFFile
from dolfin import Function, FunctionSpace, TrialFunction, TestFunction
from dolfin import Measure, inner, nabla_grad, assemble
from dolfin import solve, plot
REFERENCE = sp.io.loadmat("data/TrainingData/ref.mat")
CURRENT_INJECTIONS = REFERENCE["Injref"]
DATA = sorted(glob.glob("data/TrainingData/data*.mat"))
# DATA = sp.io.loadmat("data/TrainingData/data*.mat")
TRUTH = sorted(glob.glob("data/GroundTruths/true*.mat"))

radius = 1
electrode_count = 32
electrode_width = np.pi / electrode_count
phase = np.pi / 2


def create_disk_mesh(radius, electrode_count, polygons, cell_size):
    center = Point(0, 0)
    domain = Circle(center, radius, polygons)
    mesh = generate_mesh(domain, cell_size)

    class Electrode(SubDomain):
        def __init__(self, theta, width):
            super().__init__()
            self.theta = theta
            self.width = width

        def inside(self, x, on_boundary):
            r = np.linalg.norm(x)
            u, v = (np.cos(self.theta), np.sin(self.theta))
            rho = np.arccos(np.dot(x, [u, v]) / r)
            proj = np.maximum(2 * np.abs(rho), self.width)
            return on_boundary and np.isclose(proj, self.width)

    topology = mesh.topology()
    subdomains = MeshFunction("size_t", mesh, topology.dim() - 1)
    for i in range(electrode_count):
        theta = 2 * np.pi * i / electrode_count + phase
        electrode = Electrode(theta, electrode_width)
        electrode.mark(subdomains, i + 1)

    return mesh, subdomains

def _interior_potential_space(mesh):
    return FunctionSpace(mesh, "CG", 1)

def _solution_space(mesh, electrode_count):
    H = _interior_potential_space(mesh)
    R = FunctionSpace(mesh, "R", 0)

    mixed = H.ufl_element()
    for i in range(electrode_count + 1):
        mixed *= R.ufl_element()

    return FunctionSpace(mesh, mixed)

def _domain_measure(mesh):
        dx = Measure('dx', domain=mesh)
        return dx

def _boundary_measure(mesh, subdomains):
        ds = Measure('ds', domain=mesh, subdomain_data=subdomains)
        return ds

def _bilinear_form(solution_space, dx, ds, electrode_count, sigma, impedance):
        # Define trial and test functions
        (u, p, *U) = TrialFunction(solution_space)
        (v, q, *V) = TestFunction(solution_space)

        a = sigma * inner(nabla_grad(u), nabla_grad(v)) * dx
        for i in range(electrode_count):
            a += 1/impedance[i] * (u - U[i]) * (v - V[i]) * ds(i + 1)

            # Enforce mean free electrode potentials
            area = assemble(1*ds(i + 1))
            a += (q*U[i] + p*V[i])/area*ds(i + 1)

        return a

def _rhs(solution_space, dx, ds, electrode_count, current_injection, F = 0):
    
    (_, _, *V) = TestFunction(solution_space)
    
    L = F * dx
    for i in range(electrode_count):
        area = assemble(1*ds(i+1))
        L += (current_injection[i] * V[i] / area) * ds(i+1)

    return L

def _solve(solution_space, bilinear_form, rhs, mesh):
    w = Function(solution_space)
    solve(bilinear_form == rhs, w)
    
    x = w.vector().get_local()
    V = x[-electrode_count:]
    
    # TODO: Find better way to split mixed function
    H = _interior_potential_space(mesh)
    v = Function(H)
    v.vector().set_local(x[:-(electrode_count+1)])
    return v, V

mesh, subdomains = create_disk_mesh(radius, 32, 300, 50)
dx = _domain_measure(mesh)
ds = _boundary_measure(mesh, subdomains)
solution_space = _solution_space(mesh, electrode_count)
a = _bilinear_form(solution_space, dx, ds, electrode_count, 1.0, np.full(electrode_count,1e-6))
L = _rhs(solution_space, dx, ds, electrode_count, CURRENT_INJECTIONS[:,0])

v,V = _solve(solution_space, a, L, mesh)
print(v.vector().get_local().shape)
print(len(mesh.coordinates()))
plot(v)
plt.show()

xdmf = XDMFFile("subdomains.xdmf")
xdmf.write(subdomains)
