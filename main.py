import glob
import numpy as np
import scipy as sp
from mshr import *
from dolfin import Point, SubDomain, MeshFunction, XDMFFile
from dolfin import FunctionSpace, TrialFunction, TestFunction
from dolfin import Measure, inner, nabla_grad, assemble

REFERENCE = glob.glob("data/TrainingData/ref.mat")
DATA = sorted(glob.glob("data/TrainingData/data*.mat"))
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


def _solution_space(mesh, electrode_count):
    H = FunctionSpace(mesh, "CG", 1)
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

        return assemble(a)

def _rhs(solution_space, dx, ds, electrode_count, current_injection, F = 0):
    
    (_, _, *V) = TestFunction(solution_space)
    
    L = F * dx
    for i in range(electrode_count):
        area = assemble(1*ds(i+1))
        L += (current_injection[i] * V[i] / area) * ds(i+1)

    return assemble(L)



mesh, subdomains = create_disk_mesh(radius, 32, 300, 50)
dx = _domain_measure(mesh)
ds = _boundary_measure(mesh, subdomains)
solution_space = _solution_space(mesh, electrode_count)
B = _bilinear_form(solution_space, dx, ds, electrode_count, 1.0, np.full(electrode_count,1e-6))

current_injections = sp.io.loadmat("data/TrainingData/ref.mat")["Injref"]
f = _rhs(solution_space, dx, ds, electrode_count, current_injections[:,0])
xdmf = XDMFFile("subdomains.xdmf")
xdmf.write(subdomains)

