import glob
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mshr import *
from dolfin import Point, SubDomain, MeshFunction, XDMFFile
from dolfin import Function, FunctionSpace, TrialFunction, TestFunction
from dolfin import Measure, inner, nabla_grad, assemble
from dolfin import solve, plot
# REFERENCE = sp.io.loadmat("data/TrainingData/ref.mat")
# CURRENT_INJECTIONS = REFERENCE["Injref"]
# DATA = sorted(glob.glob("data/TrainingData/data*.mat"))
# # DATA = sp.io.loadmat("data/TrainingData/data*.mat")
# TRUTH = sorted(glob.glob("data/GroundTruths/true*.mat"))

# radius = 1
# electrode_count = 32
# electrode_width = np.pi / electrode_count
# phase = np.pi / 2


def create_disk_mesh(radius, electrode_count, polygons = 300, cell_size = 50):
    center = Point(0, 0)
    domain = Circle(center, radius, polygons)
    mesh = generate_mesh(domain, cell_size)
    electrode_width = np.pi / electrode_count
    phase = np.pi / 2
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


class FenicsForwardModel:
    def __init__(self, mesh, subdomains, electrode_count, impedance, background_conductivity):
        self.mesh = mesh
        self.subdomains = subdomains
        self.electrode_count = electrode_count
        self.impedance = impedance
        self.background_conductivity = background_conductivity
     
        self.solution_space = self._solution_space()
        self.a = self._bilinear_form()
        
    def solve_forward(self, current_injection):
        ds = self._boundary_measure()
        
        (_, _, *V) = TestFunction(self.solution_space)
        
        L = 0*ds
        for i in range(self.electrode_count):
            area = assemble(1*ds(i+1))
            L += (current_injection[i] * V[i] / area) * ds(i+1)

        return self._solve(L)
    
    def solve_P(self, pertubation, y):
        dx = self._domain_measure()
        
        (v, _, *V) = TestFunction(self.solution_space)
        
        L = inner(nabla_grad(y), nabla_grad(v))*pertubation*dx
        y, Y = self._solve(L)
        return y, Y
    
    def _solution_space(self):
        H = self._interior_potential_space()
        R = FunctionSpace(self.mesh, "R", 0)

        mixed = H.ufl_element()
        for i in range(self.electrode_count + 1):
            mixed *= R.ufl_element()

        return FunctionSpace(self.mesh, mixed)
        
    def _interior_potential_space(self):
        return FunctionSpace(self.mesh, "CG", 1)

    def _domain_measure(self):
            dx = Measure('dx', domain=self.mesh)
            return dx

    def _boundary_measure(self):
            ds = Measure('ds', domain=self.mesh, subdomain_data=self.subdomains)
            return ds

    def _bilinear_form(self):
            # Define trial and test functions
            (u, p, *U) = TrialFunction(self.solution_space)
            (v, q, *V) = TestFunction(self.solution_space)

            dx = self._domain_measure() 
            ds = self._boundary_measure()
            a = self.background_conductivity * inner(nabla_grad(u), nabla_grad(v)) * dx
            for i in range(self.electrode_count):
                a += 1/self.impedance[i] * (u - U[i]) * (v - V[i]) * ds(i + 1)

                # Enforce mean free electrode potentials
                area = assemble(1*ds(i + 1))
                a += (q*U[i] + p*V[i])/area*ds(i + 1)

            return a

    def _rhs(self, current_injection, F = 0):
        
        (_, _, *V) = TestFunction(self.solution_space)
        
        dx = self._domain_measure()
        ds = self._boundary_measure()
        L = F * dx
        for i in range(self.electrode_count):
            area = assemble(1*ds(i+1))
            L += (current_injection[i] * V[i] / area) * ds(i+1)

        return L

    def _solve(self, rhs):
        w = Function(self.solution_space)
        solve(self.a == rhs, w)
        
        x = w.vector().get_local()
        U = x[-self.electrode_count:]
        
        # TODO: Find better way to split mixed function
        H = self._interior_potential_space()
        u = Function(H)
        u.vector().set_local(x[:-(self.electrode_count+1)])
        return u, U

# mesh, subdomains = create_disk_mesh(radius, 32, 300, 50)
# dx = _domain_measure(mesh)
# ds = _boundary_measure(mesh, subdomains)
# solution_space = _solution_space(mesh, electrode_count)
# a = _bilinear_form(solution_space, dx, ds, electrode_count, 1.0, np.full(electrode_count,1e-6))
# L = _rhs(solution_space, dx, ds, electrode_count, CURRENT_INJECTIONS[:,0])

# v,V = _solve(solution_space, a, L, mesh)
# print(v.vector().get_local().shape)
# print(len(mesh.coordinates()))
# plot(v)
# plt.show()

# xdmf = XDMFFile("subdomains.xdmf")
# xdmf.write(subdomains)
