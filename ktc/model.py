import glob
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from itertools import product

from mshr import *
from dolfin import Point, SubDomain, Cell
from dolfin import Function, FunctionSpace, MeshFunction, TrialFunction, TestFunction
from dolfin import Measure, inner, nabla_grad, assemble, project
from dolfin import solve, plot

def create_disk_mesh(radius, electrode_count, polygons = 300, fineness = 50):
    center = Point(0, 0)
    domain = Circle(center, radius, polygons)
    mesh = generate_mesh(domain, fineness)
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
            dot = np.clip(np.dot(x, [u, v])/r,-1,1)
            rho = np.arccos(dot)
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
    def _mark_partitions(self, reconstruction_mesh):
        class Partition(SubDomain):
            def __init__(self, reconstruction_mesh, cell_index):
                    super().__init__()
                    self.cell_index = cell_index
                    self.reconstruction_mesh
                    
            def inside(self, x, _):
                point = Point(x)
                cell = Cell(self.reconstruction_mesh, self.cell_index)
                return cell.contains(point)

        topology = self.mesh.topology()
        subdomains = MeshFunction('size_t', self.mesh, dim=topology.dim())
        for i in range(reconstruction_mesh.num_cells()):
            partition = Partition(i)
            partition.mark(subdomains, i)
            
        return subdomains
    
    def _basis(self, n, W, H):
        # TODO: Verify that the mesh representation is correct
        
        en = np.zeros(W.dim())
        en[n] = 1.0
        
        chi = Function(W)
        chi.set_allow_extrapolation(True)
        chi.vector().set_local(en)
        proj_chi = project(chi,H)
        # proj_chi = proj_chi.vector().set_local(np.around(proj_chi.vector().get_local()))
        # TODO: Verify difference is insignificant
        return proj_chi

    def _compute_coefficients(self, u, W, H):
        N = W.dim()
        J = len(u)
        
        dx = self._domain_measure()
        coeffs = np.zeros((J,J,N))
        for n in range(N):
            print("Compute coefficient n=%d"%(n))
            for (i,j) in product(range(J),range(J)):
                chi = self._basis(n, W, H)
                integrand = inner(nabla_grad(u[i]),nabla_grad(u[j]))*chi
                coeffs[i,j,n] = assemble(integrand*dx)
                
        return coeffs
        
    def __init__(self, mesh, subdomains, electrode_count, impedance, conductivity):
        self.mesh = mesh
        self.subdomains = subdomains
        self.electrode_count = electrode_count
        self.impedance = impedance
        self.conductivity = conductivity
     
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
    
    def gradient(self, W):
        H = self._interior_potential_space()
        
        pass
    
    def _gradient_inner_product(self, u, v, B):
        dx = self._domain_measure()
        integrand = inner(nabla_grad(u), nabla_grad(v))*B
        return assemble(integrand*dx)
    
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
            a = self.conductivity * inner(nabla_grad(u), nabla_grad(v)) * dx
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