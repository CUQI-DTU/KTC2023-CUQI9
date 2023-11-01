import numpy as np
from dolfin import (
    Function,
    FunctionSpace,
    Measure,
    MeshFunction,
    Point,
    SubDomain,
    TestFunction,
    TrialFunction,
    assemble,
    inner,
    nabla_grad,
    solve,
)
from mshr import Circle, generate_mesh
from dolfin import Cell


def create_disk_mesh(radius, electrode_count, polygons=300, fineness=50):
    """
    Create a mesh representation of a disk and subdomains representing the
    electrodes.

    The electrodes are evenly spaced on the boundary of the disk and the
    width of each electrode is `pi / n` where `n` is the electrode count.

    Parameters
    ----------
    radius : float
        The radius of the disk.

    electrode_count : int
        The number of electrodes on the boundary of the disk.

    polygons : int, optional.

    fineness : float, optional.

    Returns
    -------
    mesh : dolfin.Mesh
        A mesh representation of the disk.

    subdomians : dolfin.MeshFunction
        The subdomains mark the electrodes on the boundary of the disk counting
        anticlockwise from 12-o'clock. The first electrode is marked with 0.
    """

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

            # Compute the normalised projection of x onto (u, v)
            proj = np.clip(np.dot(x, [u, v]) / r, -1, 1)
            rho = np.arccos(proj)

            # Project the angle to the edge of the electrode
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
    """
    A FEniCS implementation of the forward model for EIT.

    Parameters
    ----------
    mesh : dolfin.Mesh
        The mesh to use.

    subdomains : dolfin.MeshFunction
        The subdomains of the mesh.

    electrode_count : int
        The number of electrodes.

    z : array_like
        The impedance of each electrode.

    sigma : dolfin.Function
        The conductivity of the mesh.

    """

    def __init__(self, mesh, subdomains, electrode_count, z, sigma):
        self.mesh = mesh
        self.subdomains = subdomains
        self.electrode_count = electrode_count
        self.z = z
        self.sigma = sigma

        self.solution_space = self._solution_space()
        self.a = self._bilinear_form()
    
    def _mark_partitions(self, recon_mesh):
        class Partition(SubDomain):
            def __init__(self, recon_mesh, cell_index):
                super().__init__()
                self.cell_index = cell_index
                self.recon_mesh = recon_mesh

            def inside(self, x, _):
                point = Point(x)
                cell = Cell(self.recon_mesh, self.cell_index)
                return cell.contains(point)

        topology = self.mesh.topology()
        subdomains = MeshFunction("size_t", self.mesh, dim=topology.dim())
        for i in range(recon_mesh.num_cells()):
            partition = Partition(recon_mesh, i)
            partition.mark(subdomains, i)

        return subdomains
    
    def gradient(self, recon_mesh, u, Mpat):
        M = recon_mesh.num_cells()
        H = self._interior_potential_space()
        Nn = H.dim()
        L = self.electrode_count
        Q = L-1
        K = len(u)
        
        phi = TestFunction(H)
        psi = TrialFunction(H)

        subdomains = self._mark_partitions(recon_mesh)
        dx = Measure("dx", self.mesh, subdomain_data=subdomains)
        
        C = np.vstack([np.ones(L-1), -np.identity(L-1)])
        # Ctilde = np.hstack([np.zeros((L, Nn+L-1)), C])
        # Mtilde = np.hstack([Mpat, Ctilde])
        A = assemble(self.a).array()
        Mtilde = np.hstack([np.zeros((np.shape(Mpat)[0], Nn)), Mpat @ C])
        print(A.shape)
        print(Mtilde.shape)
        Gamma = np.linalg.solve(A, Mtilde.T)
        Gammatilde = Gamma[:Nn]
        
        
        alphatilde_list = []
        for ui in u:
            alphatilde_list.append(ui.vector().get_local())
        alphatilde = np.array(alphatilde_list)
        
        # grad = np.zeros((L*K,0))
        grad_list = []
        for l in range(M):
            integrand = inner(nabla_grad(phi), nabla_grad(psi))
            dBsigma = assemble(integrand * dx(l)).array()
            
            # grad_list = np.hstack([grad, -Gammatilde.T @ dBsigma @ alphatilde.T])
            col = -Gammatilde.T @ dBsigma @ alphatilde.T
            grad_list.append(col.flatten(order = "F"))

        return np.array(grad_list).T
    
       
    def _unpack_solution(self, U):
        return np.concatenate([[-U.sum()], U]) 

    def solve_forward(self, current_injection):
        ds = self._boundary_measure()

        (_, *V) = TestFunction(self.solution_space)

        L = 0 * ds
        for i in range(1, self.electrode_count):
            area = assemble(1 * ds(i + 1))
            L += (current_injection[i] * V[i-1] / area) * ds(i + 1)


        area = assemble(1 * ds(0 + 1))
        for i in range(1, self.electrode_count):
            L -= (current_injection[0] * V[i-1] / area) * ds(0 + 1)

        u, U = self._solve(self.a, L)
        return u, self._unpack_solution(U)

    def solve_pertubation(self, pertubation, y):
        dx = self._domain_measure()

        (v, *V) = TestFunction(self.solution_space)

        L = inner(nabla_grad(y), nabla_grad(v)) * pertubation * dx
        y, Y = self._solve(self.a, L)
        return y, Y

    def _solution_space(self):
        H = self._interior_potential_space()
        R = FunctionSpace(self.mesh, "R", 0)

        mixed = H.ufl_element()
        for i in range(self.electrode_count - 1):
            mixed *= R.ufl_element()

        return FunctionSpace(self.mesh, mixed)

    def _interior_potential_space(self):
        return FunctionSpace(self.mesh, "CG", 1)

    def _domain_measure(self):
        dx = Measure("dx", domain=self.mesh)
        return dx

    def _boundary_measure(self):
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.subdomains)
        return ds

    def _bilinear_form(self):
        # Define trial and test functions
        (u, *U) = TrialFunction(self.solution_space)
        (v, *V) = TestFunction(self.solution_space)

        dx = self._domain_measure()
        ds = self._boundary_measure()

        # Construct system matrix a = [B, C; C^T, G]
        # Make B component
        B = self.sigma * inner(nabla_grad(u), nabla_grad(v)) * dx
        for i in range(self.electrode_count):
            B += 1 / self.z[i] * u * v * ds(i + 1)

        # Make C component
        C = 0 * dx
        for i in range(1, self.electrode_count):
            C += 1 / self.z[0] * (u * V[i-1] + v * U[i-1]) * ds(0 + 1)
            C -= 1 / self.z[i] * (u * V[i-1] + v * U[i-1]) * ds(i + 1)

        # Make G component
        G = 0 * dx
        for i in range(1, self.electrode_count):
            G += 1 / self.z[i] * (U[i-1] * V[i-1]) * ds(i + 1)
            for j in range(1, self.electrode_count):
                G += 1 / self.z[0] * (U[i-1] * V[j-1]) * ds(0 + 1)

        A = B + C + G

        return A

    def _solve(self, a, L):
        w = Function(self.solution_space)
        solve(a == L, w)

        x = w.vector().get_local()
        U = x[-(self.electrode_count - 1):]

        # TODO: Find better way to split mixed function
        H = self._interior_potential_space()
        u = Function(H)
        u.vector().set_local(x[: -(self.electrode_count - 1)])
        return u, U
