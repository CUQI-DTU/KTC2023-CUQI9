"""
Note by Amal Alghamdi: This code is copied from the project report: Depth
Dependency in Electrical Impedance Tomography with the Complete
Electrode Model by Anders Eltved and Nikolaj Vestbjerg Christensen (Appendix D.5). Some
modifications are made.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
from dolfin import *
from mshr import *

def create_disk_mesh(radius, n, F):
    center = Point(0, 0)
    domain = Circle(center, radius, n)
    mesh = generate_mesh(domain, F)
    return mesh

def _create_inclusion(phantom):
    high_conductivity = 1e1
    low_conductivity = 1e-2
    background_conductivity = background_conductivity
    # Define conductivity
    phantom_float = np.zeros(phantom.shape)
    phantom_float[phantom == 0] = background_conductivity
    phantom_float[np.isclose(phantom, 1, rtol=0.01)] = low_conductivity
    phantom_float[phantom == 2] = high_conductivity

    plt.figure()
    im = plt.imshow(phantom_float)
    plt.colorbar(im)  # norm= 'log'
    plt.savefig("phantom.png")

    self.inclusion = Inclusion(phantom_float, degree=0)

class Inclusion(UserExpression):
    def __init__(self, phantom, **kwargs):
        super().__init__(**kwargs)
        x_grid = np.linspace(-1, 1, 256)
        y_grid = np.linspace(-1, 1, 256)
        self._interpolater = RegularGridInterpolator(
            (x_grid, y_grid), phantom, method="nearest")

    def eval(self, values, x):
        values[0] = self._interpolater([x[0], x[1]])



class FenicsForwardModel:
    def __init__(self, electrode_count):
        self.electrode_count = electrode_count
        self.impedance = np.full(electrode_count, 1e-6)

        self.background_conductivity = 0.8
        self.mesh = create_disk_mesh(1, 300, 50)

        self._build_subdomains()
        self.B_background = self.build_b(self.background_conductivity, electrode_count)

    def _build_subdomains(self):
        electrode_count = self.electrode_count
        e_l = np.pi / electrode_count
        d_e = 2*np.pi / electrode_count - e_l

        # Define subdomain mesh
        self.subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 0)

        # Define subdomains
        def twopiarctan(x):
            val = np.arctan2(x[1], x[0])
            if val < 0:
                val = val+2*np.pi
            return val

        class e(SubDomain):
            def inside(self, x, on_boundary):
                theta = twopiarctan(x)
                # print "theta inside", theta
                if theta1 > theta2:
                    return on_boundary and ((theta >= 0
                                            and theta <= theta2) or (theta >= theta1
                                                                    and theta <= 2*np.pi))
                return on_boundary and theta >= theta1 \
                    and theta <= theta2
                # return  theta>=theta1 and theta<=theta2

        for i in range(1, electrode_count+1):
            shift_theta = np.pi/2 - np.pi/(2*electrode_count)
            # print "shift_theta", shift_theta
            # print electrode_count
            theta1 = np.mod((i - 1) * (e_l+d_e) + shift_theta, 2*np.pi)
            theta2 = np.mod(theta1+e_l, 2*np.pi)
            # print i
            # print theta1
            # print theta2
            e1 = e()  # create instance
            e1 .mark(self.subdomains, i)  # mark subdomains
            xdmf = XDMFFile("subdomains.xdmf")
            xdmf.write(self.subdomains)

    def solve_forward(self, injection_pattern):

        #electrode_count = self.electrode_count

        # Define vector of contact impedance
        # z = 10e-6  # 0.1 # Impedence


        # Define H1 room
        H = FunctionSpace(self.mesh, 'CG', 1)

        # Loop over current patterns
        num_inj = 76  # Number of injection pattern
        # num_inj_tested = 76
        B = self.B_background

        Q = np.zeros((electrode_count, num_inj))
        Diff = np.zeros((electrode_count-1, num_inj))
        q_list = []

        V = self._solution_space()
        ds = _boundary_measure()
        for i in range(num_inj)[:num_inj_tested]:
            print("injection pattern"+str(i))
            Q_i, q = self.solver(injection_patterns[:, i], B, V, ds, self.electrode_count)
            q_list.append(q)

            Q[:, i] = Q_i
            Diff[:, i] = np.diff(Q_i)

        Uel_sim = -Diff.flatten(order='F')
        return Uel_sim, Q, q_list

    def solve_P(self, y_list, sigma_perturb):
        V = self._solution_space

        electrode_count = self.electrode_count

        # Define H1 room
        H1 = FunctionSpace(self.mesh, 'CG', 1)
        B = self.B_background
        v = TestFunction(V)

        w_list = []
        for y in y_list:

            f = -sigma_perturb * inner(nabla_grad(y[electrode_count]), nabla_grad(v[electrode_count])) * dx

            rhs = assemble(f)

            # Compute solution
            w = Function(V)
            solve(B, w.vector(), rhs)
            w_list.append(w)

        return w_list




    def build_subdomains(self, electrode_count, mesh):
        def twopiarctan(x):
            val = np.arctan2(x[1], x[0])
            if val < 0:
                val = val+2*np.pi
            return val

        e_l = np.pi / electrode_count
        d_e = 2*np.pi / electrode_count - e_l

        # Define subdomain mesh
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

       # Define subdomains
        class e(SubDomain):
            def inside(self, x, on_boundary):
                theta = twopiarctan(x)
                # print "theta inside", theta
                if theta1 > theta2:
                    return on_boundary and ((theta >= 0
                                             and theta <= theta2) or (theta >= theta1
                                                                      and theta <= 2*np.pi))
                return on_boundary and theta >= theta1 \
                    and theta <= theta2
                # return  theta>=theta1 and theta<=theta2

        for i in range(1, electrode_count+1):
            shift_theta = np.pi/2 - np.pi/(2*electrode_count)
            # print "shift_theta", shift_theta
            # print electrode_count
            theta1 = np.mod((i - 1) * (e_l+d_e) + shift_theta, 2*np.pi)
            theta2 = np.mod(theta1+e_l, 2*np.pi)
            # print i
            # print theta1
            # print theta2
            e1 = e()  # create instance
            e1 .mark(subdomains, i)  # mark subdomain

        return subdomains

    def _solution_space(self):
        R = FunctionSpace(self.mesh, "R", 0)
        H1 = FunctionSpace(self.mesh, "CG", 1)

        spacelist = None

        for i in range(self.electrode_count):

            if i == 1:
                spacelist = R.ufl_element()
            else:
                spacelist *= R.ufl_element()

        spacelist *= H1.ufl_element()
        spacelist *= R.ufl_element()

        # Create function space
        V = FunctionSpace(mesh, spacelist)

        return V

    def _boundary_measure(self):
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.subdomains)
        return ds


    def build_b(self, sigma, electrode_count):

        V = self._solution_space()
        ds = self._boundary_measure()

        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        B = sigma * inner(nabla_grad(u[electrode_count]), nabla_grad(v[electrode_count])) * dx

        for i in range(electrode_count):
            B += 1/self.impedance[i] * (u[electrode_count]-u[i])*(v[electrode_count]-v[i]) * ds(i + 1)
            #TODO: check if this is correct for P operator
            B += (v[electrode_count+1]*u[i] / assemble(1*ds(i+1))) * ds(i+1)
            B += (u[electrode_count+1]*v[i] / assemble(1*ds(i+1))) * ds(i+1)

        return assemble(B)


    def solver(self, I, B, V, ds, electrode_count):
       # def 2 pi function

        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        f = 0 * ds(1)

        for i in range(electrode_count):
            f += (I[i] * v[i] / assemble(1*ds(i+1))) * ds(i+1)

        rhs = assemble(f)

        # Compute solution
        q = Function(V)
        solve(B, q.vector(), rhs)

        Q = q.vector().get_local()[:electrode_count]

        return Q, q
