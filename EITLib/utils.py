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
from scipy.sparse import diags

def create_disk_mesh(radius, n, F):
    center = Point(0, 0)
    domain = Circle(center, radius, n)
    mesh = generate_mesh(domain, F)
    return mesh

class  EITFenics:
    def __init__(self, mesh, L=32, background_conductivity=0.8):
        self.L = L

        impedance_scalar = 1e-6
        self.impedance = []
        for i in range(self.L):
            self.impedance.append(impedance_scalar)

        self.background_conductivity = background_conductivity
        self.mesh = mesh
        
        self._build_subdomains()
        self.V, self.dS = self.build_spaces(self.mesh, L, self.subdomains)
        self._build_D_sub()
        self.B_background = self.build_b(self.background_conductivity, self.V, self.dS, L)

    def _build_D_sub(self):
        L = self.L
        self.D_sub = diags([-np.ones(L-1), np.ones(L)], [1, 0]).toarray()[:-1, :]

    def _build_subdomains(self):
        L = self.L
        e_l = np.pi / L
        d_e = 2*np.pi / L - e_l

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

        for i in range(1, L+1):
            shift_theta = np.pi/2 - np.pi/(2*L)
            # print "shift_theta", shift_theta
            # print L
            theta1 = np.mod((i - 1) * (e_l+d_e) + shift_theta, 2*np.pi)
            theta2 = np.mod(theta1+e_l, 2*np.pi)
            # print i
            # print theta1
            # print theta2
            e1 = e()  # create instance
            e1 .mark(self.subdomains, i)  # mark subdomains
            xdmf = XDMFFile("subdomains.xdmf")
            xdmf.write(self.subdomains)
    def _create_inclusion(self, phantom):

        plt.figure()
        im = plt.imshow(phantom)
        plt.colorbar(im)  # norm= 'log'
        plt.savefig("phantom.png")

        self.inclusion = Inclusion(phantom, degree=0)

    def evaluate_target_external(self, injection_patterns, sigma_values, u_measure, compute_grad=None, num_inj_tested=None):
        print("* evaluate_target_external called")
        self.inclusion = Function(self.H_sigma)
        self.inclusion.vector().set_local(sigma_values)
        _, _, q_list = self.solve_forward(injection_patterns, self.inclusion, num_inj_tested)

        if compute_grad:
            v_list = self.solve_adjoint(q_list, self.inclusion, u_measure)
            grad = self.evaluate_gradient(q_list, v_list).vector()[:]
        else:
            grad = None
        
        return self.self.evaluate_target_functional(q_list, u_measure), grad


    def solve_forward(self, injection_patterns, phantom=None, num_inj_tested=None):

        # phantom can be FEniCS function
        if not isinstance(phantom, np.ndarray): 
            self.inclusion = phantom
        # phantom can be numpy array
        elif phantom is not None:
                self._create_inclusion(phantom)

        L = self.L

        # Define H1 room
        H1 = FunctionSpace(self.mesh, 'CG', 1)

        # Loop over current patterns
        num_inj = 76  # Number of injection pattern
        # num_inj_tested = 76

        B = self.B_background if phantom is None else self.build_b(self.inclusion, self.V, self.dS, L)

        Q = np.zeros((L, num_inj))
        Diff = np.zeros((L-1, num_inj))
        q_list = []
        print("solve forward")
        for i in range(num_inj)[:num_inj_tested]:
            #print("injection pattern"+str(i))
            Q_i, q = self.solver(injection_patterns[:, i], B, self.V, self.dS, L)
            q_list.append(q)

            Q[:, i] = Q_i
            Diff[:, i] = np.diff(Q_i)
        print("end solve forward")
        Uel_sim = -Diff.flatten(order='F')
        return Uel_sim, Q, q_list
    
    def solve_adjoint(self, q_list, phantom, u_measure):

        # phantom can be FEniCS function
        if not isinstance(phantom, np.ndarray): 
            self.inclusion = phantom
        # phantom can be numpy array
        elif phantom is not None:
                self._create_inclusion(phantom)

        L = self.L

        # Define H1 room
        H1 = FunctionSpace(self.mesh, 'CG', 1)

        B_transpose = self.build_b_adjoint(self.inclusion, self.V, self.dS, L)

        v_list = []
        print("solve adjoint")
        for i, q in enumerate(q_list):
            rhs_sub =-self.D_sub.T@(self.D_sub@q.vector().get_local()[:L] - u_measure[i*(L-1):(i+1)*(L-1)] )
            rhs = Function(self.V).vector()

            rhs.set_local(np.concatenate((rhs_sub, np.zeros(self.V.dim()-L))))

            # print("injection pattern"+str(i))
            v = Function(self.V)
            solve(B_transpose, v.vector(), rhs)
            v_list.append(v)

        return v_list
    
    def evaluate_gradient(self, q_list, v_list):

        # loop over q_list and v_list
        grad = Function(self.H_sigma)
        grad.vector().zero()
        L = self.L
        sigma = TestFunction(self.H_sigma)
        for i, (q, v) in enumerate(zip(q_list, v_list)):
            grad.vector().axpy( 1, assemble(sigma * inner(nabla_grad(q[L]), nabla_grad(v[L]))*dx ))
            # print(i)
            #plt.figure()
            #plt.plot( assemble(sigma * inner(nabla_grad(q[L]), nabla_grad(v[L]))*dx )[:100])
        
        return grad
        
    def evaluate_target_functional(self, q_list, u_measure):

        L = self.L
        J = 0

        for i, q in enumerate(q_list):
            J +=0.5 * np.linalg.norm(self.D_sub@q.vector().get_local()[:L] - u_measure[i*(L-1):(i+1)*(L-1)] )**2

        return J


    def solve_P(self, y_list, sigma_perturb):
        L = self.L

        # Define H1 room
        H1 = FunctionSpace(self.mesh, 'CG', 1)
        B = self.B_background
        v = TestFunction(self.V)

        w_list = []
        for y in y_list:

            f = -sigma_perturb * inner(nabla_grad(y[L]), nabla_grad(v[L])) * dx 

            rhs = assemble(f)

            # Compute solution
            w = Function(self.V)
            solve(B, w.vector(), rhs)
            w_list.append(w)

        return w_list




    def build_subdomains(self, L, mesh):
        def twopiarctan(x):
            val = np.arctan2(x[1], x[0])
            if val < 0:
                val = val+2*np.pi
            return val
    
        e_l = np.pi / L
        d_e = 2*np.pi / L - e_l
    
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
    
        for i in range(1, L+1):
            shift_theta = np.pi/2 - np.pi/(2*L)
            # print "shift_theta", shift_theta
            # print L
            theta1 = np.mod((i - 1) * (e_l+d_e) + shift_theta, 2*np.pi)
            theta2 = np.mod(theta1+e_l, 2*np.pi)
            # print i
            # print theta1
            # print theta2
            e1 = e()  # create instance
            e1 .mark(subdomains, i)  # mark subdomain
    
        return subdomains
    
    def build_spaces(self, mesh, L, subdomains):
        R = FunctionSpace(mesh, "R", 0)
        H1 = FunctionSpace(mesh, "CG", 1)
        self.H_sigma = FunctionSpace(mesh, "DG", 0)
    
        spacelist = None
    
        for i in range(1, L+1):
    
            if i == 1:
                spacelist = R.ufl_element()
            else:
                spacelist *= R.ufl_element()
    
        spacelist *= H1.ufl_element()
        spacelist *= R.ufl_element()
    
        # Create function space
        V = FunctionSpace(mesh, spacelist)
    
        # Define new measures associated with the boundaries
        dS = Measure('ds', domain=mesh, subdomain_data=subdomains)
    
        return V, dS
    
    
    def build_b(self, sigma, V, dS, L):
    
        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)
    
        B = sigma * inner(nabla_grad(u[L]), nabla_grad(v[L])) * dx
    
        for i in range(L):
            B += 1/self.impedance[i] * (u[L]-u[i])*(v[L]-v[i]) * dS(i + 1)
            #TODO: check if this is correct for P operator
            B += (v[L+1]*u[i] / assemble(1*dS(i+1))) * dS(i+1)
            B += (u[L+1]*v[i] / assemble(1*dS(i+1))) * dS(i+1)
    
        return assemble(B)
    
    def build_b_adjoint(self, sigma, V, dS, L):
    
        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)
    
        B = sigma * inner(nabla_grad(u[L]), nabla_grad(v[L])) * dx
    
        for i in range(L):
            B += 1/self.impedance[i] * (u[L]-u[i])*(v[L]-v[i]) * dS(i + 1)
            #TODO: check if this is correct for P operator
            B += (v[L+1]*u[i] / assemble(1*dS(i+1))) * dS(i+1)
            B += (u[L+1]*v[i] / assemble(1*dS(i+1))) * dS(i+1)
    
        return assemble(adjoint(B))
    
    
    def solver(self, I, B, V, dS, L):  # sigma ,L, I , Z ,mesh, subdomains )
       # def 2 pi function
    
        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)
    
        f = 0*dS(1)
    
        for i in range(L):
            f += (I[i] * v[i] / assemble(1*dS(i+1))) * dS(i+1)
    
        rhs = assemble(f)
    
        # Compute solution
        q = Function(V)
        solve(B, q.vector(), rhs)
    
        Q = q.vector().get_local()[:L]
    
        return Q, q
    

class Inclusion(UserExpression):
    def __init__(self, phantom, **kwargs):
        super().__init__(**kwargs)
        x_grid = np.linspace(-1, 1, 256)
        y_grid = np.linspace(-1, 1, 256)
        self._interpolater = RegularGridInterpolator(
            (x_grid, y_grid), phantom, method="nearest")

    def eval(self, values, x):
        values[0] = self._interpolater([x[0], x[1]])
