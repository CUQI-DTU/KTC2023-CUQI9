"""
Note by Amal Alghamdi: This code is copied from the project report: Depth 
Dependency in Electrical Impedance Tomography with the Complete 
Electrode Model by Anders Eltved and Nikolaj Vestbjerg Christensen (Appendix D.5). Some 
modifications are made.
"""

import numpy as np
from dolfin import *


def build_subdomains(L, mesh):
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


def build_spaces(mesh, L, subdomains):
    R = FunctionSpace(mesh, "R", 0)
    H1 = FunctionSpace(mesh, "CG", 1)

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
    dS = Measure('ds', domain=mesh)[subdomains]

    return V, dS


def build_b(sigma, Z, V, dS, L):

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    B = sigma * inner(nabla_grad(u[L]), nabla_grad(v[L])) * dx

    for i in range(L):
        B += 1/Z[i] * (u[L]-u[i])*(v[L]-v[i]) * dS(i + 1)
        B += (v[L+1]*u[i] / assemble(1*dS(i+1))) * dS(i+1)
        B += (u[L+1]*v[i] / assemble(1*dS(i+1))) * dS(i+1)

    return assemble(B)


def solver(I, B, V, dS, L):  # sigma ,L, I , Z ,mesh, subdomains )
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
