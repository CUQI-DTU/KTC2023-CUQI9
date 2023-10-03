"""
Note by Amal Alghamdi: This code is copied from the project report: Depth 
Dependency in Electrical Impedance Tomography with the Complete 
Electrode Model by Anders Eltved and Nikolaj Vestbjerg Christensen (Appendix D.5). Some 
modifications are made.
"""

import numpy as np 
from dolfin import *

def solver( sigma ,L, I , Z ,mesh ) :
 # def 2 pi function
  def twopiarctan( x ) :
    val=np.arctan2( x [ 1 ] , x [ 0 ] )
    if val <0:
      val=val+2*np.pi
    return val
  
  e_l=np.pi /L
  d_e=2*np.pi /L - e_l


  class thetavalues ( Expression ) :
    def eval( self , theta , x ) :
      theta[0]=np .arctan2 ( x [ 1 ] , x [ 0 ] )


  # Define subdomain mesh
  subdomains = FacetFunction( "size_t" , mesh )
  subdomains.set_all( 0 )

 # Define subdomains
  class e(SubDomain) :
    def inside( self , x , on_boundary ) :
      theta=twopiarctan ( x )
      return on_boundary and theta>=theta1 and theta<=theta2

  R = FunctionSpace (mesh , "R" , 0 )
  H1 = FunctionSpace (mesh , "CG" , 1 )

  spacelist = None

  for i in range ( 1 , L+1) :
    theta1 = ( i -1) *( e_l+d_e )
    theta2 = theta1+e_l
    e1 = e ( ) # create instance
    e1 .mark( subdomains , i ) # mark subdomain
    if i==1:
      spacelist=R.ufl_element()
    else:
      spacelist*=R.ufl_element()
  
  spacelist*=H1.ufl_element()
  spacelist*=R.ufl_element()

  # Create function space
  V = FunctionSpace(mesh, spacelist)

  # Define new measures associated with the boundaries
  dS = Measure ( 'ds' , domain=mesh ) [ subdomains ]

  # Define trial and test functions
  u = TrialFunction(V)
  v = TestFunction(V)


  f = 0*dS ( 1 )

  B = sigma * inner ( nabla_grad (u [L ] ) , nabla_grad ( v [L ] ) ) *dx
  for i in range (L) :
    B += 1/Z [ i ] * ( u [L]-u [ i ] )*( v [L]-v [ i ] ) *dS( i +1)
    B += ( v [L+1]*u [ i ] / assemble (1*dS ( 0 ) ) ) *dS ( 0 )
    B += (u [L+1]*v [ i ] / assemble (1*dS ( 0 ) ) ) *dS ( 0 )
    f += ( I [ i ] * v [ i ] / assemble (1*dS ( 0 ) ) ) *dS ( 0 )

  # Compute solution
  q = Function (V)
  solve(B == f , q )

  return q
