import spglib
from ase.io.vasp import read_vasp
from ipdb import set_trace
from sympy import symbols
import sympy
import numpy as np
from sympy.physics.quantum import TensorProduct
import scipy

n, k, m, s, jj, t, alpha = symbols("n k m s jj t alpha")

E = sympy.exp(1j*k*t/2) * sympy.Matrix(
    [[sympy.exp(1j*m*t*alpha/2), 0],
     [0,sympy.exp(-1j*m*t*alpha/2)]
    ]) @ sympy.Matrix(
    [[0, 1],
     [1, 0]
    ]) ** jj @ sympy.Matrix(
    [[sympy.exp(2j*m*s*alpha), 0],
     [0,sympy.exp(-2j*m*s*alpha)]
    ])


E2 = sympy.exp(1j*k*t) * sympy.Matrix(
    [[sympy.exp(1j*m*t*alpha/2), 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, sympy.exp(-1j*m*t*alpha/2)]
    ]) @ sympy.Matrix(
    [[0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [1, 0, 0, 0]
    ]) ** jj @ sympy.Matrix(
    [[sympy.exp(2j*m*s*alpha), 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, sympy.exp(-2j*m*s*alpha)]
    ])


E = E.subs({t:2, s:0, jj:0,k:0})
E2 = E2.subs({t:2, s:0, jj:0,k:0})


DE = TensorProduct(E.conjugate(), E2)


num = 4
PE = DE

A = sympy.diag(1,sympy.exp(-1j*m*alpha),sympy.exp(-1j*m*alpha),sympy.exp(-2j*m*alpha),sympy.exp(2j*m*alpha),sympy.exp(1j*m*alpha),sympy.exp(1j*m*alpha),1)

m = 1
n = 4

A = np.array(A.subs({"m":m, "alpha":2*sympy.pi/n}).conjugate()).astype(np.complex128)


r1 = TensorProduct(np.eye(3),np.eye(2))
r2 = TensorProduct(np.array([[1,0,0],[0,1,0],[0,0,-1]]), np.array([[0,1],[1,0]]))
res = (r1 + r2)/2

set_trace()


