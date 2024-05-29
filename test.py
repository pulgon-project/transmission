import spglib
from ase.io.vasp import read_vasp
from ipdb import set_trace
from sympy import symbols
import sympy
import numpy as np


n, k, m, s, jj, t, alpha = symbols("n k m s jj t alpha")

E = sympy.exp(1j*k*t/2) * sympy.Matrix(
    [[sympy.exp(1j*m*t*alpha/2), 0],
     [0,sympy.exp(-1j*m*t*alpha/2)]
    ]) @ sympy.Matrix(
    [[0, 1],
     [1, 0]
    ]) @ sympy.Matrix(
    [[sympy.exp(1j*m*s*alpha), 0],
     [0,sympy.exp(-1j*m*s*alpha)]
    ])


E2 = sympy.exp(1j*k*t) * sympy.Matrix(
    [[sympy.exp(1j*m*t*alpha/2), 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, sympy.exp(-1j*m*t*alpha)]
    ]) @ sympy.Matrix(
    [[0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [1, 0, 0, 0]
    ]) @ sympy.Matrix(
    [[sympy.exp(2j*m*s*alpha), 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, sympy.exp(-1j*m*s*alpha)]
    ])


E = E.subs({t:0})
E2 = E2.subs({t:0})

A = sympy.exp(1j*k*t/2)
A = A.subs({t:0})

B = (-1)**(jj) * sympy.exp(1j*k*t/2)
B = B.subs({t:0})

DE = sympy.Matrix(np.outer(E.conjugate(), E2))
# DE = DE.subs({t: 0})

DA = sympy.Matrix(np.outer(A.conjugate(), E2))
DB = sympy.Matrix(np.outer(B.conjugate(), E2))
# DA = DA.subs({t: 0})


set_trace()


