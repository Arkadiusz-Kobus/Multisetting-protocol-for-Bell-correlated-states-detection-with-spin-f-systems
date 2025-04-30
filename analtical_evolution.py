import numpy as np
import matplotlib.pyplot as pl
from scipy.linalg import eigh
from itertools import permutations
from sympy.physics.quantum.cg import CG

epsilon = 1e-9
N = 5 # number of atoms
s = 3 # quanum spin number

# generates all partitions of integer n into up to k integers, all of which do not exceed m
def partitions(n, m, k):
        if m >= n and k > 0: yield [n]
        for e in range(n - 1 if m >= n else m, 0, -1):
            for p in partitions(n - e, e, k - 1):
                yield p + [e]
               
# our basis consists of all possible divisions of N atoms into 2s+1 Zeeman states
basis = {}
index = 0
for p in partitions(N, N, 2*s + 1):
    p = p + [0]*(2*s + 1 - len(p))
    for q in permutations(p):
        if q not in basis:
            basis[q] = index
            index += 1
dim = len(basis)
       
# we use initial state with whole populaaiton in 0 Zeeman state
initial_state = np.zeros(dim, dtype = complex)
initial_state[basis[tuple([0]*s + [N] + [0]*s)]] = 1
   
def S(k, n): # generates matrix of a_k\dagger*a_n operator
    result = np.zeros((dim, dim), dtype = complex)
    for initial in basis:
        final = list(initial)
        final[k + s] += 1
        final[n + s] -= 1
        if final[n + s] >= 0:
            result[basis[tuple(final)], basis[initial]] = np.sqrt(initial[n + s]*final[k + s])
    return result

def herm(A):
    return A + A.T.conj()

def expected_value(observable, state):
    return (state.conj().T @ observable @ state).real

def variance(observable, state):
    return ((state.conj().T @ observable) @ (observable @ state)).real - \
        expected_value(observable, state)**2

def covariance(A, B, state):
    return ((state.conj().T @ A) @ (B @ state)).real - expected_value(A, state)*expected_value(B, state)

def dot(u, v): return sum(p[0]*p[1] for p in zip(u,v))
 
# we diagonalise hamiltonian and calculate evolution of initial state
D = None
P = None
def evolution(initial_state, hamiltonian, dt, t_min, t_max, recalculate = False):
    global P, D
    if (D is None and P is None) or recalculate:
        eigenvalues, eigenvectors = eigh(hamiltonian, driver = "evd", \
                                         check_finite = False, overwrite_a = True)
        D, P = np.diag(eigenvalues), eigenvectors
    state = P.T.conj() @ initial_state
    state = np.array([np.exp(-1j*t_min*D[i,i])*state[i] for i in range(dim)])
    infinitesimal_evolution = [np.exp(-1j*dt*D[i,i]) for i in range(dim)]
    current_time = t_min
    while current_time < t_max:
        in_our_basis = P @ state
        in_our_basis /= np.linalg.norm(in_our_basis)
        yield in_our_basis
        current_time += dt
        state = np.array([infinitesimal_evolution[i]*state[i] for i in range(dim)])
        state /= np.linalg.norm(state)

# we calculate spin squezzing parameter and spin length for entire evolution
def compute_parameters(data, symmetric = True):

    if symmetric: 
        Sx = {m: (S(0,m) + S(m,0) + S(0,-1*m) + S(-1*m,0))/2/np.sqrt(2) for m in range(1, s + 1)}
        Sy = {m: -1j*(S(0,m) - S(m,0) + S(0,-1*m) - S(-1*m,0))/2/np.sqrt(2) for m in range(1, s + 1)}
        Sz = {m: -1*((S(m,m) + S(-1*m,-1*m) + S(m,-1*m) + S(-1*m,m))/2 - S(0,0))/2 for m in range(1, s + 1)}
    else: 
        Sx = {m: (S(0,m) + S(m,0) - S(0,-1*m) - S(-1*m,0))/2/np.sqrt(2) for m in range(1, s + 1)}
        Sy = {m: -1j*(S(0,m) - S(m,0) - S(0,-1*m) + S(-1*m,0))/2/np.sqrt(2) for m in range(1, s + 1)}
        Sz = {m: -1*((S(m,m) + S(-1*m,-1*m) - S(m,-1*m) - S(-1*m,m))/2 - S(0,0))/2 for m in range(1, s + 1)}
        
    S_vec = {m: (Sx[m], Sy[m], Sz[m]) for m in range(1, s + 1)}
    squeeze = {m: [] for m in range(1, s + 1)}
    v = {m: [] for m in range(1, s + 1)}
    
    for state in data:
        for m in range(1, s + 1):
            avg_spin = [expected_value(spin_operator, state) for spin_operator in S_vec[m]]
            v[m].append(np.linalg.norm(avg_spin)/N)
            direction = avg_spin/np.linalg.norm(avg_spin)
            if abs(np.dot(direction, (0,0,1))) > 1 - epsilon: S1, S2 = S_vec[m][0], S_vec[m][1]
            else:
                orth_vec = np.cross((0,0,1), direction)
                orth_vec /= np.linalg.norm(orth_vec)
                S1, S2 = dot(orth_vec, S_vec[m]), dot(np.cross(direction, orth_vec), S_vec[m])
            V1, V2, Cov = variance(S1, state), variance(S2, state), covariance(S1, S2, state)
            min_variance = (V1 + V2 - np.sqrt((V1 - V2)**2 + 4*Cov**2))/2
            squeeze[m].append(N*min_variance/np.linalg.norm(avg_spin)**2)
       
    return (v, squeeze)

#%%

def Clebsch(f1,m1,f2,m2,f3,m3):
    return float(CG(f1,m1,f2,m2,f3,m3).doit())

alpha = [np.sqrt(s*(s + 1) + m*(1 - m)) for m in range(1, s + 1)]
J_plus = dot([S(m, m - 1) + S(1 - m, -1*m) for m in range(1, s + 1)], alpha)
J_minus = J_plus.T.conj()
Jz = dot([S(m,m) for m in range(-1*s, s + 1)], range(-1*s, s + 1))
J2 = Jz @ Jz + J_minus @ J_plus + J_plus @ J_minus

A = S(0, 0) @ S(0, 0) +\
    2*herm(dot([S(0, m) @ S(0, -1*m) for m in range(1, s + 1)], [(-1)**m for m in range(1, s + 1)])) +\
    4*dot([S(m, n) @ S(-1*m, -1*n) for m in range(1, s + 1) for n in range(1, s + 1)],\
          [(-1)**(m + n) for m in range(1, s + 1) for n in range(1, s + 1)])
A /= (2*s + 1)

B = 0
if s == 3:
    for M in range(-2, 3):
        for m in range(-3, 4):
            for n in range(-3, 4):
                C = Clebsch(3, m, 3, M - m, 2, M)*Clebsch(3, n, 3, M - n, 2, M)
                if abs(C) > epsilon: B += C*S(m, n) @ S(M - m, M - n)

c2 = 1
c3 = 30
hamiltonian = (J2/N/2 + c2*A/N/2 + c3*B/N/2)

#%%

t_min = 0
t_max = 0.2
t_len = 50
dt = (t_max - t_min)/t_len

current_time = t_min
time_len = 0
while current_time < t_max:
    time_len += 1
    current_time += dt

time = list(np.linspace(t_min, t_max, time_len))

data = evolution(initial_state, hamiltonian, dt, t_min, t_max)
v, xi = compute_parameters(data)

#%%

pl.rcParams['font.size'] = 13
pl.rcParams['font.serif'] = "Times"
pl.rcParams['mathtext.fontset'] = 'cm'

for m in range(1, s + 1): pl.plot(time, xi[m])
pl.ylim([0.,1.1])
pl.xlabel(r"Time of evolution $[\hbar/c]$")
pl.ylabel(r"Squeezing parameter $\xi^2$")
pl.legend([r"$\mu=" + str(m) + r"$" for m in range(1, s+1)])
pl.plot([t_min, t_max], [1]*2, c = "black", alpha = 0.25)
