import numpy as np
import matplotlib.pyplot as pl
from sympy import Symbol, diff, lambdify
from sympy.physics.quantum.cg import CG
from scipy.integrate import solve_ivp
from time import time
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize

f = 3 # spin quantum number
m_range = list(range(-f, f + 1)) # allowed spin levels
a = {m: Symbol(r"a_" + str(m)) for m in m_range} # anihilation operators
k = {m: Symbol(r"a^\dagger_" + str(m)) for m in m_range} # creation operators
for m in range(f + 1, 2*f + 1): a[m], a[-m], k[m], k[-m] = 0, 0, 0, 0 # we extend field operators with zeros if they do not correspond to valid spin levels, this simplifies some definitions

def cg(f1,m1,f2,m2,f3,m3): # the Clebsch–Gordan coefficients
    return float(CG(f1,m1,f2,m2,f3,m3).doit())

def A(F,M): # particle number operator for bosonic pair in |F,M> state
    return sum([cg(f,m,f,M-m,F,M)*k[m]*k[M-m] for m in m_range])*\
        sum([cg(f,m,f,M-m,F,M)*a[m]*a[M-m] for m in m_range])
       
def dH(H): # derivative of H over field operators
    return [-1j*diff(H, k[m]) for m in m_range]

def solve(t, N, step): # we generate initial random distribution of states corresponding to all atoms in 0 spin level
    psi0 = np.zeros(2*f + 1, dtype = complex)
    random_complex = np.random.normal(0, 0.5, (2*f + 1)) + 1j * np.random.normal(0, 0.5, (2*f + 1))
    psi0[:] = random_complex
    psi0[f] += np.sqrt(N)
    return solve_ivp(step, (0, tf), psi0, t_eval = t).y

# we generate J^2 operator
def alpha(m): return np.sqrt(f*(f + 1) + m*(1 - m))
J_plus = sum([alpha(m)*(k[m]*a[m - 1] + k[1 - m]*a[-m]) for m in range(1, f + 1)])
J_minus = sum([alpha(m)*(a[m]*k[m - 1] + a[1 - m]*k[-m]) for m in range(1, f + 1)])
Jz = sum([m*(k[m]*a[m] - k[-m]*a[-m]) for m in range(1, f + 1)])
J2 = Jz*Jz + J_minus*J_plus

def hamiltonian(c1, c2, c3): # BEC hamiltonian
    C2 = A(0, 0) if f > 1 else 0
    C3 = sum(A(2, M) for M in range(-2, 3)) if f > 2 else 0
    return (c1*J2 + c2*C2 + c3*C3)/2/N

def squeezing(m, N, psi): # we calculate average spin length and squeezing parameter
    a_0 = psi[f,:]
    g_s = (psi[f-m,:]+psi[f+m,:])/np.sqrt(2)
   
    Jx = np.mean(np.conj(g_s)*a_0 + np.conj(a_0)*g_s)/2
    Jy = 1j*np.mean(np.conj(g_s)*a_0 - np.conj(a_0)*g_s)/2
    Jz = np.mean(np.conj(a_0)*a_0 + np.conj(g_s)*g_s)/2
   
    Jx_var = 0.25 * (np.mean(np.conj(g_s)*np.conj(g_s)*a_0*a_0) +
                      np.mean(np.conj(a_0)*np.conj(a_0)*g_s*g_s) +
                      2*np.mean(np.conj(a_0)*np.conj(g_s)*a_0*g_s) - 0.5)
   
    Jy_var = 0.25 * (-np.mean(np.conj(g_s)*np.conj(g_s)*a_0*a_0) -
                      np.mean(np.conj(a_0)*np.conj(a_0)*g_s*g_s) +
                      2*np.mean(np.conj(a_0)*np.conj(g_s)*a_0*g_s) - 0.5)
    Cov = 1j*0.25*(np.mean(np.conj(g_s)*np.conj(g_s)*a_0*a_0) -
                    np.mean(np.conj(a_0)*np.conj(a_0)*g_s*g_s))
    
    # minimal variance is smallest eigenvalue of covariance matrix
    min_var = (Jx_var + Jy_var - np.sqrt((Jx_var - Jy_var)**2 + 4*Cov**2))/2
    ksi = np.real(4*min_var/N)
    v = np.linalg.norm([Jx, Jy, Jz])/N
    
    return (ksi, v)

def tangents(c1, c2, c3): # initial derivative of squeezing parameter
    if f == 1:
        return abs(c1)
    if f == 2:
        return abs(6*c1 - 2*c2/5), abs(2*c2/5)
    if f == 3:
        return abs(12*c1 - 2*c2/7 - 2*c3/7), abs(2*c2/7), abs(2*c2/7 - 10*c3/21)

#%% simulation parameters

N = 26400 # atom number
iters = 10**5 # number of trajectories
nt = 100 # number of time steps
tf = 5 # final time
t = np.linspace(0, tf, nt) # time range
c1, c2, c3 = 1, 21, 0 # hamiltonian coefficients

# we generate hamiltonian, its derivative and step function that substitutes trajectory parameters for field operators
H = hamiltonian(c1, c2, c3)
dh = dH(H.expand())
fun = lambdify([x for x in a.values() if x != 0] + [x for x in k.values() if x != 0], dh)
def step(t, psi):
    return fun(*psi, *np.conj(psi))

#%% solving equations of motion for each trajectory

if __name__ == '__main__':
    start_time = time()
    with Pool(processes = cpu_count()) as pool:
        state_vec = np.array(pool.starmap(solve, [(t, N, step) for _ in range(iters)])).T
    end_time = time()
    execution_time = end_time - start_time  
    print(f"Execution time: {execution_time} seconds")
    
    # np.save('data.npy', np.array(state_vec))

#%% calculating spin length and squeezing parameters

state_vec = np.load('data.npy')

results = {}
for m in range(1, f + 1):
    results[m] = [squeezing(m, N, state) for state in state_vec]

#%% plotting results
        
colors = ['blue', 'orange', 'green']
pl.rcParams['font.size'] = 13
pl.rcParams['font.serif'] = "Times"
pl.rcParams['mathtext.fontset'] = 'cm'

for m in range(1, f + 1):
    pl.plot(t, [x[0] for x in results[m]], c = colors[m - 1])
    
tan = tangents(c1, c2, c3)
pl.plot(t, [1 - tan[m - 1]*time for time in t], '--', c = "black")

pl.xlabel(r"Time of evolution $[N\hbar/c_1]$", fontsize = 13)
pl.legend([r"$\xi^2_" + str(m) + r"$" for m in range(1, f + 1)] +\
          [r"$1 - \beta_{1/2/3}t$" for m in range(1, f + 1)], loc = 4)
pl.yscale('log')
pl.ylim([10**-3, 2])
pl.text(1.4, 0.002, r"$c_2/c_1=21,\, c_3/c_1=0$")
pl.text(4.9, 1, "(a)")
pl.show()
