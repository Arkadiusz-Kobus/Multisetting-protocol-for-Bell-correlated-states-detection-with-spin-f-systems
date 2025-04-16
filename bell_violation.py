import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize

f = 3 # spin quantum number
N = 26400 # atom number
nt = 100 # number of time steps
tf = 5 # final time
t = np.linspace(0, tf, nt) # time range

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

#%% calculating spin length and squeezing parameters

state_vec = np.load('data.npy')

results = {}
for m in range(1, f + 1):
    results[m] = [squeezing(m, N, state) for state in state_vec]
    
#%% calculating Bell correlator
    
def L(k, w, xi, v):
    def cost(theta, k, xi, v):
        return k**2 + 4*v*np.cos(theta) @ np.arange(1 - k, k, 2) - (1 - 4*v*xi)*sum(np.sin(theta))**2
    return sum(w[mu]*minimize(lambda theta: cost(theta, k[mu], xi[mu], v[mu]),\
                              np.linspace(0, np.pi, k[mu]), bounds = [(0, np.pi)]*k[mu])['fun'] for mu in range(1, f + 1))/\
        sum(w[mu]*k[mu]**2 for mu in range(1, f + 1))
        
k = {mu: 3 for mu in range(1, f + 1)}
w = {mu: 1 for mu in range(1, f + 1)}

correlator = [L(k, w, {mu: results[mu][i][0] for mu in range(1, f + 1)},\
              {mu: results[mu][i][1] for mu in range(1, f + 1)}) for i in range(nt)]
    
pl.rcParams['font.size'] = 13
pl.rcParams['font.serif'] = "Times"
pl.rcParams['mathtext.fontset'] = 'cm'

pl.plot(t, correlator, c = "black")
pl.xlabel(r"Time of evolution $[N\hbar/c_1]$")
pl.ylabel(r"$L_{opt}/E_{max}$")
pl.xlim([0, tf])
pl.ylim([-0.3, 0.3])
pl.fill_between([0,tf], [-0.25,-0.25], [-0.3, -0.3], color = "black", alpha = 0.1)
print(min(correlator))
pl.show()