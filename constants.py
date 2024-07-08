import numpy as np

e = 1.60217662e-19 # C
hbar = 1.0545718e-34 # m^2 * kg / s
h = 6.62607004e-34 # m^2 * kg / s
R_K = 25812.807557 # Ohm
Phi0 = 2.067833831e-15 # Wb
phi0 = Phi0 / (2. * np.pi) # Wb
k_B = 1.38064852e-23 # m^2 * kg / (s^2 * K)

C_J_unit_area = 45.e-15 # F / um^2
C_J_unit_area_fF = 45. # fF / um^2

Delta_Al_eV = 0.18e-3 # eV
Delta_Al = Delta_Al_eV * e

def GHz2fF(E_C):
    return 1.e6 * e**2. / (2. * E_C * h)

def fF2GHz(C):
    return 1.e6 * e**2. / (2. * C * h)
    
def GHz2nH(E_L):
    return phi0**2. / (E_L * h)

def nH2GHz(L):
    return phi0**2. / (L * h)
    
def uA2GHz(I0):
    return 1.e-15 * phi0 * I0 / h
    
def GHz2uA(E_J):
    return 1.e15 * E_J * h / phi0