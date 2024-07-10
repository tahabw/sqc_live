import numpy as np
import cupy as cp
import scipy as sc
import cupyx.scipy.linalg as ns_linalg
import cupyx.scipy.sparse as sparse
import scipy.special as special

# Constants:
h = 6.626e-34
hbar = h/(2*np.pi)
e = 1.602e-19
phi0 = h/(2*e)
phi0bar = phi0/(2*np.pi)
gap = 0.00017*e # from Kittel
Zq = phi0bar/(2*e)
kb = 1.38e-23

def progress(pos, tot):
    if pos % int(tot/10) == 0:
        print('.', end='')

# Conversion helpers
def cap(EC):
    EC = EC*1e9*h
    return e**2/(2*EC)

def ind(EL):
    EL = EL*1e9*h
    return phi0bar**2/EL


def enc(C):
    C = C*1e9*h
    return e**2/(2*C)


def enl(L):
    L = L*1e9*h
    return phi0bar**2/L
    
def phizpf(L, C):
    """ Phase-difference ZPF 
    Parameters: inductance L and capacitance C
    """
    EL = enl(L)
    EC = enc(C)
    return (2*EC/EL)**0.25


def Nzpf(L, C):
    """ Cooper-Pair number ZPF 
    Parameters: inductance L and capacitance C
    """
    EL = enl(L)
    EC = enc(C)
    return 0.5*(0.5*EL/EC)**0.25

# Helpers for sparse operators

def normalize(v):
    norm = cp.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def matel(x, i, j):
    """ 
    Helper to compute matrix elements of the cos or sin operator

    Parameters:
    x: argument of the cos/sin
    i,j: indices of the matrix element

    """
    n = min(i, j)
    m = max(i, j)
    factor = (-0.5)**((m-n)/2.) \
            *1./np.sqrt(special.poch(n+1,m-n)) \
            *x**(m-n) \
            *np.exp(-0.25*x**2) \
            *special.eval_genlaguerre(n, m-n, 0.5*x**2)
    return factor


def cosx(p, i, j):
    """ 
    Matrix element of the cos operator

    Parameters:
    p: zpf of operator
    i,j: indices of the matrix element

    """
    if (i-j)%2 == 0:
        return matel(np.sqrt(2)*p, i, j)
    else:
        return 0
    
    
def sinx(p, i, j):
    """ 
    Matrix element of the sin operator

    Parameters:
    p: zpf of operator
    i,j: indices of the matrix element

    """
    if (i-j)%2 == 1:
        comp = -1j*matel(np.sqrt(2)*p, i, j)
        if abs(np.imag(comp)) > 1e-10:
            raise ValueError("Matrix element is complex")
        else:
            return np.real(comp)
    else:
        return 0
    
def matrices_trig_sparse(FOCK, EC, EL):
    """ 
    Matrix element of the sin operator

    Parameters:

    FOCK: dimension of op
    EL, EC: inductive and charging energy

    """

    zpf = phizpf(ind(EL), cap(EC))

    COS = np.zeros((FOCK,FOCK), dtype=np.cfloat)
    SIN = np.zeros((FOCK,FOCK), dtype=np.cfloat)
    for i in range(FOCK):
        for j in range(FOCK):
            COS[i,j] += cosx(zpf, i, j)
            SIN[i,j] += sinx(zpf, i, j)
    COSsparse = sparse.dia_matrix(sc.sparse.dia_array(COS))
    SINsparse = sparse.dia_matrix(sc.sparse.dia_array(SIN))

    return COSsparse, SINsparse

def Q(dim,EL,EC):

    Qr = cp.zeros((dim,dim), dtype=cp.cfloat)
    for i in range(dim):
        for j in range(dim):
            if i == j-1:
                Qr[i,j] -= 1j*0.5*(EL/(2*EC))**0.25*cp.sqrt(j)
            elif i == j+1:
                Qr[i,j] += 1j*0.5*(EL/(2*EC))**0.25*cp.sqrt(i)
    return Qr


# Helpers to avoid loops over elements

def destroy(N):
    data = cp.sqrt(cp.arange(1, N, dtype=complex))
    ind = cp.arange(1,N, dtype=np.int32)
    ptr = cp.arange(N+1, dtype=np.int32)
    ptr[-1] = N-1
    return sparse.csr_matrix((data,ind,ptr),shape=(N,N))

def create(N):
    data = cp.sqrt(cp.arange(1, N, dtype=complex))
    ind = cp.arange(1,N, dtype=np.int32)
    ptr = cp.arange(N+1, dtype=np.int32)
    ptr[-1] = N-1
    return sparse.csr_matrix((data,ind,ptr),shape=(N,N)).transpose()
    
def position(N):
    return cp.sqrt(2)/2 * (destroy(N)+create(N))

def momentum(N):
    return 1j*cp.sqrt(2)/2 * (create(N)-destroy(N))

def cosm(op):
   return .5*(ns_linalg.expm(1j*op.toarray()) + ns_linalg.expm(-1j*op.toarray()))

def sinm(op):
    return -.5j*(ns_linalg.expm(1j*op.toarray()) - ns_linalg.expm(-1j*op.toarray()))

