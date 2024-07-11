import numpy as np
from scipy.special import eval_hermite as hpoly
import scipy.sparse as sps
import Helpers_Erwan as he

import cupy as cp
# import matplotlib.pyplot as plt
import scipy as sc
import cupyx.scipy.sparse as sparse
from cupyx.scipy.sparse.linalg import eigsh
import scipy.special as special
import cupyx.scipy.special as special_cp
import time
import os

class Kite_Transmon_Erwan(object):
    """Helper class to build the full Kite_Transmon Hamiltonian""" 
    def __init__(self, dim, params):

        self.EC, self.EJ, self.EL, self.y, self.dC, self.dJ, self.dL = params

        self.DIM, self.fock, self.FOCK = dim
        self.Nc = 2* self.DIM + 1

        self.H = None
        self.ev = None
        self.evec = None

    def reset_cache(self):
        self.H = None
        self.ev = None
        self.evec = None
        
    def Nsigma_op(self):  
        """
        Charge operator of the sigma mode in the computational gauge in oscillator basis.

        """
        ECr, ELr = 0.5*self.EC, 2*self.EL # effective energy scale in computational gauge
        Qr = he.Q(self.fock,ECr,ELr)

        return cp.kron(cp.eye(self.Nc),cp.kron(Qr, cp.eye(self.FOCK)))

    def Ndelta_op(self):
        """
        Charge operator of the delta mode in the computational gauge in oscillator basis.

        """
        ECr, ELr = 0.5*self.EC, 2*self.EL # effective energy scale in computational gauge
        Qf = he.Q(self.FOCK,ECr,ELr)
                    
        return cp.kron(cp.eye(self.Nc),cp.kron(cp.eye(self.fock), Qf))


    def N_op(self, ng): 
        """
        Charge operator of the capacitance mode in the computational gauge in oscillator basis.

        """
        
        low = cp.ceil(ng - cp.double(self.Nc) / 2.)
        Qc = cp.diag( cp.linspace(low, low + self.Nc - 1, self.Nc) )
            
        return cp.kron(Qc,cp.kron(cp.eye(self.fock),cp.eye(self.FOCK)))

    def _Hamiltonian2_sparse(self, ng): #r: phisigma / f: phidelta

        ECc, ECr, ECf = self.y*self.EC, 0.5*self.EC, 0.5*self.EC
        ELr, ELf = 2*self.EL, 2*self.EL

        COSr_sparse, SINr_sparse = he.matrices_trig_sparse(self.fock, ECr, ELr)
        COSf_sparse, SINf_sparse = he.matrices_trig_sparse(self.FOCK, ECf, ELf)
        
        Qc = np.zeros((self.Nc,self.Nc), dtype=np.cfloat)
        COSc = np.zeros((self.Nc,self.Nc), dtype=np.cfloat)
        SINc = np.zeros((self.Nc,self.Nc), dtype=np.cfloat)
        Hr = np.zeros((self.fock,self.fock), dtype=np.cfloat)
        Qr = np.zeros((self.fock,self.fock), dtype=np.cfloat)
        Fr = np.zeros((self.fock,self.fock), dtype=np.cfloat)
        Hf = np.zeros((self.FOCK,self.FOCK), dtype=np.cfloat)
        Qf = np.zeros((self.FOCK,self.FOCK), dtype=np.cfloat)
        Ff = np.zeros((self.FOCK,self.FOCK), dtype=np.cfloat)
        for i in range(self.Nc):
            I = i-self.DIM
            Qc[i,i] += I-ng
            for j in range(self.Nc):
                if i == j-1:
                    COSc[i,j] += 0.5
                    SINc[i,j] += +0.5*1j
                elif i == j+1:
                    COSc[i,j] += 0.5
                    SINc[i,j] += -0.5*1j
                    
        for i in range(self.fock):
            Hr[i,i] += np.sqrt(8*ECr*ELr)*i
            for j in range(self.fock):
                if i == j-1:
                    Qr[i,j] -= 1j*0.5*(ELr/(2*ECr))**0.25*np.sqrt(j)
                    Fr[i,j] += (2*ECr/ELr)**0.25*np.sqrt(j)
                elif i == j+1:
                    Qr[i,j] += 1j*0.5*(ELr/(2*ECr))**0.25*np.sqrt(i)
                    Fr[i,j] += (2*ECr/ELr)**0.25*np.sqrt(i)
                    
        for i in range(self.FOCK):
            Hf[i,i] += np.sqrt(8*ECf*ELf)*i
            for j in range(self.FOCK):
                if i == j-1:
                    Qf[i,j] -= 1j*0.5*(ELf/(2*ECf))**0.25*np.sqrt(j)
                    Ff[i,j] += (2*ECf/ELf)**0.25*np.sqrt(j)
                elif i == j+1:
                    Qf[i,j] += 1j*0.5*(ELf/(2*ECf))**0.25*np.sqrt(i)
                    Ff[i,j] += (2*ECf/ELf)**0.25*cp.sqrt(i)
                    
        Qc_sparse = sparse.dia_matrix(sc.sparse.dia_array(Qc))
        Qr_sparse = sparse.dia_matrix(sc.sparse.dia_array(Qr))
        Qf_sparse = sparse.dia_matrix(sc.sparse.dia_array(Qf))
        Fr_sparse = sparse.dia_matrix(sc.sparse.dia_array(Fr))
        Ff_sparse = sparse.dia_matrix(sc.sparse.dia_array(Ff))
        Hr_sparse = sparse.dia_matrix(sc.sparse.dia_array(Hr))
        Hf_sparse = sparse.dia_matrix(sc.sparse.dia_array(Hf))
        COSc_sparse = sparse.dia_matrix(sc.sparse.dia_array(COSc))
        SINc_sparse = sparse.dia_matrix(sc.sparse.dia_array(SINc))
        #diagonal terms
        H0 = 4 * ECc * sparse.kron(Qc_sparse@Qc_sparse, sparse.eye(self.fock*self.FOCK)) \
            + sparse.kron(sparse.eye(self.Nc), sparse.kron(Hr_sparse, sparse.eye(self.FOCK))) \
            + sparse.kron(sparse.eye(self.Nc*self.fock), Hf_sparse) \
            + 4 * ECc * sparse.kron(sparse.eye(self.Nc), sparse.kron(Qr_sparse@Qr_sparse, sparse.eye(self.FOCK)))
        #capacitive coupling terms
        H1 = + 8 * ECc * sparse.kron(Qc_sparse, sparse.kron(Qr_sparse, sparse.eye(self.FOCK))) \
                - 2* self.dC * sparse.kron(sparse.eye(self.Nc), sparse.kron(Qr_sparse, Qf_sparse))
        #inductive coupling terms
        H2 = + self.dL * sparse.kron(sparse.eye(self.Nc), sparse.kron(Fr_sparse, Ff_sparse))
        #junction terms that go with cos(phiext/2)
        H3 = - 2 * self.EJ * (sparse.kron(COSc_sparse, sparse.kron(COSr_sparse, COSf_sparse)) + sparse.kron(SINc_sparse, sparse.kron(SINr_sparse, COSf_sparse))) \
                + self.dJ *( sparse.kron(SINc_sparse, sparse.kron(COSr_sparse, SINf_sparse)) - sparse.kron(COSc_sparse, sparse.kron(SINr_sparse, SINf_sparse)))
        #junction terms that go with sin(phiext/2)
        H4 = + 2 * self.EJ * (sparse.kron(COSc_sparse, sparse.kron(COSr_sparse, SINf_sparse)) + sparse.kron(SINc_sparse, sparse.kron(SINr_sparse, SINf_sparse))) \
                        + self.dJ *( sparse.kron(SINc_sparse, sparse.kron(COSr_sparse, COSf_sparse)) - sparse.kron(COSc_sparse, sparse.kron(SINr_sparse, COSf_sparse)))
        return H0 + H1 + H2, H3, H4


    def Hamiltonian_sparse(self, ng, phiext):
        if self.H is None:
            H = self._Hamiltonian2_sparse(ng)
            self.H = H[0] + cp.cos(phiext/2)*H[1] + cp.sin(phiext/2)*H[2]
        return self.H


    def first_n_states_sparse(self, ng, phiext, n_cutoff): 
        if self.ev is None or self.evec is None:
            self.ev, self.evec = eigsh(self.Hamiltonian_sparse(ng, phiext), k=n_cutoff, which='SA', return_eigenvectors=True)
        return self.ev, self.evec
    


class HarmonicOscillator(object):
    """Harmonic oscillator Hamiltonian."""
    def __init__(self, frequency, num_osc=20):
        self._frequency = 0.
        self._num_osc = 0
        self.frequency = frequency
        self.num_osc = num_osc

    def __str__(self):
        units = 'GHz'
        return ('Harmonic oscillator f = %.4f %s.'
                % (self.frequency, units))

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if value != self._frequency:
            self._frequency = value
            self._reset_cache()

    @property
    def num_osc(self):
        return self._num_osc

    @num_osc.setter
    def num_osc(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError('The number of levels must be positive.')
        if value != self._num_osc:
            self._num_osc = value
            self._reset_cache()

    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._H_op = None
        self._b_op = None
        self._eye_op = None

    def eye(self):
        """
        Identity operator.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            Identity operator.
        """
        if self._eye_op is None:
            self._eye_op = cp.eye(self._num_osc)
        return self._eye_op

    def b(self):
        """Annihilation operator."""
        if self._b_op is None:
            self._b_op = he.destroy(self._num_osc)
        return self._b_op
    
    def b_dag(self):
        """Creation operator."""
        if self._b_op is None:
            self._b_op = he.create(self._num_osc)
        return self._b_op

    def H(self):
        """Hamiltonian."""
        if self._H_op is None:
            self._H_op = self._frequency * self.b_dag() * self.b()
        return self._H_op

class Kite_Transmon_Taha(object):
    """Helper class to build the full Kite_Transmon Hamiltonian""" 
    def __init__(self, params):
        for key in ['E_C','E_CJ','E_J', 'E_L', 'n_g', 'phi_ext', 'num_qbt_1', 'num_qbt_2', 'num_qbt_0','num_osc']:
            if key not in params:
                raise ValueError('%s should be specified.' % key)
        
        self._E_L = 0.
        self._E_C = 0.
        self._E_CJ = 0.
        self._E_J = 0.
        self._phi_ext = 0.
        self._n_g = 0
        self._num_qbt = 0
        self._num_qbt_mode0 = 0
        self._num_qbt_mode1 = 0
        self._num_qbt_mode2 = 0
        self._num_osc = 0

        # Specify the inductive energy of each inductance.
        self.E_L = params['E_L']
        # Specify the charging energy of each JJ.
        self.E_CJ = params['E_CJ']
        # Specify the Josephson energy of each JJ.
        self.E_J = params['E_J']
        # Specify the charging energy of the shunting capacitance.
        self.E_C = params['E_C']
        # Specify the phi_ext defined as a fraction of Phi_0.
        self.phi_ext = np.array([params['phi_ext']]).flatten()[0]

        # Specify the offset charge defined as a fraction of the single
        # Cooper pair charge, i.e. 2e.
        if np.array(params['n_g']).size > 1:
            n_g = params['n_g'][0]
        else:
            n_g = params['n_g']
        self.n_g = n_g

        # Specify the number of states in the qubit basis for mode sigma.
        self.num_qbt_mode0 = params['num_qbt_0']
        # Specify the number of states in the qubit basis for mode delta.
        self.num_qbt_mode1 = params['num_qbt_1']
        # Specify the number of states in the qubit basis for mode phi.
        self.num_qbt_mode2 = params['num_qbt_2']
        
        self.num_qbt = self.num_qbt_mode0*self.num_qbt_mode1*self.num_qbt_mode2

        self.num_osc = params['num_osc']

        self.phi_sigma_zpf = (8*(.5*self.E_CJ + self.E_C)/(2*self.E_L) )**.25
        self.phi_delta_zpf = (8*(.5*self.E_CJ)/(2*self.E_L) )**.25


    @property
    def E_CJ(self):
        return self._E_CJ

    @E_CJ.setter
    def E_CJ(self, value):
        if value <= 0:
            raise ValueError('Charging energy must be positive.')
        if value != self._E_CJ:
            self._E_CJ = value
            self._reset_cache()

    @property
    def E_J(self):
        return self._E_J

    @E_J.setter
    def E_J(self, value):
        if value <= 0:
            raise ValueError('Josepshson energy must be positive.')
        if value != self._E_J:
            self._E_J = value
            self._reset_cache()    

    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value):
        if value <= 0:
            raise ValueError('Charging energy must be positive.')
        if value != self._E_C:
            self._E_C = value
            self._reset_cache()

    @property
    def E_L(self):
        return self._E_L

    @E_L.setter
    def E_L(self, value):
        if value <= 0:
            raise ValueError('Josephson energy must be positive.')
        if value != self._E_L:
            self._E_L = value
            self._reset_cache()
      
    @property
    def n_g(self):
        return self._n_g

    @n_g.setter
    def n_g(self, value):
        # n_g is defined as a fraction of 2e.
        if value != self._n_g:
            self._n_g = value
            self._reset_cache()

    @property
    def phi_ext(self):
        return self._phi_ext

    @phi_ext.setter
    def phi_ext(self, value):
        # n_g is defined as a fraction of 2e.
        if value != self._phi_ext:
            self._phi_ext = value
            self._reset_cache()

    @property
    def num_qbt_mode0(self):
        return self._num_qbt_mode0

    @num_qbt_mode0.setter
    def num_qbt_mode0(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError('The number of oscillator levels '
                    'must be positive.')
        if value != self._num_qbt_mode0:
            self._num_qbt_mode0 = value
            self._reset_cache()

    @property
    def num_qbt_mode1(self):
        return self._num_qbt_mode1

    @num_qbt_mode1.setter
    def num_qbt_mode1(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError('The number of oscillator levels '
                    'must be positive.')
        if value != self._num_qbt_mode1:
            self._num_qbt_mode1 = value
            self._reset_cache()

    @property
    def num_qbt_mode2(self):
        return self._num_qbt_mode2

    @num_qbt_mode2.setter
    def num_qbt_mode2(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError('The number of oscillator levels '
                    'must be positive.')
        if value != self._num_qbt_mode2:
            self._num_qbt_mode2 = value
            self._reset_cache()
    
    @property
    def num_osc(self):
        return self._num_osc

    @num_osc.setter
    def num_osc(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError('The number of oscillator levels '
                    'must be positive.')
        if value != self._num_osc:
            self._num_osc = value
            self._reset_cache()


    def _reset_cache(self):
        """Reset the cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None
        self._H_op = None
        self._eye_op = None
        self._N_op = None
        self._phi_delta_op = None
        self._phi_sigma_op = None


    # Phi operators and functions of Phi operators
    def _phi_sum_osc(self):
        """Flux (phase) operator in the oscillator basis for mode sigma"""
        op = self.phi_sigma_zpf * he.position(self.num_qbt_mode0)
        return cp.kron(op, cp.kron(cp.eye(self.num_qbt_mode1), cp.eye(self.num_qbt_mode2)))
    
    def _phi_sum_osc_square(self):
        """Flux (phase) operator in the oscillator basis for mode sigma"""
        op = self.phi_sigma_zpf * he.position(self.num_qbt_mode0)
        return cp.kron(op**2, cp.kron(cp.eye(self.num_qbt_mode1), cp.eye(self.num_qbt_mode2)))
    
    def _sinphi_sum_osc(self):
        """Flux (phase) operator in the oscillator basis for mode sigma"""
        op = self.phi_sigma_zpf * he.position(self.num_qbt_mode0)
        return cp.kron(he.sinm(op), cp.kron(cp.eye(self.num_qbt_mode1), cp.eye(self.num_qbt_mode2)))
    
    def _cosphi_sum_osc(self):
        """Flux (phase) operator in the oscillator basis for mode sigma"""
        op = self.phi_sigma_zpf * he.position(self.num_qbt_mode0)
        return cp.kron(he.cosm(op), cp.kron(cp.eye(self.num_qbt_mode1), cp.eye(self.num_qbt_mode2)))

    def _phi_diff_osc(self):
        """Flux (phase) operator in the oscillator basis for mode delta"""
        op = self.phi_delta_zpf * he.position(self.num_qbt_mode1)
        return cp.kron(cp.eye(self.num_qbt_mode0), cp.kron(op, cp.eye(self.num_qbt_mode2)))
    
    def _phi_diff_osc_square(self):
        """Flux (phase) operator in the oscillator basis for mode delta"""
        op = self.phi_delta_zpf * he.position(self.num_qbt_mode1)
        return cp.kron(cp.eye(self.num_qbt_mode0), cp.kron(op**2, cp.eye(self.num_qbt_mode2)))
    
    def _cosphi_diff_osc(self):
        """Flux (phase) operator in the oscillator basis for mode delta"""
        op = self.phi_delta_zpf * he.position(self.num_qbt_mode1)
        return cp.kron(cp.eye(self.num_qbt_mode0), cp.kron(he.cosm(op), cp.eye(self.num_qbt_mode2)))
    
    def _sinphi_diff_osc(self):
        """Flux (phase) operator in the oscillator basis for mode delta"""
        op = self.phi_delta_zpf * he.position(self.num_qbt_mode1)
        return cp.kron(cp.eye(self.num_qbt_mode0), cp.kron(he.sinm(op), cp.eye(self.num_qbt_mode2)))
    
    def _cos_phi_cap_chg(self):
        """cos (operator) in the charge basis for mode phi"""
        off_diag = .5 * cp.ones(self.num_qbt_mode2 - 1)
        op = cp.diag(off_diag, 1) + cp.diag(off_diag, -1)

        return cp.kron(cp.eye(self.num_qbt_mode0), cp.kron(cp.eye(self.num_qbt_mode1), op))
        
    def _sin_phi_cap_chg(self):
        """cos (operator) in the charge basis for mode phi"""
        off_diag = .5j * cp.ones(self._num_qbt_mode2 - 1)
        op = cp.diag(off_diag, 1) - cp.diag(off_diag, -1)

        return cp.kron(cp.eye(self.num_qbt_mode0), cp.kron(cp.eye(self.num_qbt_mode1), op))
    
    # Charge operators and functions of Charge operators

    def _n_sum_osc(self):
        """Charge operator in the oscillator basis for mode sigma"""
        op = (self.phi_sigma_zpf)**-1 * he.momentum(self.num_qbt_mode0)
        return cp.kron(op, cp.kron(cp.eye(self.num_qbt_mode1), cp.eye(self.num_qbt_mode2)))
    
    def _n_sum_osc_square(self):
        """Charge operator in the oscillator basis for mode sigma"""
        op = (self.phi_sigma_zpf)**-1 * he.momentum(self.num_qbt_mode0)
        return cp.kron(op**2, cp.kron(cp.eye(self.num_qbt_mode1), cp.eye(self.num_qbt_mode2)))
    
    def _n_diff_osc(self):
        """Charge operator in the oscillator basis for mode delta"""
        op = (self.phi_delta_zpf)**-1 * he.momentum(self.num_qbt_mode1)
        return cp.kron(cp.eye(self.num_qbt_mode0), cp.kron(op, cp.eye(self.num_qbt_mode2)))
    
    def _n_diff_osc_square(self):
        """Charge operator in the oscillator basis for mode delta"""
        op = (self.phi_delta_zpf)**-1 * he.momentum(self.num_qbt_mode1)
        return cp.kron(cp.eye(self.num_qbt_mode0), cp.kron(op**2, cp.eye(self.num_qbt_mode2)))

    def _n_cap_chg(self):
        """Charge operator in the charge basis for mode phi"""
        num_chg = self.num_qbt_mode2
        low = cp.ceil(self.n_g - cp.double(num_chg) / 2.)
        Qc = cp.diag( cp.linspace(low, low + num_chg - 1, num_chg) )
            
        return cp.kron(Qc,cp.kron(cp.eye(self.fock),cp.eye(self.FOCK)))

    def _n_coupled_cap(self):
        """Charge operator in the charge basis for mode phi"""
        num_chg = self.num_qbt_mode2
        low = cp.ceil(self.n_g - cp.double(num_chg) / 2.)
        op = cp.diag( cp.linspace(low, low + num_chg - 1, num_chg) )
                     
        return self._n_sum_osc_square() + 2*cp.kron((self.phi_sigma_zpf)**-1 * he.momentum(self.num_qbt_mode0), cp.kron(cp.eye(self.num_qbt_mode1), (op - self.n_g))) \
        + cp.kron( cp.eye(self.num_qbt_mode0), cp.kron(cp.eye(self.num_qbt_mode1), (op - self.n_g)**2))


    def _H_osc(self):
        """Qubit Hamiltonian in the oscillator basis."""
        
        return 4. * self.E_C * self._n_coupled_cap() + 2. * self.E_CJ * (self._n_diff_osc_square() + self._n_sum_osc_square()) + (
                self.E_L * ( self._phi_sum_osc_square() + self._phi_diff_osc_square() ) - 2 * self.E_J * 
                (self._cosphi_diff_osc()*cp.cos(self.phi_ext*cp.pi) - self._sinphi_diff_osc()*cp.sin(self.phi_ext*cp.pi)) * 
                  (self._cos_phi_cap_chg()*self._cosphi_sum_osc() + self._sin_phi_cap_chg()*self._sinphi_sum_osc()) )


    def _spectrum_osc(self):
        """Eigen-energies and eigenstates in the oscillator basis."""

        if self._eigvals is None or self._eigvecs is None:
            
            h = sps.csr_matrix(self._H_osc())
            
            raw_eigvals, raw_eigvecs = sps.linalg.eigs(h, k=self.num_osc, which='SR', return_eigenvectors=True)
            
            real_eigvals = raw_eigvals.real
            sorted_indices = real_eigvals.argsort()
            sorted_eigvals = real_eigvals[sorted_indices]
            
            self._eigvecs = raw_eigvecs[:, sorted_indices]

            self._eigvals = sorted_eigvals
        return self._eigvals, self._eigvecs

    def levels(self):
        """
        Eigen-energies of the qubit.

        Parameters
        ----------
        None.

        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        return self._spectrum_osc()[0][:self._num_osc]
        
    def states(self):
        """
        Eigenstates of the qubit.

        Parameters
        ----------
        None.

        Returns
        -------
        numpy.ndarray
            Array of eigenstates.
        """
        return self._spectrum_osc()[1][:self._num_osc]
        
    def _check_level(self, level):
        if level < 0 or level >= self._num_osc:
            raise ValueError('The level index is out of bounds.')

    def level(self, level):
        """
        Energy of a single level of the qubit.

        Parameters
        ----------
        level : int
            Qubit level, starting from zero.

        Returns
        -------
        float
            Energy of the qubit level.
        """
        self._check_level(level)
        return self._spectrum_osc()[0][level]

    def frequency(self, level1, level2):
        """
        Transition energy/frequency between two levels of the qubit.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        float
            Transition energy/frequency between `level1` and `level2`
            defined as the difference of energies. Positive
            if `level1` < `level2`.
        """
        self._check_level(level1)
        self._check_level(level2)
        return self.level(level2) - self.level(level1)

    def eye(self):
        """
        Identity operator in the qubit eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            Identity operator.
        """
        if self._eye_op is None:
            self._eye_op = cp.eye(self._num_osc)
        return self._eye_op

    def H(self):
        """
        Qubit Hamiltonian in its eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            Hamiltonian operator.
        """
        if self._H_op is None:
            self._H_op = cp.diag(self.levels()[:self._num_osc])
        return self._H_op

    def phi_sigma(self):
        """
        Generalized-flux operator in the qubit eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            Flux operator.
        """
        if self._phi_sigma_op is None:
            op = cp.zeros((self._num_osc, self._num_osc), dtype=complex) 

            for i in range(self._num_osc):
                for j in range(self._num_osc):
                    op[i, j] = self.phi_sigma_ij(i, j)

            self._phi_sigma_op = op
        return self._phi_sigma_op

    def phi_delta(self):
        """
        Generalized-flux operator in the qubit eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            Flux operator.
        """
        if self._phi_delta_op is None:
            op = cp.zeros((self._num_osc, self._num_osc), dtype=complex) 

            for i in range(self._num_osc):
                for j in range(self._num_osc):
                    op[i, j] = self.phi_delta_ij(i, j)

            self._phi_delta_op = op

        return self._phi_delta_op


    def N(self):
        """
        Charge operator in the qubit eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            Charge operator.
        """
        if self._N_op is None:
            n_op = cp.zeros((self._num_osc, self._num_osc), dtype=complex) 

            for i in range(self._num_osc):
                for j in range(self._num_osc):
                    n_op[i, j] = self.N_ij(i, j)

            self._N_op = n_op

        return self._N_op

    def phi_sigma_ij(self, level1, level2):
        """
        Flux matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the flux operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum_osc()[1]
        return evecs[level1].transpose()*self._phi_sum_osc()*evecs[level2]

    def phi_delta_ij(self, level1, level2):
        """
        Flux matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the flux operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum_osc()[1]
        return evecs[level1].transpose()*self._phi_diff_osc()*evecs[level2]
 

    def N_ij(self, level1, level2):
        """
        Charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the charge operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum_osc()[1]
        return evecs[level1].transpose()*self._n_cap_chg()*evecs[level2]
    
    def n_sigma_ij(self, level1, level2):
        """
        Charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the charge operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum_osc()[1]
        return self._n_sum_osc().matrix_element(evecs[level1].dag(),
                                            evecs[level2])
    
    def n_delta_ij(self, level1, level2):
        """
        Charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the charge operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum_osc()[1]
        return evecs[level1].transpose()*self._n_diff_osc()*evecs[level2]
    
    def dephasing_op_CC_ij(self, level1, level2):
        """
        Charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the charge operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum_osc()[1]
        operator = 2 *(self._cosphi_diff_osc()*cp.cos(self.phi_ext*cp.pi) - self._sinphi_diff_osc()*cp.sin(self.phi_ext*cp.pi)) * (
                      self._cos_phi_cap_chg()*self._cosphi_sum_osc() + self._sin_phi_cap_chg()*self._sinphi_sum_osc()) 
        return evecs[level1].transpose()*operator*evecs[level2]
    
    def dephasing_op_Flux_ij(self, level1, level2):
        """
        Charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the charge operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum_osc()[1]
        operator = self.E_J * (self._cosphi_diff_osc()*cp.sin(self.phi_ext*cp.pi) + self._sinphi_diff_osc()*cp.cos(self.phi_ext*cp.pi)) *(
            self._cos_phi_cap_chg()*self._cosphi_sum_osc() + self._sin_phi_cap_chg()*self._sinphi_sum_osc()) 
        return evecs[level1].transpose()*operator*evecs[level2]
    
    def dephasing_op_Flux_ij_2(self, level1, level2):
        """
        Charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the charge operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum_osc()[1]
        operator =  .5* self.E_J * (self._cosphi_diff_osc()*cp.cos(self.phi_ext*cp.pi) - self._sinphi_diff_osc()*cp.sin(self.phi_ext*cp.pi)) * (
                  self._cos_phi_cap_chg()*self._cosphi_sum_osc() + self._sin_phi_cap_chg()*self._sinphi_sum_osc()) 
        return evecs[level1].transpose()*operator*evecs[level2]
    
    def dephasing_op_Chg_ij(self, level1, level2):
        """
        Charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the charge operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum_osc()[1]
        operator = 8 * self.E_C * (self._n_cap_chg + self._n_sum_osc - self.n_g)
        return evecs[level1].transpose()*operator*evecs[level2]

class Model(object):
    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None
        self._eye_op = None
        self._H_op = None
        self._weights = None
        
    @property
    def num_tot(self):
        return self._num_tot

    def _spectrum(self):
        """Eigen-energies and eigenstates in the oscillator basis."""
        if self._eigvals is None or self._eigvecs is None:
            self._eigvals, self._eigvecs = \
            h = sps.csr_matrix(self._hamiltonian)
            
            raw_eigvals, raw_eigvecs = sps.linalg.eigs(h, k=self.num_tot, which='SR', return_eigenvectors=True)
            
            real_eigvals = raw_eigvals.real
            sorted_indices = real_eigvals.argsort()
            sorted_eigvals = real_eigvals[sorted_indices]
            
            self._eigvecs = raw_eigvecs[:, sorted_indices]

            self._eigvals = sorted_eigvals
        return self._eigvals, self._eigvecs

    def levels(self):
        """Eigen-energies of the coupled system.

        Parameters
        ----------
        None.

        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        return self._spectrum()[0][:self._num_tot]

    def _check_level(self, level):
        if level < 0 or level >= self._num_tot:
            raise ValueError('The level is out of bounds: 0 and %d.'
                    % self._num_tot)

    def level(self, level):
        """Energy of a single level of the qubit.

        Parameters
        ----------
        level: int
            Qubit level.

        Returns
        -------
        float
            Energy of the level.
        """
        self._check_level(level)
        return self._spectrum()[0][level]

    def frequency(self, level1, level2):
        """Transition energy/frequency between two levels of the qubit.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        float
            Transition energy/frequency between `level1` and `level2` defined
            as the difference of energies. Positive if `level1` < `level2`.
        """
        self._check_level(level1)
        self._check_level(level2)
        return self.level(level2) - self.level(level1)
    
    def states(self):
        """Eigenstates of the system.

        Parameters
        ----------
        None.

        Returns
        -------
        numpy.ndarray
            Array of eigenstates.
        """
        return self._spectrum()[1]
        
    def eye(self):
        """Identity operator.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            Identity operator.
        """
        if self._eye_op is None:
            self._eye_op = cp.eye(self._num_tot)
        return self._eye_op
        
    def H(self):
        """Hamiltonian in its eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            Hamiltonian operator.
        """
        if self._H_op is None:
            self._H_op = cp.diag(self.levels()[:self._num_tot])
        return self._H_op

    def weights(self):
        if self._weights is None:
            evecs = self._spectrum()[1]
            num_qbt = self._fluxonium.num_qbt
            self._weights = np.zeros((self._num_tot, num_qbt))
            for idx in range(self._num_tot):
                w = np.abs(np.array(evecs[idx].data.todense()))**2.
                w.shape = (num_qbt, -1)
                self._weights[idx] = np.sum(w, axis=1)
        return self._weights

class Kite_transmon_CoupledToResonator(Model):
    """
    Fluxonium Hamiltonian flux-coupled to a chain and a resonator
    modes.
    """

    def __init__(self, Kite_Transmon, params):
        resonator = HarmonicOscillator(frequency=params['f_r'],
                                       num_osc=params['num_res'])
        
        osc_tot = Kite_Transmon.num_osc * params['num_res']
        num_tot = int(params['num_tot'])
        if num_tot > osc_tot:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot
        
        H = cp.kron(Kite_Transmon.H(), resonator.eye())
        H += cp.kron(Kite_Transmon.eye(), resonator.H())
        H += (-1.j * params['g_r_J']
                  * cp.kron(Kite_Transmon.N(),
                              resonator.b() - resonator.b().dag()))
        
       
        self._hamiltonian = (H + H.dag())/2
        self._fluxonium = Kite_Transmon

        self._reset_cache()