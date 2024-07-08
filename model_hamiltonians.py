import numpy as np
import qutip as qt
from qutip.parallel import parallel_map
from scipy.special import eval_hermite as hpoly

try:
    from simple_hamiltonians import Fluxonium, HarmonicOscillator, Kite_Transmon_Slowmode
except:
    from .simple_hamiltonians import Fluxonium, HarmonicOscillator, Kite_Transmon_Slowmode


class Model(object):
    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None
        self._eye_op = None
        self._H_op = None
        self._weights = None
        self._purity = None
        
    @property
    def num_tot(self):
        return self._num_tot

    def _spectrum(self):
        """Eigen-energies and eigenstates in the oscillator basis."""
        if self._eigvals is None or self._eigvecs is None:
            self._eigvals, self._eigvecs = \
                    self._hamiltonian.eigenstates(sparse=True,
                    eigvals=self._num_tot)
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
            self._eye_op = qt.qeye(self._num_tot)
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
            self._H_op = qt.Qobj(np.diag(self.levels()[:self._num_tot]))
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
        
    def purity(self):
        if self._purity is None:
            evecs = self._spectrum()[1]
            num_qbt = self._fluxonium.num_qbt
            self._purity = np.zeros((self._num_tot, num_qbt))
            for idx in range(self._num_tot):
                w = np.abs(np.array(evecs[idx].data.todense()))**2.
                w.shape = (num_qbt, -1)
                self._purity[idx] = w[:,0]
        return self._purity


class TwoCoupledFluxoniums(Model):
    """Fluxonium Hamiltonian coupled to n harmonic modes."""
    def __init__(self, fluxonium1, fluxonium2, params):

        self._num_tot = fluxonium1.num_qbt * fluxonium2.num_qbt
        
        self._num_wq1 = fluxonium1._num_qbt
        if 'num_wq1' in params:
            self._num_wq1 = int(params['num_wq1'])
        self._num_wq2 = fluxonium2._num_qbt
        if 'num_wq2' in params:
            self._num_wq2 = int(params['num_wq2'])

        H = qt.tensor(fluxonium1.H(), fluxonium2.eye())
        H += qt.tensor(fluxonium1.eye(), fluxonium2.H())

        if 'E_int_chg' in params:
            H += params['E_int_chg'] * qt.tensor(fluxonium1.n(),
                                                 fluxonium2.n())
        if 'E_int_flx' in params:
            H += params['E_int_flx'] * qt.tensor(fluxonium1.phi(),
                                                 fluxonium2.phi())                                    
        self._hamiltonian = H
        self._fluxonium1 = fluxonium1
        self._fluxonium2 = fluxonium2
        self._reset_cache()
        
    def frequency(self, level1, level2):
        """Transition energy/frequency between two levels of the qubit.

        Parameters
        ----------
        level1, level2 : int or tuple
            The qubit levels.

        Returns
        -------
        float
            Transition energy/frequency between `level1` and `level2`
            defined as the difference of energies.
            Positive if `level1` < `level2`.
        """
        if isinstance(level1, tuple):
            level1 = self.weights(labels=True)[level1[0],level1[1]]
        if isinstance(level2, tuple):
            level2 = self.weights(labels=True)[level2[0],level2[1]]
        self._check_level(level1)
        self._check_level(level2)
        return self.level(level2) - self.level(level1)
        
    def weights(self, labels=False):
        if self._weights is None:
            evecs = self.states()

            qbt1 = self._fluxonium1._num_qbt
            qbt2 = self._fluxonium2._num_qbt
            wq1 = self._num_wq1
            wq2 = self._num_wq2
            
            weights = np.zeros((wq1*wq2, wq1, wq2), dtype=np.complex)
            for idx in range(wq1 * wq2):
                weights[idx] = \
                        evecs[idx].data.todense().reshape(qbt1, qbt2)[:wq1,:wq2]
            self._weights = np.abs(weights)**2.

            # Each label must be used only once.
            self._labels = np.zeros((wq1, wq2), dtype=np.int)
            w = self._weights.copy()
            for idx2 in range(wq2):
                for idx1 in range(wq1):
                    level_idx = np.argmax(w[:,idx1,idx2])
                    w[level_idx] = -np.ones((wq1, wq2))
                    self._labels[idx1][idx2] = level_idx
        if not labels:
            return self._weights
        else:
            return self._labels


class _ReducedHilbertSpace(Model):
    def __init__(self, system1, system2, num_tot):
        H = qt.tensor([system1.H(), system2.eye()])
        H += qt.tensor([system1.eye(), system2.H()])

        if num_tot > system1.H().shape[0] * system2.H().shape[0]:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        self._hamiltonian = H
        self._system1 = system1
        self._system2 = system2
        self._reset_cache()
        
    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        Model._reset_cache(self)
        self._b_ops = None

    def b(self):
        """Annihilation operators in the combined eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            List of annihilation operators.
        """
        if self._b_ops is None:
            self._b_ops = []
            evecs = self.states()
            num_tot = self.num_tot
            num_combined = (self._system1.H().shape[0]
                          * self._system2.H().shape[0])
            evecs_padded = np.zeros((evecs.shape[0], num_combined),
                    dtype=complex)
            for k in range(evecs.shape[0]):
                evecs_padded[k] = evecs[k].full().flatten()
            for k, system in enumerate([self._system1, self._system2]):
                bs = system.b()
                if type(bs) != list:
                    bs = [bs]
                for b_idx in bs:
                    if k:
                        b = qt.tensor([self._system1.eye(), b_idx])
                    else:
                        b = qt.tensor([b_idx, self._system2.eye()])
                    b_op = b.transform(evecs_padded)[:num_tot,:num_tot]
                    self._b_ops.append(qt.Qobj(b_op))
        return self._b_ops


class ParallelUncoupledOscillators(Model):
    """Pallel implementation of uncoupled oscillators."""
    def __init__(self, params, pairwise=True):
        if len(params['num_mod']) != len(params['frequencies']):
            raise ValueError('Oscillators are not properly defined.')

        oscillators = []
        for idx in range(len(params['frequencies'])):
            oscillators.append(HarmonicOscillator(
                    frequency=params['frequencies'][idx],
                    num_osc=params['num_mod'][idx]))

        num_tot = int(params['num_cpl'])
        if num_tot > np.prod(params['num_mod']):
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        self._reset_cache()

        if pairwise:
            while len(oscillators) > 1:
                if len(oscillators) == 2:
                    cutoff = num_tot
                else:
                    cutoff = int(2.5 * np.sqrt(num_tot))
                if len(oscillators) % 2 == 0:
                    reduced = []
                    idx = 0
                else:
                    reduced = [oscillators[0]]
                    idx = 1
                reduced = reduced + parallel_map(self._parallel,
                            range(idx, len(oscillators), 2),
                            task_args=(oscillators, cutoff),
                            num_cpus=1)
                oscillators = reduced

            self._hamiltonian = oscillators[0].H()
            self._b_ops = oscillators[0].b()
            if type(self._b_ops) != list:
                self._b_ops = [self._b_ops]
        else:
            if len(oscillators) == 1:
                self._hamiltonian = oscillators[0].H()
                self._b_ops = [oscillators[0].b()]
            else:
                system = oscillators[0]
                for k, osc in enumerate(oscillators[1:]):
                    system = _ReducedHilbertSpace(system, osc, num_tot)
                self._hamiltonian = system.H()
                self._b_ops = system.b()
                
    @classmethod
    def _parallel(cls, k, oscillators, cutoff):
        return _ReducedHilbertSpace(oscillators[k],
                                    oscillators[k+1],
                                    cutoff)

    def b(self):
        """Annihilation operators in the multi-oscillator eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            List of annihilation operators.
        """
        return self._b_ops


class UncoupledOscillators(Model):
    """Uncoupled oscillators."""
    def __init__(self, params, pairwise=True):
        if len(params['num_mod']) != len(params['frequencies']):
            raise ValueError('Oscillators are not properly defined.')

        oscillators = []
        for idx in range(len(params['frequencies'])):
            oscillators.append(HarmonicOscillator(
                    frequency=params['frequencies'][idx],
                    num_osc=params['num_mod'][idx]))

        num_tot = int(params['num_cpl'])
        if num_tot > np.prod(params['num_mod']):
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        self._reset_cache()

        if pairwise:
            while len(oscillators) > 1:
                if len(oscillators) == 2:
                    cutoff = num_tot
                else:
                    cutoff = int(2.5 * np.sqrt(num_tot))
                if len(oscillators) % 2 == 0:
                    reduced = []
                    idx = 0
                else:
                    reduced = [oscillators[0]]
                    idx = 1
                for k in range(idx, len(oscillators), 2):
                    reduced.append(_ReducedHilbertSpace(oscillators[k],
                                                        oscillators[k+1],
                                                        cutoff))
                oscillators = reduced

            self._hamiltonian = oscillators[0].H()
            self._b_ops = oscillators[0].b()
            if type(self._b_ops) != list:
                self._b_ops = [self._b_ops]
        else:
            if len(oscillators) == 1:
                self._hamiltonian = oscillators[0].H()
                self._b_ops = [oscillators[0].b()]
            else:
                system = oscillators[0]
                for k, osc in enumerate(oscillators[1:]):
                    system = _ReducedHilbertSpace(system, osc, num_tot)
                self._hamiltonian = system.H()
                self._b_ops = system.b()

    def b(self):
        """Annihilation operators in the multi-oscillator eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            List of annihilation operators.
        """
        return self._b_ops


class CoupledOscillators(Model):
    """Coupled oscillators."""
    def __init__(self, params):
        if len(params['num_mod']) != len(params['frequencies']):
            raise ValueError('Oscillators are not properly defined.')

        oscillators = []
        for idx in range(len(params['frequencies'])):
            oscillators.append(HarmonicOscillator(
                    frequency=params['frequencies'][idx],
                    num_osc=params['num_mod'][idx]))
    
        if 'n_cross_couplings' in params:
            dim = len(oscillators)
            if params['n_cross_couplings'].shape != (dim, dim):
                raise ValueError('The charge cross coupling matrix'
                        ' is not properly defined.')
        
        if 'phi_cross_couplings' in params:
            dim = len(oscillators)
            if params['phi_cross_couplings'].shape != (dim, dim):
                raise ValueError('The flux cross coupling matrix'
                        ' is not properly defined.')

        osc_total = 1
        for osc in oscillators:
            osc_total *= osc.num_osc
        num_tot = int(params['num_cpl'])
        if num_tot > osc_total:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        for idx1, osc1 in enumerate(oscillators):
            array = []
            for idx2, osc2 in enumerate(oscillators):
                if idx1 == idx2:
                    array.append(osc1.H())
                else:
                    array.append(osc2.eye())
            if idx1 == 0:
                H = qt.tensor(*array)
            else:
                H += qt.tensor(*array)

        for idx1, osc1 in enumerate(oscillators):
            for idx2, osc2 in enumerate(oscillators):
                if idx2 > idx1:
                    if 'n_cross_couplings' in params:
                        array = []
                        for idx3 in range(len(oscillators)):
                            if idx3 == idx1:
                                array.append(osc1.b() - osc1.b().dag())
                            elif idx3 == idx2:
                                array.append(osc2.b() - osc2.b().dag())
                            else:
                                array.append(oscillators[idx3].eye())
                        H += (params['n_cross_couplings'][idx1,idx2] *
                                qt.tensor(*array))

                    if 'phi_cross_couplings' in params:
                        array = []
                        for idx3 in range(len(oscillators)):
                            if idx3 == idx1:
                                array.append(osc1.b() + osc1.b().dag())
                            elif idx3 == idx2:
                                array.append(osc2.b() + osc2.b().dag())
                            else:
                                array.append(oscillators[idx3].eye())
                        H += (params['phi_cross_couplings'][idx1,idx2] *
                                qt.tensor(*array))

        self._hamiltonian = H
        self._oscillators = oscillators
        self._reset_cache()
        
    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        Model._reset_cache(self)
        self._b_ops = None

    def b(self):
        """Annihilation operators in the multi-oscillator eigenbasis.

        Parameters
        ----------
        None.

        Returns
        -------
        :class:`qutip.Qobj`
            List of annihilation operators.
        """
        if self._b_ops is None:
            self._b_ops = []
            evecs = self.states()
            num_tot = self.num_tot
            for idx1, osc1 in enumerate(self._oscillators):
                array = []
                for idx2, osc2 in enumerate(self._oscillators):
                    if idx1 == idx2:
                        array.append(osc1.b())
                    else:
                        array.append(osc2.eye())
                b = qt.tensor(*array)

                evecs_padded = np.zeros((evecs.shape[0], b.shape[0]),
                                        dtype=complex)
                for k in range(evecs.shape[0]):
                    evecs_padded[k] = evecs[k].full().flatten()

                b_op = b.transform(evecs_padded)[:num_tot,:num_tot]
                self._b_ops.append(qt.Qobj(b_op))
        return self._b_ops


class FluxoniumCoupledToOscillators(Model):
    """Fluxonium Hamiltonian coupled to harmonic modes."""
    def __init__(self, fluxonium, oscillators, params):
        if ('n_couplings' in params and
                len(params['frequencies']) != len(params['n_couplings'])):
            raise ValueError('The number of oscillators should be equal'
                    ' to the number of charge couplings.')
        
        if ('phi_couplings' in params and
                len(params['frequencies']) != len(params['phi_couplings'])):
            raise ValueError('The number of oscillators should be equal'
                    ' to the number of flux couplings.')
        
        num_tot = int(params['num_tot'])
        if num_tot > fluxonium.num_qbt * oscillators.num_tot:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        H = qt.tensor([fluxonium.H(), oscillators.eye()])
        H += qt.tensor([fluxonium.eye(), oscillators.H()])

        for idx, b in enumerate(oscillators.b()):
            if 'n_couplings' in params:
                op = -1.j * params['n_couplings'][idx] * (b - b.dag())
                H += qt.tensor([fluxonium.n(), op])
            if 'phi_couplings' in params:
                op = params['phi_couplings'][idx] * (b + b.dag())
                H += qt.tensor([fluxonium.phi(), op])

        self._hamiltonian = (H + H.dag()) / 2.
        self._fluxonium = fluxonium
        self._coupled_oscillators = oscillators
        self._reset_cache()
        
    def hilbert_space_size(self):
        return self._hamiltonian.shape[0]
        
    def n_ij(self, level1, level2):
        """The charge matrix element between two eigenstates.

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
        evecs = self._spectrum()[1]
        
        op = qt.tensor([self._fluxonium.n(),
                        self._coupled_oscillators.eye()])

        return op.matrix_element(evecs[level1].dag(), evecs[level2])
        
    def phi_ij(self, level1, level2):
        """T
        he flux matrix element between two eigenstates.

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
        evecs = self._spectrum()[1]
        
        op = qt.tensor([self._fluxonium.phi(),
                        self._coupled_oscillators.eye()])

        return op.matrix_element(evecs[level1].dag(), evecs[level2])
        
    def half_phi_ij(self, level1, level2):
        """
        The half-flux matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the half-flux operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum()[1]
        
        op = qt.tensor([self._fluxonium.phi() / 2.,
                        self._coupled_oscillators.eye()])

        return op.matrix_element(evecs[level1].dag(), evecs[level2])
        
    def sin_half_phi_ij(self, level1, level2):
        """
        The sine of the half flux matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.

        Returns
        -------
        complex
            Matrix element of the sine of the half flux operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum()[1]

        op = qt.tensor([(.5 * self._fluxonium.phi()
                  + np.pi * self._fluxonium.phi_ext).sinm(),
                  self._coupled_oscillators.eye()])

        return op.matrix_element(evecs[level1].dag(), evecs[level2])

    def b_ij(self, level1, level2, mode=0):
        """
        The bosonic operator matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.
            
        mode: int
            Index of the harmonic mode.

        Returns
        -------
        complex
            Matrix element of the bosonic operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum()[1]
        
        op = qt.tensor([self._fluxonium.eye(),
                        self._coupled_oscillators.b()[mode]])

        return op.matrix_element(evecs[level1].dag(), evecs[level2])
    
    def b_dag_ij(self, level1, level2, mode=0):
        """
        The bosonic operator matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            Qubit levels.
            
        mode: int
            Index of the harmonic mode.

        Returns
        -------
        complex
            Matrix element of the bosonic operator.
        """
        self._check_level(level1)
        self._check_level(level2)
        evecs = self._spectrum()[1]
        
        op = qt.tensor([self._fluxonium.eye(),
                        self._coupled_oscillators.b()[mode].dag()])

        return op.matrix_element(evecs[level1].dag(), evecs[level2])



class FluxoniumChargeCoupledToResonator(Model):
    """Fluxonium Hamiltonian charge coupled to a resonator mode."""
    def __init__(self, fluxonium, params):
        resonator = HarmonicOscillator(frequency=params['f_r'],
                                       num_osc=params['num_res'])
        
        osc_tot = fluxonium.num_qbt * params['num_res']
        num_tot = int(params['num_tot'])
        if num_tot > osc_tot:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        H = qt.tensor(fluxonium.H(), resonator.eye())
        H += qt.tensor(fluxonium.eye(), resonator.H())

        H += (-1.j * params['g_r_J']
                  * qt.tensor(fluxonium.n(),
                              resonator.b() - resonator.b().dag()))
        self._hamiltonian = (H + H.dag())/2
        self._fluxonium = fluxonium

        self._reset_cache()
    


class FluxoniumChargeCoupledToOneModeAndResonator(Model):
    """
    Fluxonium Hamiltonian charge-coupled to a chain and a resonator
    modes.
    """
    def __init__(self, fluxonium, params):
        mode = HarmonicOscillator(frequency=params['f_m'],
                            num_osc=params['num_chn'])
        resonator = HarmonicOscillator(frequency=params['f_r'],
                                 num_osc=params['num_res'])
        
        osc_tot = fluxonium.num_qbt * params['num_chn'] * params['num_res']
        num_tot = int(params['num_tot'])
        if num_tot > osc_tot:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        H = qt.tensor(fluxonium.H(), mode.eye(), resonator.eye())
        H += qt.tensor(fluxonium.eye(), mode.H(), resonator.eye())
        H += qt.tensor(fluxonium.eye(), mode.eye(), resonator.H())

        mode_n = mode.b() - mode.b().dag()
        resonator_n = resonator.b() - resonator.b().dag()
        H += 1.j * params['g_m_J'] * qt.tensor(fluxonium.n(),
                                               mode_n,
                                               resonator.eye())
        H += 1.j * params['g_r_J'] * qt.tensor(fluxonium.n(),
                                               mode.eye(),
                                               resonator_n)
        H += params['g_m_r'] * qt.tensor(fluxonium.eye(),
                                         mode_n,
                                         resonator_n)
        self._hamiltonian = H
        self._fluxonium = fluxonium
        self._reset_cache()

        
class FluxoniumFluxCoupledToOneModeAndResonator(Model):
    """
    Fluxonium Hamiltonian flux-coupled to a chain and a resonator
    modes.
    """
    def __init__(self, fluxonium, params):
        mode = HarmonicOscillator(frequency=params['f_m'],
                            num_osc=params['num_chn'])
        resonator = HarmonicOscillator(frequency=params['f_r'],
                                 num_osc=params['num_res'])
        
        osc_tot = fluxonium.num_qbt * params['num_chn'] * params['num_res']
        num_tot = int(params['num_tot'])
        if num_tot > osc_tot:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        H = qt.tensor(fluxonium.H(), mode.eye(), resonator.eye())
        H += qt.tensor(fluxonium.eye(), mode.H(), resonator.eye())
        H += qt.tensor(fluxonium.eye(), mode.eye(), resonator.H())

        mode_phi = mode.b() + mode.b().dag()
        resonator_phi = resonator.b() + resonator.b().dag()
        H += params['g_m_J'] * qt.tensor(fluxonium.phi(),
                                         mode_phi,
                                         resonator.eye())
        H += params['g_r_J'] * qt.tensor(fluxonium.phi(),
                                         mode.eye(),
                                         resonator_phi)
        H += params['g_m_r'] * qt.tensor(fluxonium.eye(),
                                         mode_phi,
                                         resonator_phi)
        self._hamiltonian = H
        self._fluxonium = fluxonium
        self._reset_cache()


class FluxoniumFluxCoupledToTwoModesAndResonator(Model):
    """
    Fluxonium Hamiltonian flux-coupled to two chain and a resonator
    modes.
    """
    def __init__(self, fluxonium, params):
        mode1 = HarmonicOscillator(frequency=params['f_m1'],
                            num_osc=params['num_chn1'])
        mode2 = HarmonicOscillator(frequency=params['f_m2'],
                            num_osc=params['num_chn2'])
        resonator = HarmonicOscillator(frequency=params['f_r'],
                                 num_osc=params['num_res'])
        
        osc_tot = (fluxonium.num_qbt * params['num_chn1']
                   * params['num_chn2'] * params['num_res'])
        num_tot = int(params['num_tot'])
        if num_tot > osc_tot:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        H = qt.tensor(fluxonium.H(),
                mode1.eye(), mode2.eye(), resonator.eye())
        H += qt.tensor(fluxonium.eye(),
                mode1.H(), mode2.eye(), resonator.eye())
        H += qt.tensor(fluxonium.eye(),
                mode1.eye(), mode2.H(), resonator.eye())
        H += qt.tensor(fluxonium.eye(),
                mode1.eye(), mode2.eye(), resonator.H())

        mode1_phi = mode1.b() + mode1.b().dag()
        mode2_phi = mode2.b() + mode2.b().dag()
        resonator_phi = resonator.b() + resonator.b().dag()
        H += params['g_m1_J'] * qt.tensor(fluxonium.phi(),
                                          mode1_phi,
                                          mode2.eye(),
                                          resonator.eye())
        H += params['g_m2_J'] * qt.tensor(fluxonium.phi(),
                                          mode1.eye(),
                                          mode2_phi,
                                          resonator.eye())
        H += params['g_r_J'] * qt.tensor(fluxonium.phi(),
                                         mode1.eye(),
                                         mode2.eye(),
                                         resonator_phi)
        H += params['g_m1_r'] * qt.tensor(fluxonium.eye(),
                                          mode1_phi,
                                          mode2.eye(),
                                          resonator_phi)
        H += params['g_m2_r'] * qt.tensor(fluxonium.eye(),
                                          mode1.eye(),
                                          mode2_phi,
                                          resonator_phi)
        H += params['g_m1_m2'] * qt.tensor(fluxonium.eye(),
                                           mode1_phi,
                                           mode2_phi,
                                           resonator.eye())
        self._hamiltonian = H
        self._fluxonium = fluxonium
        self._reset_cache()

class TransmonChargeCoupledToResonator(Model):
    """Transmon Hamiltonian charge coupled to a resonator mode."""
    def __init__(self, transmon, params):
        resonator = HarmonicOscillator(frequency=params['f_r'],
                                       num_osc=params['num_res'])
        
        osc_tot = transmon.num_qbt * params['num_res']
        num_tot = int(params['num_tot'])
        if num_tot > osc_tot:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        H = qt.tensor(transmon.H(), resonator.eye())
        H += qt.tensor(transmon.eye(), resonator.H())

        H += (1.j * params['g_r_J']
                  * qt.tensor(transmon.n(),
                              resonator.b() - resonator.b().dag()))
        self._hamiltonian = H
        self._transmon = transmon
        self._reset_cache()

    def weights(self):
        if self._weights is None:
            evecs = self._spectrum()[1]
            num_qbt = self._transmon.num_qbt
            self._weights = np.zeros((self._num_tot, num_qbt))
            for idx in range(self._num_tot):
                w = np.abs(np.array(evecs[idx].data.todense()))**2.
                w.shape = (num_qbt, -1)
                self._weights[idx] = np.sum(w, axis=1)
        return self._weights
    
class Kite_transmon(Model):
    """
    Fluxonium Hamiltonian flux-coupled to a chain and a resonator
    modes.
    """

    def __init__(self, params):
        for key in ['E_C', 'E_J', 'E_L', 'E_CJ', 'eps_c', 'eps_j', 'eps_l', 'n_g', 'phi_ext',
                     'num_osc', 'num_qbt_1', 'num_qbt_2', 'num_qbt_0','num_tot_Kite']:
            if key not in params:
                raise ValueError('%s should be specified.' % key)

        # Specify the inductive energy of each inductance.
        self.E_L = params['E_L']
        # Specify the charging energy of each JJ.
        self.E_CJ = params['E_CJ']
        # Specify the Josephson energy of each JJ.
        self.E_J = params['E_J']
        # Specify the charging energy of the shunting capacitance.
        self.E_C = params['E_C']

        # Asymmetry in charging energy between 2 JJ 
        self.eps_c = params['eps_c']
        # Asymmetry in Josephson energy between 2 JJ
        self.eps_j = params['eps_j']
        # Asymmetry in inductive energy between 2 inductances
        self.eps_l = params['eps_l']

        # Specify the offset charge defined as a fraction of the single
        # Cooper pair charge, i.e. 2e.
        if np.array(params['n_g']).size > 1:
            n_g = params['n_g'][0]
        else:
            n_g = params['n_g']
        self.n_g = n_g

        # Specify the phi_ext defined as a fraction of Phi_0.
        self.phi_ext = -np.array([params['phi_ext']]).flatten()[0]
        # Specify the number of states in the oscillator basis.
        self.num_osc = params['num_osc']

        # Specify the number of states in the qubit basis for mode 0 (slow).
        self.num_qbt_mode0 = params['num_qbt_0']
        # Specify the number of states in the qubit basis for mode 1.
        self.num_qbt_mode1 = params['num_qbt_1']
        # Specify the number of states in the qubit basis for mode 2.
        self.num_qbt_mode2 = params['num_qbt_2']
        

        # Create instances for each mode of the circuit:
        fluxonium_1 = Fluxonium(param_kite2fluxonium(params,1))
        # Fluxonium 2 is mode theta 2
        fluxonium_2 = Fluxonium(param_kite2fluxonium(params,2))
        # Mode phi (slow mode of the shunting capacitance)
        SM = Kite_Transmon_Slowmode(param_kite2fluxonium(params,0))
        
        self.slow_mode = SM
        self.Fluxonium_theta1 = fluxonium_1
        self.Fluxonium_theta2 = fluxonium_2
        

        num_tot = int(params['num_tot_Kite'])
        if num_tot > self.num_qbt_mode0*self.num_qbt_mode1*self.num_qbt_mode2:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        H = qt.tensor(SM.eye(), fluxonium_1.H(), fluxonium_2.eye())
        H += qt.tensor(SM.eye(), fluxonium_1.eye(), fluxonium_2.H())
        H += qt.tensor(SM.H(), fluxonium_1.eye(), fluxonium_2.eye())

        # Add inductive coupling between mode theta1 and mode phi
        H -= (self.E_L  + self.eps_l ) * qt.tensor([(self.E_L / (4. * self.E_C ))**.25 * SM.phi(),fluxonium_1.eye(),fluxonium_2.eye()])* (
            qt.tensor(SM.eye(), fluxonium_1.phi(), fluxonium_2.eye()) )
        
        # Add inductive coupling between mode theta2 and mode phi
        H -= (self.E_L  - self.eps_l ) * qt.tensor([(self.E_L / (4. * self.E_C ))**.25 * SM.phi(),fluxonium_1.eye(),fluxonium_2.eye()]) * (
            qt.tensor(SM.eye(), fluxonium_1.eye(), fluxonium_2.phi()) )

        self._hamiltonian = (H + H.dag())/2
        self._reset_cache()

    def weights(self):
        evecs = self._spectrum()[1]

        num_qbt_mode0 = self.num_qbt_mode0
        num_qbt_mode1 = self.num_qbt_mode1
        num_qbt_mode2 = self.num_qbt_mode2
        num_tot = self.num_tot
        num_max = max(num_qbt_mode0, num_qbt_mode1, num_qbt_mode2)
        # need to check the ordering of the weights
        weights_modes= np.zeros((num_tot, num_max,3))
        for idx in range(num_tot):
            w = np.abs(np.array(evecs[idx].data.todense()))**2.

            w.shape = (num_qbt_mode0, num_qbt_mode1, num_qbt_mode2)
            w_mode0 = np.sum(w, axis=(1,2))
            w_mode1 = np.sum(w, axis=(2,0))
            w_mode2 = np.sum(w, axis=(0,1))
            w_mode0.resize((num_max,),refcheck=False)
            w_mode1.resize((num_max,),refcheck=False)
            w_mode2.resize((num_max,),refcheck=False)

            weights_modes[idx,:,0] = w_mode0
            weights_modes[idx,:,1]  = w_mode1
            weights_modes[idx,:,2]  = w_mode2
        return weights_modes
    
    def wavefunctions(self, level, theta1_arr, theta2_arr, phi_arr):

        def one_over_sqrt_factorial(lvl):
            if lvl <= 170:
                return np.math.factorial(lvl)**-.5
            else:
                return (np.math.factorial(170)**-.5
                        * (np.math.factorial(lvl) 
                           / np.math.factorial(170))**-.5)
        
        def ho_wf(xp, lvl, zpf):
            # Calcultes wave functions of the harmonic oscillator.
            coeff = (2**lvl * np.sqrt(np.pi) * zpf)**-.5
            coeff *= one_over_sqrt_factorial(lvl)
            coeff *= np.exp(-.5 * (xp / zpf)**2) 
            return coeff * hpoly(lvl, xp / zpf)
        
        coeffs = np.array(self.states()[level].data.todense())
        coeffs.shape = (self.num_qbt_mode0, self.num_qbt_mode1, self.num_qbt_mode2)

        wavefunction = np.zeros((theta1_arr.size, theta2_arr.size, phi_arr.size ))

        for i in range(self.num_qbt_mode0):
            for j in range(self.num_qbt_mode1): 
                for k in range(self.num_qbt_mode2):
                   wf_mode = coeffs[i, j, k] * (
                    self.Fluxonium_theta1.wavefunc(j, theta1_arr)[:, np.newaxis, np.newaxis] *
                    self.Fluxonium_theta2.wavefunc(k, theta2_arr)[np.newaxis, :, np.newaxis] *
                    ho_wf(phi_arr, i, (4*self.E_C / self.E_L)**.25)[np.newaxis, np.newaxis, :] )
                   wavefunction = wavefunction + wf_mode

        return wavefunction

    def compute_wavefunction(self, level, theta1_arr, theta2_arr, phi_arr):
        
        def one_over_sqrt_factorial(lvl):
            if lvl <= 170:
                return np.math.factorial(lvl)**-.5
            else:
                return (np.math.factorial(170)**-.5
                        * (np.math.factorial(lvl) 
                           / np.math.factorial(170))**-.5)
        
        def ho_wf(xp, lvl, zpf):
            # Calcultes wave functions of the harmonic oscillator.
            coeff = (2**lvl * np.sqrt(np.pi) * zpf)**-.5
            coeff *= one_over_sqrt_factorial(lvl)
            coeff *= np.exp(-.5 * (xp / zpf)**2) 
            return coeff * hpoly(lvl, xp / zpf)
        
        coeffs = np.array(self.states()[level].data.todense())
        coeffs.shape = (self.num_qbt_mode0, self.num_qbt_mode1, self.num_qbt_mode2)

        # Compute wavefunctions for each mode
        wf_phi = np.array([ho_wf(phi_arr, i, (4 * self.E_C / self.E_L)**.25) for i in range(self.num_qbt_mode0)])
        wf_theta1 = np.array([self.Fluxonium_theta1.wavefunc(j, theta1_arr) for j in range(self.num_qbt_mode1)])
        wf_theta2 = np.array([self.Fluxonium_theta2.wavefunc(k, theta2_arr) for k in range(self.num_qbt_mode2)])

        # Initialize wavefunction array
        wavefunction = np.zeros((theta1_arr.size, theta2_arr.size, phi_arr.size), dtype=np.complex_)

        # Compute the wavefunction using broadcasting
        for i in range(self.num_qbt_mode0):
            for j in range(self.num_qbt_mode1): 
                for k in range(self.num_qbt_mode2):
                    wf_mode = (coeffs[i, j, k] * 
                            wf_theta1[j, :, np.newaxis, np.newaxis] * 
                            wf_theta2[k, np.newaxis, :, np.newaxis] * 
                            wf_phi[i, np.newaxis, np.newaxis, :])
                    wavefunction += wf_mode

        return wavefunction


class Kite_Fluxonium(Model):
    """
    Fluxonium Hamiltonian flux-coupled to a chain and a resonator
    modes.
    """

    def __init__(self, params):
        for key in ['E_C', 'E_J', 'E_L', 'E_CJ', 'E_Ls', 'eps_c', 'eps_j', 'eps_l', 'theta_ext', 'phi_ext',
                     'num_osc', 'num_qbt_1', 'num_qbt_2', 'num_qbt_3','num_tot']:
            if key not in params:
                raise ValueError('%s should be specified.' % key)

        # Specify the inductive energy of each inductance.
        self.E_L = params['E_L']
        # Specify the charging energy of each JJ.
        self.E_CJ = params['E_CJ']
        # Specify the Josephson energy of each JJ.
        self.E_J = params['E_J']
        # Specify the charging energy of the shunting capacitance.
        self.E_C = params['E_C']
        # Specify the inductive energy of the shunting inductance.
        self.E_Ls = params['E_Ls']
        # Asymmetry in charging energy between 2 JJ 
        self.eps_c = params['eps_c']
        # Asymmetry in Josephson energy between 2 JJ
        self.eps_j = params['eps_j']
        # Asymmetry in inductive energy between 2 inductances
        self.eps_l = params['eps_l']

        # Specify the theta_ext defined as a fraction of Phi_0
        self.theta_ext = np.array([params['theta_ext']]).flatten()[0]

        # Specify the phi_ext defined as a fraction of Phi_0.
        self.phi_ext = np.array([params['phi_ext']]).flatten()[0]
        # Specify the number of states in the oscillator basis.
        self.num_osc = params['num_osc']
        # Specify the number of states in the qubit basis for mode 1.
        self.num_qbt_mode1 = params['num_qbt_1']
        # Specify the number of states in the qubit basis for mode 2.
        self.num_qbt_mode2 = params['num_qbt_2']
        # Specify the number of states in the qubit basis for mode 0 (slow).
        self.num_qbt_mode0 = params['num_qbt_0']

        # Create instances for each mode of the circuit:
        # Fluxonium 1 is mode theta 1 with the flux dependency 
        params_temp = params.copy()
        params_temp['phi_ext'] = self.phi_ext 
        fluxonium_1 = Fluxonium(param_kite2fluxonium(params_temp,1))
        # Fluxonium 2 is mode theta 2
        params_temp['theta_ext'] = self.theta_ext 
        fluxonium_2 = Fluxonium(param_kite2fluxonium(params_temp,2))

        # ZPF and freq of the slow Phi mode 
        SM_phi_zpf = (8. * self.E_C / (self.E_Ls + 2*self.E_L))**.25 
        SM_f= (8* self.E_C * (self.E_Ls + 2*self.E_L) )**.5
        # Mode Phi (slow mode of the shunting capacitance/inductance)
        SM = HarmonicOscillator(frequency=SM_f,
                                       num_osc=params['num_qbt'])
        
        self.Fluxonium_theta1 = fluxonium_1
        self.Fluxonium_theta2 = fluxonium_2
        self.slow_mode = SM

        num_tot = int(params['num_tot'])
        if num_tot >  self.num_qbt_mode0*self.num_qbt_mode1*self.num_qbt_mode2:
            raise ValueError('The number of levels is too high.')
        self._num_tot = num_tot

        H = qt.tensor(fluxonium_1.H(), fluxonium_2.eye(), SM.eye())
        H += qt.tensor(fluxonium_1.eye(), fluxonium_2.H(), SM.eye())
        H += qt.tensor(fluxonium_1.eye(), fluxonium_2.eye(), SM.H())

        # Add inductive coupling between mode theta1 and mode phi
        H += (params['E_L'] + params['eps_l'] ) * qt.tensor(fluxonium_1.phi(),
                                         fluxonium_2.eye(),
                                         SM_phi_zpf * 2**-.5 * (SM.b() + SM.b().dag()))
        
        # Add inductive coupling between mode theta2 and mode phi
        H += (params['E_L'] - params['eps_l'] ) * qt.tensor(fluxonium_1.eye(),
                                         fluxonium_2.phi(),
                                         SM_phi_zpf * 2**-.5 * (SM.b() + SM.b().dag()))

        self._hamiltonian = H
        self._reset_cache()
        

def param_kite2fluxonium(params,mode):
    params_1 = params.copy()
    if mode == 0:
        params_1['E_C'] = params['E_C']
        params_1['E_L'] = 2*params['E_L']
        params_1['num_osc'] = params['num_qbt_0']

    if mode == 1:
        params_1['E_J'] = params['E_J'] + params['eps_j']
        params_1['E_C'] = params['E_CJ'] + params['eps_c']
        params_1['E_L'] = params['E_L'] + params['eps_l']
        params_1['num_qbt'] = params_1['num_qbt_1']
        params_1['phi_ext'] = -params['phi_ext']

    if mode == 2:
        params_1['E_J'] = params['E_J'] - params['eps_j']
        params_1['E_C'] = params['E_CJ'] - params['eps_c']
        params_1['E_L'] = params['E_L'] - params['eps_l']
        params_1['num_qbt'] = params_1['num_qbt_2']

        if 'theta_ext'in params:
            params_1['phi_ext']=['theta_ext']
        else:
            params_1['phi_ext']=   0*params['phi_ext']/2  #

    return params_1