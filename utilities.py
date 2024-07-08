import os
import sys
#import h5py
import numpy as np

# from devices import devices
from local_settings import samples_path
from simple_hamiltonians import Fluxonium, Transmon, Kite_Transmon_Full
from model_hamiltonians import (UncoupledOscillators,
                                FluxoniumCoupledToOscillators,FluxoniumChargeCoupledToResonator,
                                TransmonChargeCoupledToResonator, Kite_transmon, Kite_Fluxonium)

units = {
        'E_L': 'GHz',
        'E_C': 'GHz',
        'E_J': 'GHz',
        'E_B': 'GHz',
        'E_Bs': 'GHz',
        'offset': 'GHz',
        'phi_ext': 'Phi_0',
        'phi_wf': 'Phi_0',
        'n_g': '2e',
        'f_m': 'GHz',
        'f_m1': 'GHz',
        'f_m2': 'GHz',
        'f_r': 'GHz',
        'g_m_J': 'GHz',
        'g_r_J': 'GHz',
        'g_m_r': 'GHz',
        'g_m1_J': 'GHz',
        'g_m1_r': 'GHz',
        'g_m2_J': 'GHz',
        'g_m2_r': 'GHz',
        'g_m1_m2': 'GHz',
        'frequencies': 'GHz',
        'couplings': 'GHz',
        'n_couplings': 'GHz',
        'phi_couplings': 'GHz',
        'n_cross_couplings': 'GHz',
        'phi_cross_couplings': 'GHz',
        'cutoff_cpl': 'GHz',
        'levels': 'GHz',
        'weights': '',
        'E_C_m': 'GHz',
        'E_L_m': 'GHz',
        'E_C_m1': 'GHz',
        'E_L_m1': 'GHz',
        'E_C_m2': 'GHz',
        'E_L_m2': 'GHz',
        'E_C_r': 'GHz',
        'E_L_r': 'GHz',
        'Ea_C': 'GHz',
        'Ea_g': 'GHz',
        'Ea_J': 'GHz',
        'Ephi_C_tilde': 'GHz',
        'Eb_g': 'GHz',
        'LJ': 'nH',
        'L': 'nH',
        'L1': 'nH',
        'L2': 'nH',
        'L3': 'nH',
        'M': 'nH',
        'Lr': 'nH',
        'CJ': 'fF',
        'Cm': 'fF',
        'Cm1': 'fF',
        'Cm2': 'fF',
        'Cr': 'fF',
        'I0': 'uA',
        'mu': '',
        'rho': '',
        'absolute_error': 'GHz',
        'relative_error': '',
        'run_time': 's',
        'N': '',
        'N_chain': '',
        'readout_Q_ext': '',
        'readout_Q_int': '',
        'readout_kappa': 's^-1',
        'readout_frequency': 'GHz',
        'T': 'K',
        'Q_dielectric': '',
        'Gamma1_P': 's^-1',
        'Gamma1_QP': 's^-1',
        'Gamma1_DL': 's^-1',
        'Gamma2_CQPS': 's^-1',
        'E_Ls': 'GHz',
        'E_Js': 'GHz',
        'phi_sq': '',
        'n_sq': '',
        'wfs_flx': 'Phi_0^(-1/2)',
        'cos_phi_phi_ext': '',
        'cos_phi_n_g': '',
        'sin_phi_phi_ext': '',
        'phi_01': '',
        'n_01': '',
        'half_phi_01': '',
        'sin_half_phi_01': '',
        'b_01': '',
        'overlap_01': '',
        'overlap': '',
        'frequency_01': 'GHz',
        'phi_ext_01': 'Phi_0',
        'T1': 'us',
        'T1_err': 'us',
        'T2R': 'us',
        'T2R_err': 'us',
        'T2SE': 'us',
        'T2SE_err': 'us',
        'Qint': '',
        'Qext': '',
        'readout_frequency': 'GHz',
        'tan_delta_C': '',
        'Tdl': 'K',
        'Tps': 'K',
        'xqp_slip': '',
        'xqp_chain': '',
        }
        
arrays = [
        'frequencies',
        'levels',
        'phi_ext',
        'phi_wf',
        'n_g',
        'E_Bs',
        'weights',
        'wfs_flx',
        'num_mod',
        'phi_couplings',
        'n_couplings',
        'phi_cross_couplings',
        'n_cross_couplings',
        'Gamma1_P',
        'Gamma1_QP',
        'Gamma1_DL',
        'Gamma2_CQPS',
        'E_Ls',
        'E_Js',
        'phi_sq',
        'n_sq',
        'cos_phi_phi_ext',
        'cos_phi_n_g',
        'sin_phi_phi_ext',
        'phi_01',
        'n_01',
        'half_phi_01',
        'sin_half_phi_01',
        'b_01',
        'overlap_01',
        'overlap',
        'frequency_01',
        'phi_ext_01',
        'T1',
        'T1_err',
        'T2R',
        'T2R_err',
        'T2SE',
        'T2SE_err'
        ]
        
def matrix2vector(matrix):
    n_modes = matrix.shape[0]
    vector = np.zeros(int(n_modes * (n_modes - 1) / 2))
    start = 0
    for shift in range(1, n_modes):
        diag_length = n_modes - shift
        vector[start:start+diag_length] = np.diag(matrix, shift)
        start += diag_length
    return vector
    
def vector2matrix(vector):
    n_modes = int((1 + np.sqrt(1 + 8 * vector.size)) / 2)
    matrix = np.zeros((n_modes, n_modes))
    idx = 0
    for shift in range(1, n_modes):
        for diag_idx in range(n_modes - shift):
            matrix[diag_idx,shift+diag_idx] = vector[idx]
            idx += 1
    return matrix

def get_device(sample):
    try:
        device = devices[sample]
        device['sample'] = sample
        return device
    except KeyError:
        raise KeyError("Device '%s' could not be found." % sample)
    
def get_fit(sample, fit='fluxonium fit'):
    partpath = get_device(sample)[fit]
    if partpath is None:
        raise ValueError("Device '%s' does not have %s." % (sample, fit))
    fullpath = os.path.join(samples_path, sample, partpath)
    return load_fit(fullpath)
    
def get_frequency(sample, fit='fluxonium fit', phi_ext=0., i=0, f=1):
    partpath = get_device(sample)[fit]
    if partpath is None:
        raise ValueError("Device '%s' does not have %s." % (sample, fit))
    fullpath = os.path.join(samples_path, sample, partpath)
    fit = load_fit(fullpath)
    phis_ext = fit['phi_ext']
    freqs = fit['levels'][:,f] - fit['levels'][:,i]
    return np.interp(phi_ext, phis_ext, freqs, period=1.)
    
def get_coherence(sample):
    partpath = get_device(sample)['coherence data']
    if partpath is None:
        raise ValueError("Device '%s' does not have %s." % (sample, fit))
    fullpath = os.path.join(samples_path, sample, partpath)
    return load_fit(fullpath)
    
def get_assumptions(sample):
    partpath = get_device(sample)['coherence data']
    if partpath is None:
        raise ValueError("Device '%s' does not have %s." % (sample, fit))
    partpath = partpath.replace('.hdf5', '_assumptions.hdf5')
    fullpath = os.path.join(samples_path, sample, partpath)
    return load_fit(fullpath)

def save_fit(filename, data):
    with h5py.File(filename, 'w') as f:
        if 'comment' in data:
            f.attrs['comment'] = data['comment']
            
        f.attrs['filename'] = filename

        grp_fit = f.create_group('fit')
        grp_units = grp_fit.create_group('units')

        for key in data:
            try:
                if key in arrays:
                    grp_fit.create_dataset(key, data=data[key])
                    if key in units:
                        grp_units.attrs[key] = units[key]
                else:
                    grp_fit.attrs[key] = data[key]
                    if key in units:
                        grp_units.attrs[key] = units[key]
            except:
                print("Variable '%s' has not been saved." % key)

def load_fit(filename):
    data = {}
    with h5py.File(filename, 'r') as f:
        grp_fit = f['fit']
        for key in grp_fit.attrs.keys():
            data[key] = grp_fit.attrs[key]
      
        for key in grp_fit.keys():
            if key != 'units':
                data[key] = np.array(grp_fit[key])

    return data

def print_param(name, value, array=False):
    if name.startswith('num_') or name.startswith('N'):
        if isinstance(value, (list, np.ndarray)):
            print('%s: %s' % (name, value))
        else:
            print('%s: %d' % (name, value))
        return
    space = ' '
    if name in units:
        unit = units[name]
    else:
        unit = ''
    if name == 'absolute_error':
        value *= 1.e3
        unit = 'MHz'
    if name == 'relative_error':
        value *= 1.e2
        space = ''
        unit = '%'
    if name == 'filename':
        print(filename)
    if not array:
        if unit != '':
            if value < 1.e4:
                print('%s: %.4f%s%s' % (name, value, space, unit))
            else:
                print('%s: %.3e%s%s' % (name, value, space, unit))
        else:
            try:
                if value < 1.e4:
                    print('%s: %.4f' % (name, value))
                else:
                    print('%s: %.3e' % (name, value))
            except TypeError:
                if value < 1.e4:
                    print('%s: %.4f%+.4fj' % (name, value.real, value.imag))
                else:
                    print('%s: %.3e%+.3ej' % (name, value.real, value.imag))
    elif isinstance(value, np.ndarray) and len(np.shape(value)) == 2:
        print('%s:\n%s%s%s' % (name, value, space, unit))
    elif len(value):
        if unit != '':
            print('%s: %s%s%s' % (name, value, space, unit))
        else:
            print('%s: %s' % (name, value))
    else:
        print('%s: None' % name)

def print_params(params):
    print('===Parameters===')
    for key, value in params.items():
        if isinstance(value, np.ndarray) or isinstance(value, list):
            value = np.array([value]).squeeze()
            if value.size == 1:
                try:
                    print_param(key, value[0])
                except IndexError:
                    print_param(key, value)
            elif key in arrays and value.size < 16:
                print_param(key, value, array=True)
        elif isinstance(value, list):
            print_param(key, value, array=True)
        elif isinstance(value, str):
            print('%s: %s' % (key, value))
        elif not isinstance(value, list):
            print_param(key, value)
    print('================')

def compute_fluxonium_spectrum(params):
    # Compute energy levels.
    qubit = Fluxonium(params)

    phi_ext = params['phi_ext']
    levels = np.zeros((phi_ext.size, params['num_qbt']))

    for idx_phi_ext, value in enumerate(phi_ext):
        qubit.phi_ext = value
        levels[idx_phi_ext] = qubit.levels()
        sys.stdout.write('Progress: %5.1f%%\r'
                % (100. * (idx_phi_ext + 1) / len(phi_ext)))

    params['levels'] = levels

    return params

def compute_fluxonium_coupled_to_oscillators_spectrum(params):
    # Compute energy levels.
    qubit = Fluxonium(params)

    phi_ext = params['phi_ext']
    levels = np.zeros((phi_ext.size, params['num_tot']))
    weights = np.zeros((phi_ext.size, params['num_tot'], params['num_qbt']))

    oscillators = UncoupledOscillators(params)
    for idx_phi_ext, value in enumerate(phi_ext):
        qubit.phi_ext = value
        system = FluxoniumCoupledToOscillators(qubit, oscillators, params)
        levels[idx_phi_ext] = system.levels()
        weights[idx_phi_ext] = system.weights()
        sys.stdout.write('Progress: %5.1f%%\r'
                % (100. * (idx_phi_ext + 1) / len(phi_ext)))

    params['levels'] = levels
    params['weights'] = weights

    return params

#### Bad code starts here

def labels(osc_num, lvls, weight, system, params, oscillators):
    res = np.zeros((len(lvls),2,osc_num+1))
    weights_mode_array = weights_modes(system, params)
    mode_ref = uncoupled_oscillators_basis(params, oscillators)

    for lvl_number in range(len((lvls))):
        res[lvl_number][0][0] = np.argmax(weight[lvl_number][:])
        res[lvl_number][1][0] = np.amax(weight[lvl_number][:])

        max_mode = np.argmax(weights_mode_array[lvl_number][:])
        max_mode_val = np.amax(weights_mode_array[lvl_number][:])

        for n in range(osc_num):
            res[lvl_number][0][n+1] = mode_ref[max_mode][n]
            res[lvl_number][1][n+1] = max_mode_val
    
    return res

def uncoupled_oscillators_basis(params,oscillators):
    # valid for n=2 only rn
    freqs = np.array(params['frequencies'])*1000
    res = np.zeros((params['num_cpl'],len(freqs)))
    ham = np.diag(oscillators.H(),0)*1000
    
    for i, f_ham in enumerate(ham) :
        n = round(np.amax(np.abs(f_ham)/freqs)) + 1
        matrix = np.array([[freqs[0] * k + freqs[1] * l for l in range(n)] for k in range(n)])
        res[i][:]= np.argwhere( np.abs(matrix - f_ham) < 1e-5)[0] # selecting only the first one, no handle of degeneracies
    
    return res

def weights_modes(system, params):
    evecs = system._spectrum()[1]
    num_qbt = system._fluxonium.num_qbt
    num_cpl = params['num_cpl']
    weights_modes= np.zeros((system._num_tot, num_cpl))
    for idx in range(system._num_tot):
        w = np.abs(np.array(evecs[idx].data.todense()))**2.
        w.shape = (num_qbt, -1)
        weights_modes[idx] = np.sum(w, axis=0)
    return weights_modes

def fluxonium_coupled_to_oscillators_spectrum_labels(params):
    # Labels the eigenenergies wrt quantum numbers of basis

    # Compute energy levels.
    qubit = Fluxonium(params)

    phi_ext = params['phi_ext']
    num_osc = len(params['frequencies'])

    levels = np.zeros((phi_ext.size, params['num_tot']))
    weights = np.zeros((phi_ext.size, params['num_tot'], params['num_qbt']))
    label = np.zeros((phi_ext.size, params['num_tot'], 2, num_osc+1))

    oscillators = UncoupledOscillators(params,pairwise=False)

    for idx_phi_ext, value in enumerate(phi_ext):
        qubit.phi_ext = value
        system = FluxoniumCoupledToOscillators(qubit, oscillators, params)
        levels[idx_phi_ext] = system.levels()
        weights[idx_phi_ext] = system.weights()
        label[idx_phi_ext] = labels(num_osc,levels[idx_phi_ext],weights[idx_phi_ext],system,params,oscillators)
        sys.stdout.write('Progress: %5.1f%%\r'
                % (100. * (idx_phi_ext + 1) / len(phi_ext)))
        
    params['levels'] = levels
    params['weights'] = weights

    return label

def energy_find(array,levels,array_target):
    energy = np.zeros(len(levels[:,0]))
    idx = -1
    for idx_phi_ext in range(len(levels[:,0])):
        for lvl_index, lvl in enumerate(levels[idx_phi_ext,:]):
            if np.array_equal(array[idx_phi_ext,lvl_index,0,:],array_target):
                energy[idx_phi_ext]=lvl
                idx = lvl_index
    return energy, idx

def nonlinearities_uncoupledbasis_compute(l,array,levels,omega_a,omega_b):
    El00 , _ = energy_find(array,levels,np.array([l,0,0]))
    El10 , _ = energy_find(array,levels,np.array([l,1,0]))
    El01 , _ = energy_find(array,levels,np.array([l,0,1]))
    El11 , _ = energy_find(array,levels,np.array([l,1,1]))
    El20 , _ = energy_find(array,levels,np.array([l,2,0]))
    El02 , _ = energy_find(array,levels,np.array([l,0,2]))
    
    eta_a_l = El20 - 2*El10 + El00    
    khi_a_l =  El10 - El00 - omega_a - eta_a_l

    eta_b_l = El02 - 2*El01 + El00
    khi_b_l =  El01 - El00 - omega_b - eta_b_l
    
    ksi_ab_l = El11 - El10 - El01 + El00

    return eta_a_l, khi_a_l, eta_b_l, khi_b_l, ksi_ab_l

def fluxonium_coupled_to_one_oscillator_spectrum_labels(params):
    # Labels the eigenenergies wrt quantum numbers of basis

    # Compute energy levels.
    qubit = Fluxonium(params)

    qubit.phi_ext = params['phi_ext']

    levels = np.zeros(params['num_tot'])
    weights = np.zeros((params['num_tot'], params['num_qbt']))
    label = np.zeros((params['num_tot'], 2, 2))

    system = FluxoniumChargeCoupledToResonator(qubit, params)
    levels = system.levels()
    weights= system.weights()
    label = labels_one_mode(levels,weights,system,params)
        
    params['levels'] = levels
    params['weights'] = weights

    return label

def labels_one_mode(lvls, weight, system, params):
    res = np.zeros((len(lvls),2,2))
    evecs = system._spectrum()[1]
    num_qbt = params['num_qbt']
    num_cpl = params['num_res']
    weights_modes= np.zeros((system._num_tot, num_cpl))
    for idx in range(system._num_tot):
        w = np.abs(np.array(evecs[idx].data.todense()))**2.
        w.shape = (num_qbt, -1)
        weights_modes[idx] = np.sum(w, axis=0)
    
    for lvl_number in range(len((lvls))):
        res[lvl_number][0][0] = np.argmax(weight[lvl_number][:])
        res[lvl_number][1][0] = np.amax(weight[lvl_number][:])

        res[lvl_number][0][1] = np.argmax(weights_modes[lvl_number][:])
        res[lvl_number][1][1] = np.amax(weights_modes[lvl_number][:])

    return res

def nonlinearities_uncoupledbasis_onemode_compute(l,array,levels,omega_a):
    El0 = 0
    El1 = 0
    El2 = 0 
    lvl_index = 0
    while El0 == 0 or El1 == 0 or El2 == 0:
        if np.array_equal(array[lvl_index,0,:],[l,0]):
            El0=levels[lvl_index]
        if np.array_equal(array[lvl_index,0,:],[l,1]):
            El1=levels[lvl_index]
        if np.array_equal(array[lvl_index,0,:],[l,2]):
            El2=levels[lvl_index]
        lvl_index+=1
        if lvl_index >= len(levels)-1:
            if El0 == 0:
                El0 = float("nan")
            if El1 == 0:
                El1 = float("nan")
            if El2 == 0:
                El2 = float("nan")
            break

    
    eta_a_l = El2 - 2*El1 + El0    
    khi_a_l =  El1 - El0 - omega_a - eta_a_l

    return eta_a_l, khi_a_l

def transmon_coupled_to_one_oscillator_spectrum_labels(params):
    # Labels the eigenenergies wrt quantum numbers of basis

    # Compute energy levels.
    qubit = Transmon(params)

    qubit.phi_ext = params['phi_ext']

    levels = np.zeros(params['num_tot'])
    weights = np.zeros((params['num_tot'], params['num_qbt']))
    label = np.zeros((params['num_tot'], 2, 2))

    system = TransmonChargeCoupledToResonator(qubit, params)
    levels = system.levels()
    weights= system.weights()
    label = labels_one_mode(levels,weights,system,params)
        
    params['levels'] = levels
    params['weights'] = weights

    return label


## code for Kite:

def compute_KiteT_spectrum(params):
    # Compute energy levels.
    

    phi_ext = params['phi_ext']
    n_g = params['n_g']
    num_max = max(params['num_qbt_0'], params['num_qbt_1'], params['num_qbt_2'])

    levels = np.zeros((phi_ext.size, n_g.size, params['num_tot_Kite']))
    weights = np.zeros((phi_ext.size,  n_g.size,
                         params['num_tot_Kite'], num_max , 3))
    label = np.zeros((phi_ext.size, n_g.size, params['num_tot_Kite'], 2, 3))

    for idx_phi_ext, value_phi in enumerate(phi_ext):
        for idx_n_g, value_ng in enumerate(n_g):
            params['phi_ext'] = value_phi
            params['n_g']  = value_ng
            qubit = Kite_transmon(params)
            levels[idx_phi_ext,idx_n_g,:] = qubit.levels()
            weights[idx_phi_ext,idx_n_g] = qubit.weights()
            label[idx_phi_ext,idx_n_g] = labels_Kite(levels[idx_phi_ext,idx_n_g],weights[idx_phi_ext,idx_n_g])
            sys.stdout.write('Progress: %5.1f%%\r'
                    % (100. * (idx_phi_ext + 1) / (len(phi_ext))))

    params['levels'] = levels
    params['weights'] = weights

    params['phi_ext'] = phi_ext
    params['n_g'] = n_g

    return params, label

def compute_KiteTFull_spectrum(params):
    # Compute energy levels.
    

    phi_ext = params['phi_ext']
    n_g = params['n_g']
    num_max = params['num_osc']

    levels = np.zeros((phi_ext.size, n_g.size, num_max))

    for idx_phi_ext, value_phi in enumerate(phi_ext):
        for idx_n_g, value_ng in enumerate(n_g):
            params['phi_ext'] = value_phi
            params['n_g']  = value_ng
            qubit = Kite_Transmon_Full(params)
            levels[idx_phi_ext,idx_n_g,:] = qubit.levels()
            sys.stdout.write('Progress: %5.1f%%\r'
                    % (100. * (idx_phi_ext + 1) / (len(phi_ext))))

    params['levels'] = levels

    params['phi_ext'] = phi_ext
    params['n_g'] = n_g

    return params

def labels_Kite(lvls, weight):

    res = np.zeros((len(lvls),2,3))

    for lvl_number in range(len((lvls))):
        for i in range(3):
            res[lvl_number][0][i] = np.argmax(weight[lvl_number,:,i])
            res[lvl_number][1][i] = np.amax(weight[lvl_number,:,i])  
              
    return res