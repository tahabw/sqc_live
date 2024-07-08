import numpy as np

from simple_hamiltonians import Kite_Transmon_Full

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def matrix_elements(qubit,i,j):
    return qubit.N_ij(i,j),qubit.n_sigma_ij(i,j), qubit.n_delta_ij(i,j), qubit.phi_sigma_ij(i,j), qubit.phi_delta_ij(i,j)

def matrix_elements_dephasing(qubit,i,j):
    return qubit.dephasing_op_CC_ij(i,j), qubit.dephasing_op_Chg_ij(i,j), qubit.dephasing_op_Flux_ij(i,j)
 
def process(parameters):
    eng_l, eng_cs, eng_j, eng_cj, params = parameters
    params['E_L'] = eng_l
    params['E_C'] = eng_cs
    params['E_J'] = eng_j
    params['E_CJ'] = eng_cj
    

    qubit = Kite_Transmon_Full(params)
    qubit_spectrum = qubit.levels()

    E0 = qubit_spectrum[0]
    
    new_row = {'E_L': eng_l, 'E_J': eng_j, 'E_Cs': eng_cs, 'E_Cj': eng_cj,'E0' : E0}

    state_pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]

    for (i, j) in state_pairs:
        omega_ij = qubit_spectrum[j] - qubit_spectrum[i]

        N_ij, n_sigma_ij, n_delta_ij, phi_sigma_ij, phi_delta_ij = matrix_elements(qubit, i, j)

        new_row.update({
            f'omega_{i}{j}':omega_ij,
            f'N_{i}{j}': N_ij,
            f'n_sigma_{i}{j}': n_sigma_ij,
            f'n_delta_{i}{j}': n_delta_ij,
            f'phi_sigma_{i}{j}': phi_sigma_ij,
            f'phi_delta_{i}{j}': phi_delta_ij
        })

    qubit.phi_ext = params['phi_ext']*.99999
    omega_01_dphi_m= np.abs(qubit.levels()[1] - qubit.levels()[0])
    qubit.phi_ext = params['phi_ext']*1.00001
    omega_01_dphi_p= np.abs(qubit.levels()[1] - qubit.levels()[0])

    new_row.update({
            f'omega_01_dphim':omega_01_dphi_m,
            f'omega_01_dphip':omega_01_dphi_p
        })

    return new_row

def CSV_gen(filename,EJ_EC_sweep_bool=True,EJ2_EL_sweep_bool = False):
    # Initial guess for the qubit parameters coupled to two modes
    params = {'E_C': 1.645, # The charging energy of the shunting cap
              'E_L': 2.14, # The inductive energy of the branch inductance
              'E_J': 21.43, # The Josephson energy of a JJ
              'E_CJ': 7.14, # The charging energy of a JJ
              'eps_l' : 0., # Asymmetry in inductive energy between L of each branch
              'eps_j' : 0., # Asymmetry in Josephson energy between JJ of each branch
              'eps_c' : 0., # Asymmetry in charging energy between JJ of each branch
              'phi_ext': 0.5, # External flux threading the loop
              'n_g' : 0., # Offset charge in the island
              'num_qbt_0': 30,
              'num_qbt_1': 30,
              'num_qbt_2': 20,
              'num_osc': 30,
              'error_type': 'relative_error',
              'data_set': 'data3'
            }
    

    if EJ_EC_sweep_bool:
        EL_sweep = [1,2] #  GHz
        omega_p = 25 # junction plasma freq in GHz
        EJoverEC_sweep = np.linspace(.5,6,15)
        ECJ_sweep = np.sqrt(omega_p**2/(8*EJoverEC_sweep))
        EJ_sweep = np.sqrt(omega_p**2*EJoverEC_sweep/8)
        ECs_sweep = np.linspace(20,100,5)*1e-3

    elif EJ2_EL_sweep_bool:
        
        omega_p = 25 # junction plasma freq in GHz
        EJoEC= 3 # 3 too small reduces EJ2. too large reduces Gamma (which needs to be of order EL to beat fluxonium). 
        EJ_sweep = [omega_p*np.sqrt(EJoEC)/np.sqrt(8)]
        ECJ_sweep = [omega_p/np.sqrt(EJoEC)/np.sqrt(8)]

        EL_sweep = np.linspace(.7,3,11)
        ECs_sweep = np.linspace(20,100,5)*1e-3

    else :
        raise ValueError('Sweep not defined')
    

    all_params = [(eng_l, eng_cs, eng_j, eng_cj, params) for eng_l in EL_sweep
                      for eng_cs in ECs_sweep for eng_j in EJ_sweep for eng_cj in ECJ_sweep]
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process, all_params), total=len(all_params)))

    # Convert results to a DataFrame
    res = pd.DataFrame(results)

    compression_options = dict(method='zip', archive_name=f'{filename}.csv')
    res.to_csv(f'{filename}.zip', compression=compression_options, index=False)