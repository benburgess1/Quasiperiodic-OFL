import numpy as np
import Dark_Approximant as DA
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Approximant'))
import Approximant_Bandstructure as ABS
import Approximant_Vectors as AV
import Approximant_Curvature as AC

### ---- System Parameters ---- ###
a = 3
R = 5
l = np.arange(R)
G_vects_exact = np.column_stack((np.cos(2*np.pi*l/R),
                                np.sin(2*np.pi*l/R)))
G_vects = AV.square_approximant(a=a, G_vects=G_vects_exact)
V1_vals = np.array([10.0])
V0_vals = np.array([0.])
p1 = 1
p2 = 0
phi1 = 0
phi2 = 0
phi_up, phi_down = DA.calc_phi(R, p1, p2, phi1, phi2)
max_idx = 25
### -------------------------- ###

### -- Calculation Settings -- ###
cutoff = 5.5
basis = ABS.calc_square_basis_states(a=a, cutoff=cutoff)
dE = 0.001
sparse = True
num_evals = 30
N_q = 31
gauge_idx = 0
calc_idx = True
E_min = 0.5
E_max = 2.5
compensate_V = True
compensate_filename = 'Data/B_eff/dV_extended_R5_a3_M4000_p11_p20_phi10_phi20.npz'
### -------------------------- ###


for i in range(len(V1_vals)):
    V1 = V1_vals[i]
    V0 = V0_vals[i]

    print(f'Evaluating parameter set {i+1} out of {len(V1_vals)}:')
    print(f'R = {R}, V1 = {V1}, V0 = {V0}, p1 = {p1}, p2 = {p2}, phi1 = {phi1}, phi2 = {phi2}, a = {a}, cutoff = {cutoff}')
    print(f'Basis Size: {basis[0].shape[0]} x 2')
    print('Performing diagonalization:')
    qx_vals = np.linspace(-0.5/a, 0.5/a, N_q)
    qy_vals = np.copy(qx_vals)
    E_vals, evects_arr = DA.calc_BS_surface(qx=qx_vals, qy=qy_vals, basis=basis, G_vects=G_vects, R=R,
                                            V1=V1, V0=V0, phi_up=phi_up, phi_down=phi_down, 
                                            sparse=sparse, num_evals=num_evals, return_evects=True,
                                            compensate_V=compensate_V, filename=compensate_filename)
    
    print('Calculating density of states...')
    dos_vals, E_bins = ABS.calc_DoS(E_vals=E_vals, dE=dE)
    print('Done')

    if calc_idx:
        max_idx = ABS.calc_max_idx(E_vals, E_min=E_min, E_max=E_max)
    curv_vals, C = AC.calc_curvature_NonAb_fromfile(evects_arr=evects_arr, 
                                              calc_chern=True, 
                                              n_vals=np.arange(int(max_idx+1)), save=False,
                                              gauge_idx=gauge_idx)

    # file_str = 'DarkState/Data/5Fold/Data_'
    file_str = 'Data/5Fold/Updated/Data_'
    file_str += ('R' + str(int(R)) + '_a' + str(int(a)) + '_c' + str(np.round(cutoff,1)) + '_V1' + str(np.round(V1,4)) +  
                 '_V0' + str(np.round(V0,4)) + '_p1' + str(np.round(p1,4)) + '_p2' + str(np.round(p2,4)) + 
                 '_phi1' + str(np.round(phi1,4)) + '_phi2' + str(np.round(phi2,4)) + '.npz')
    np.savez(file_str, qx_vals=qx_vals, qy_vals=qy_vals, 
             E_vals=E_vals, dos_vals=dos_vals, E_bins=E_bins, dE=dE,
             curv_vals=curv_vals, C=C, gauge_idx=gauge_idx, max_idx=max_idx,
             V1=V1, V0=V0, p1=p1, p2=p2, phi1=phi1, phi2=phi2, R=R,
             basis=basis, a=a, cutoff=cutoff, 
             compensate_V=compensate_V, compensate_filename=compensate_filename)


