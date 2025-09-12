import numpy as np
import Approximant_Bandstructure as ABS
import Approximant_Vectors as AV
import Approximant_Curvature as AC

### ---- System Parameters ---- ###
N = 5
phi0 = 0.
a = 3
R = 8
l = np.arange(R)
G_vects_exact = np.column_stack((np.cos(2*np.pi*l/R),
                                np.sin(2*np.pi*l/R)))
G_vects = AV.square_approximant(a=a, G_vects=G_vects_exact)
g_vects = np.roll(G_vects, -1, axis=0) - G_vects
# g1 = np.roll(G_vects, -1, axis=0) - G_vects
# g2 = np.roll(G_vects, -2, axis=0) - G_vects
# g3 = np.roll(G_vects, -3, axis=0) - G_vects
# g_vects = np.row_stack((g1, g2, g3))

U_vals = np.array([0.3])
V_vals = np.array([0.24])
# data = np.load('Req_5Fold.npz')
# U_vals = data['P_req'][:,0]
# V_vals = data['P_req'][:,1]
# V = np.arange(0.01, 0.4, 0.02)
# U = np.arange(0., 0.41, 0.02)
# U_vals, V_vals = np.meshgrid(U, V)
# U_vals = U_vals.flatten()
# V_vals = V_vals.flatten()
# U_vals = np.array([0.15])
# V_vals = np.array([0.025])
# V_vals = [-0.15 * np.exp(-1j*phi0) * np.ones(R)]
# V_vals = [0.09 * np.exp(-1j*0.5 - 1j*phi0) * np.ones(R)]
# V_vals = [-0.0354 * np.exp(-1j*2*N*np.pi/R - 1j*phi0) * np.ones(R)]
# phi_vals = np.concatenate((-10*np.pi/8 * np.ones(8), -20*np.pi/8 * np.ones(8)))
# Vs = np.array([0.0353])
# V_vals = [np.concatenate((-V * np.exp(-1j*2*N*np.pi/R - 1j*phi0) * np.ones(R), 
#                           -V * (np.sin(4*np.pi/R)/np.sin(2*np.pi/R)) * np.exp(-1j*4*N*np.pi/R - 1j*phi0) * np.ones(R),
#                           -V * (np.sin(6*np.pi/R)/np.sin(2*np.pi/R)) * np.exp(-1j*6*N*np.pi/R - 1j*phi0) * np.ones(R)))
#                           for V in Vs]
# phi_vals = np.arange(1.1, 1.5, 0.1)
# phi_vals = np.array([2.0])
# V_vals = [0.15 * np.exp(1j * phi) * np.ones(5) for phi in phi_vals]
# N_vals = np.array([5])

idx_dict = {1:1, 2:3, 3:13, 4:27, 5:45, 6:55, 7:81 , 8:111, 9:125, 10:163}
max_idx = 13
### -------------------------- ###

### -- Calculation Settings -- ###
cutoff = 2.5
basis = ABS.calc_square_basis_states(a=a, cutoff=cutoff)
dE = 0.001
sparse = True
num_evals = 30
N_q = 201
gauge_idx = 0
calc_idx = False
use_dict_idx = False
E_min = -0.2
E_max = 0.1
### -------------------------- ###


for i in range(len(U_vals)):
    U0 = U_vals[i]
    V = V_vals[i]
    # N = N_vals[i]

    if isinstance(V, np.ndarray):
        V0 = np.abs(V[0])
    else:
        V0 = abs(V)
        V = V0 * np.ones(g_vects.shape[0])

    print(f'Evaluating parameter set {i+1} out of {len(U_vals)}:')
    print(f'U = {U0}, V = {V0}, N = {N}, a = {a}, cutoff = {cutoff}')
    # print(f'U = {U0}, V = {np.round(V,2)}, N = {N}, a = {a}, cutoff = {cutoff}')
    print(f'Basis Size: {basis[0].shape[0]} x 2')
    print('Performing diagonalization:')
    qx_vals = np.linspace(-0.5/a, 0.5/a, N_q)
    qy_vals = np.copy(qx_vals)
    E_vals, evects_arr = ABS.calc_BS_surface(qx=qx_vals, qy=qy_vals, basis=basis, 
                                 G_vects=G_vects, U0=U0, V=V/2, N=N, sparse=sparse, num_evals=num_evals,
                                 return_evects=True, R=R, g_vects=g_vects, phi0=phi0)
    
    print('Calculating density of states...')
    dos_vals, E_bins = ABS.calc_DoS(E_vals=E_vals, dE=dE)
    print('Done')

    if calc_idx:
        max_idx = ABS.calc_max_idx(E_vals, E_min=E_min, E_max=E_max)
    elif use_dict_idx:
        max_idx = idx_dict[a]
    curv_vals, C = AC.calc_curvature_NonAb_fromfile(evects_arr=evects_arr, 
                                              calc_chern=True, 
                                              n_vals=np.arange(int(max_idx+1)), save=False,
                                              gauge_idx=gauge_idx)

    # file_str = 'Approximant/Data/5Fold/Data_NonAb'
    # file_str += ('_a' + str(int(a)) + '_U' + str(np.round(U0,4)) +  '_N' 
    #             + str(int(np.round(N))) + '_V' + str(np.round(V0,4)) + '.npz')
    file_str = 'Data/8Fold/Old_Data/Data_'
    file_str += ('R' + str(int(R)) + '_a' + str(int(a)) + '_c' + str(np.round(cutoff,1)) + '_U' + str(np.round(U0,4)) +  '_N' 
                + str(int(np.round(N))) + '_V' + str(np.round(V0,4)) + '_fine.npz')
    np.savez(file_str, qx_vals=qx_vals, qy_vals=qy_vals, 
             E_vals=E_vals, dos_vals=dos_vals, E_bins=E_bins, dE=dE,
             curv_vals=curv_vals, C=C, gauge_idx=gauge_idx, max_idx=max_idx,
             U0=U0, N=N, V0=V0, V=V, R=R,
             basis=basis, a=a, cutoff=cutoff)


