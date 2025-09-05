import numpy as np
import Calc_Bandstructure as CBS
import Calc_Curvature as CC


U_vals = np.array([0.2])
V_vals = np.array([0.15])
U_vals = np.round(U_vals, 5)
V_vals = np.round(V_vals, 5)
N_vals = 3 * np.ones(U_vals.size, dtype=np.int32)

orders = 10
cutoff = 3.5
print('Calculating basis states... ', end='', flush=True)
f = 'Data/Basis_o10.npz'
data = np.load(f)
b_full = data['basis']
basis = CBS.calc_basis_states(orders=orders, cutoff=cutoff, basis=b_full) 
# basis = CBS.calc_basis_states_alt(orders=orders, cutoff=cutoff, basis=None) 
# basis = CBS.calc_basis_states(orders=orders, cutoff=cutoff, basis=None) 
# b_full = CBS.calc_basis_states(orders=orders, cutoff=None)
# np.savez('Basis_o10.npz', basis=b_full, orders=10, cutoff=None)
# basis = CBS.calc_basis_states(orders=orders, cutoff=cutoff, basis=b_full) 
print('Done')
# G_vects = AV.square_approximant(a=a)
dE = 0.001

print('Evaluating index map... ')
idx_map = CBS.generate_map(basis=basis)
print('Done')

for i in range(len(U_vals)):
    U0 = U_vals[i]
    V0 = V_vals[i]
    V = V0*np.ones(5)
    N = N_vals[i]

    print(f'Evaluating parameter set {i+1} out of {len(U_vals)}:')
    print(f'U = {U0}, V = {V0}, N = {N}, orders = {orders}, cutoff = {cutoff}')

    qx_vals = np.linspace(-0.062, 0.062, 31)
    qy_vals = np.copy(qx_vals)

    sparse = True
    num_evals = 320
    E_vals, evects_arr = CBS.calc_BS_surface(qx=qx_vals, qy=qy_vals, basis=basis, 
                                             U0=U0, V=V, N=N, sparse=sparse, num_evals=num_evals, 
                                             return_evects=True, idx_map=idx_map)
    
    print('Calculating density of states... ', end='', flush=True)
    dos_vals, E_bins = CBS.calc_DoS(E_vals=E_vals, dE=dE)
    print('Done')

    gauge_idx = 0
    # max_idx = CBS.calc_max_idx(E_vals=E_vals, dos_vals=dos_vals, E_bins=E_bins, 
    #                            E_min=-1., E_max=1.)
    # idx_vals = np.array([2, 12, 12, 32, 62, 72, 102, 126, 162, 202])
    # max_idx = idx_vals[orders-1] - 1
    max_idx = 201
    curv_vals, C = CC.calc_curvature_NonAb_fromfile(evects_arr=evects_arr, 
                                                    calc_chern=True, save=False,
                                                    gauge_idx=gauge_idx,
                                                    n_vals=np.arange(max_idx+1),
                                                    inside_QBZ=True, invert='both',
                                                    qx_vals=qx_vals,
                                                    qy_vals=qy_vals,
                                                    shift_q=True)

    file_str = 'Data/RQBZData_'
    file_str += 'o' + str(int(orders)) 
    if cutoff is not None:
        file_str += '_c' + str(cutoff)
    file_str += ('_U' + str(np.round(U0,5)) +  '_N' 
                + str(int(np.round(N))) + '_V' + str(np.round(V0,5)))
    file_str += '_extended.npz'
    np.savez(file_str, qx_vals=qx_vals, qy_vals=qy_vals, 
             E_vals=E_vals, dos_vals=dos_vals, E_bins=E_bins, dE=dE,
             curv_vals=curv_vals, C=C, gauge_idx=gauge_idx, max_idx=max_idx,
             U0=U0, N=N, V0=V0, V=V, 
             basis=basis, orders=orders, cutoff=cutoff, 
             sparse=sparse, num_evals=num_evals)
            #  method='Fukui_NonAb'
             


