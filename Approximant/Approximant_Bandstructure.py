import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import scipy as sp
import Approximant_Vectors as AV


def calc_phi(N=1, phi0=0, R=5):
    if N == 0:
        return phi0 * np.ones(R)
    else:
        return phi0 * np.ones(R) + N * np.arange(R) * 2 * np.pi / R
    

def expi(t):
    return np.cos(t) + 1j*np.sin(t)


def calc_square_basis_states(a=3, cutoff=2.5):
    dq = 1/a
    x = np.concatenate((np.arange(0,-cutoff-dq, -dq)[::-1], 
                        np.arange(0,cutoff+dq, dq)[1:]))
    y = np.copy(x)
    # x = np.arange(-cutoff,cutoff+dq,dq)
    # y = np.arange(-cutoff,cutoff+dq,dq)
    xx,yy = np.meshgrid(x, y)
    basis = np.column_stack((xx.ravel(), yy.ravel()))
    norms = np.linalg.norm(basis, axis=1)
    mask = norms <= cutoff
    basis = basis[mask]
    return (basis, np.copy(basis))


def plot_basis_states(basis, ms=5):
    fig,ax = plt.subplots()
    (b_down, b_up) = basis
    ax.plot(b_down[:,0], b_down[:,1], marker='o', color='b', ls='', ms=ms)
    ax.plot(b_up[:,0], b_up[:,1], marker='x', color='r', ls='', ms=ms)
    ax.set_aspect('equal')
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.show()


def calc_H(q, basis=calc_square_basis_states(a=3, cutoff=2.5), U0=0.02, N=1, 
           phi_vals=None, G_vects=AV.square_approximant(a=3), g_vects=None, V0=0., V=None,
           idx_map=None, R=5, phi0=0.,
           **kwargs):
    (b_down, b_up) = basis
    N_q = b_down.shape[0]
    N_basis = 2*N_q
    H = np.zeros((N_basis,N_basis), dtype=np.complex128)
    if phi_vals is None:
        phi_vals = calc_phi(N=N, R=R, phi0=phi0)
    U = -U0 * expi(-phi_vals)
    Uc = np.conjugate(U)
    if g_vects is None:
        g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    if V is None:
        V = V0 * np.ones(g_vects.shape[0])
    Vc = np.conjugate(V)
    if idx_map is None:
        for i in range(N_q):
            # Kinetic energy
            H[i,i] = np.sum((q-b_up[i,:])**2)
            H[i+N_q,i+N_q] = np.sum((q-b_down[i,:])**2)
        # if idx_map is None:
            # Same-spin couplings
            for j in range(i+1, N_q):
                # Down-to-down couplings
                dq = b_down[i] - b_down[j]
                idxs = np.where(np.isclose(-g_vects, dq, atol=0.001).all(axis=1))[0]
                if idxs.size == 1:
                    l = idxs[0]
                    H[j+N_q,i+N_q] += -V[l]
                    H[i+N_q,j+N_q] += -Vc[l]
                idxs = np.where(np.isclose(g_vects, dq, atol=0.001).all(axis=1))[0]
                if idxs.size == 1:
                    l = idxs[0]
                    H[j+N_q,i+N_q] += -Vc[l]
                    H[i+N_q,j+N_q] += -V[l]
                # Up-to-up couplings
                dq = b_up[i] - b_up[j]
                idxs = np.where(np.isclose(-g_vects, dq, atol=0.001).all(axis=1))[0]
                if idxs.size == 1:
                    l = idxs[0]
                    H[j,i] += V[l]
                    H[i,j] += Vc[l]
                idxs = np.where(np.isclose(g_vects, dq, atol=0.001).all(axis=1))[0]
                if idxs.size == 1:
                    l = idxs[0]
                    H[j,i] += Vc[l]
                    H[i,j] += V[l]
            # Spin-flip couplings
            for j in range(N_q):
                dq = b_down[i] - b_up[j]
                idxs = np.where(np.isclose(-G_vects, dq, atol=0.001).all(axis=1))[0]
                if idxs.size == 1:
                    l = idxs[0]
                    H[j,i+N_q] += U[l]
                    H[i+N_q,j] += Uc[l]
    else:
        for i in range(N_q):
            # Kinetic energy
            H[i,i] = np.sum((q-b_up[i,:])**2)
            H[i+N_q,i+N_q] = np.sum((q-b_down[i,:])**2)
        H = calc_H_from_map(H, idx_map, N_q, U, Uc, V, Vc)
    
    return H


def generate_map(basis, G_vects, g_vects=None):
    ### THIS PROBABLY DOESN'T WORK WITH CORRECTED V COUPLINGS - WOULD WANT TO 
    ### STORE MULTIPLE L VALUES PER (I,J) MATRIX ELEMENT
    (b_down, b_up) = basis
    N_q = b_down.shape[0]
    N_basis = 2*N_q
    idx_map = np.full((N_basis,N_basis), np.nan)
    if g_vects is None:
        g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    rG = G_vects.shape[0]
    rg = g_vects.shape[0]
    for i in range(N_q):
        for j in range(i+1, N_q):
            dq = b_down[i] - b_down[j]
            idxs = np.where(np.isclose(-g_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j+N_q,i+N_q] = l
                idx_map[i+N_q,j+N_q] = l + rg
            idxs = np.where(np.isclose(g_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j+N_q,i+N_q] = l + rg
                idx_map[i+N_q,j+N_q] = l
            # Up-to-up couplings
            dq = b_up[i] - b_up[j]
            idxs = np.where(np.isclose(-g_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j,i] = l
                idx_map[i,j] = l + rg
            idxs = np.where(np.isclose(g_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j,i] = l + rg
                idx_map[i,j] = l
        # Spin-flip couplings
        for j in range(N_q):
            dq = b_down[i] - b_up[j]
            idxs = np.where(np.isclose(-G_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j,i+N_q] = l
                idx_map[i+N_q,j] = l + rG
    return idx_map
    


def calc_H_from_map(H, idx_map, N_q, U, Uc, V, Vc):
    ### NOT SURE IF THIS WILL WORK WITH CORRECTED V COUPLINGS ###
    ### PROBABLY NEED TO SET H TO ZEROS FIRST ###
    ### WILL COME BACK TO THIS IF I NEED TO USE IT ###
    U_ext = np.concatenate((U,Uc))
    V_ext = np.concatenate((V,Vc))
    for i in range(N_q):
        # Same-spin couplings
        for j in range(i+1, N_q):
            # Down-to-down couplings
            l = idx_map[j+N_q,i+N_q]
            if not np.isnan(l):
                # print(l)
                l = int(l)
                H[j+N_q,i+N_q] += -V_ext[l]
                H[i+N_q,j+N_q] += -np.conj(V_ext[l])
            # Up-to-up couplings
            l = idx_map[j,i]
            if not np.isnan(l):
                l = int(l)
                H[j,i] += V_ext[l]
                H[i,j] += np.conj(V_ext[l])
        # Spin-flip couplings
        for j in range(N_q):
            l = idx_map[j,i+N_q]
            if not np.isnan(l):
                l = int(l)
                H[j,i+N_q] += U_ext[l]
                H[i+N_q,j] += np.conj(U_ext[l])
    return H



def adjust_KE(H, q, basis=calc_square_basis_states(a=3, cutoff=2.5)):
    (b_down, b_up) = basis
    N_q = b_down.shape[0]
    for i in range(N_q):
        H[i,i] = np.sum((q - b_up[i,:])**2)
        H[i+N_q,i+N_q] = np.sum((q - b_down[i,:])**2)
    return H




def calc_BS_point(q=None, return_evects=False, sparse=False, num_evals=20, 
                  H=None, **kwargs):
    if H is None:
        H = calc_H(q, **kwargs)
    if sparse:
        evals, evects = sp.sparse.linalg.eigsh(H, k=num_evals, which='SA')
        sorted_indices = np.argsort(evals)
        evals = evals[sorted_indices]
        evects = evects[:, sorted_indices]
    else:
        # print(H.shape)
        evals, evects = np.linalg.eigh(H)
        # print(evals.shape,evects.shape)
    if return_evects:
        return evals, evects
    else:
        return evals
    

def calc_BS_line(q_vals, basis=calc_square_basis_states(a=3, cutoff=2.5),
                 **kwargs):
    if 'num_evals' in kwargs:
        N = kwargs.get('num_evals')
    else:
        N_q_basis = basis[0].shape[0]
        N = 2*N_q_basis
    N_q = q_vals.shape[0]
    E_vals = np.zeros((N,N_q))
    # print(f'E_vals shape: {E_vals.shape}')
    print('Evaluating Hamiltonian...')
    H = calc_H(q=q_vals[0,:], basis=basis, **kwargs)
    print('Done')
    # print(f'H shape: {H.shape}')
    for i in range(N_q):
        print(f'Evaluating q value {i+1} of {N_q}... ' + 10*' ', end='\r')
        if i != 0:
            H = adjust_KE(H, q=q_vals[i,:], basis=basis)
        E_vals[:,i] = calc_BS_point(H=H, basis=basis, **kwargs)
    print('\nDone')
    return E_vals


def calc_BS_surface(qx, qy, basis=calc_square_basis_states(a=3, cutoff=2.5),
                    return_evects=False, idx_map=None,
                    **kwargs):
    nx = qx.size
    ny = qy.size
    N_basis = 2*basis[0].shape[0]
    if 'num_evals' in kwargs and 'sparse' in kwargs and kwargs.get('sparse'):
        N_evals = kwargs.get('num_evals')
    else:
        N_evals = N_basis
        # print(N_evals)
    E_vals = np.zeros((N_evals,nx,ny))
    if return_evects:
        evects_arr = np.zeros((N_basis,N_evals,nx,ny), dtype=np.complex128)
    print('Evaluating Hamiltonian...')
    H = calc_H(q=np.array([qx[0],qy[0]]), basis=basis, idx_map=idx_map, **kwargs)
    # H = adjust_KE(H, q=np.array([qx[0], qy[0]]), basis=basis)
    print('Done')
    for i in range(nx):
        for j in range(ny):
            print(f'Evaluating q value {i*ny+j+1} out of {nx*ny}...' + 10*' ', 
                  end='\r')
            # if i == 0  and j == 0:
            H = adjust_KE(H, q=np.array([qx[i], qy[j]]), basis=basis)
            # print(H.shape)
            # print(basis[0].shape)
            if return_evects:
                E_vals[:,i,j],evects_arr[:,:,i,j] = calc_BS_point(H=H, 
                                                                  basis=basis, 
                                                                  return_evects=return_evects, 
                                                                  **kwargs)
            else:
                E_vals[:,i,j] = calc_BS_point(H=H, basis=basis, 
                                              return_evects=return_evects,
                                              **kwargs)
    print('\nDone')
    return E_vals, evects_arr


def calc_q_GXMG(q_X=None, q_M=None, a=3):
    dq = 1/a
    if q_X is None:
        q_X = np.array([dq/2, 0.])
        q_M = np.array([dq/2, dq/2])
    q_vals_GX = np.column_stack((np.linspace(0, q_X[0], 50),
                                 np.linspace(0, q_X[1], 50)))
    q_vals_XM = np.column_stack((np.linspace(q_X[0], q_M[0], 50)[1:],
                                 np.linspace(q_X[1], q_M[1], 50)[1:]))
    q_vals_MG = np.column_stack((np.linspace(q_M[0], 0, 50)[1:],
                                 np.linspace(q_M[1], 0, 50)[1:]))
    q_vals = np.concatenate((q_vals_GX, q_vals_XM, q_vals_MG))
    return q_vals


def calc_q_BZ_boundary(q0=None, a=3):
    dq = 1/a
    if q0 is None:
        q0 = np.array([dq/2, dq/2])
    R = np.array([[0, -1],
                  [1, 0]])
    q1 = R @ q0
    q2 = R @ q1
    q3 = R @ q2
    q_vals_01 = np.column_stack((np.linspace(q0[0], q1[0], 50),
                                 np.linspace(q0[1], q1[1], 50)))
    q_vals_12 = np.column_stack((np.linspace(q1[0], q2[0], 50)[1:],
                                 np.linspace(q1[1], q2[1], 50)[1:]))
    q_vals_23 = np.column_stack((np.linspace(q2[0], q3[0], 50)[1:],
                                 np.linspace(q2[1], q3[1], 50)[1:]))
    q_vals_30 = np.column_stack((np.linspace(q3[0], q0[0], 50)[1:],
                                 np.linspace(q3[1], q0[1], 50)[1:]))
    return np.concatenate((q_vals_01, q_vals_12, q_vals_23, q_vals_30))


def calc_DoS(filename=None, E_vals=None, dE=0.01):
    if filename is not None:
        data = np.load(filename)
        E_vals = data['E_vals'].flatten()
    E_min, E_max = np.min(E_vals), np.max(E_vals)
    E_edges = np.arange(E_min-dE, E_max + dE, dE)

    # Histogram: count number of states in each bin
    dos_counts, _ = np.histogram(E_vals, bins=E_edges)

    # Normalize to get DOS (per unit energy)
    dos = dos_counts / dE

    # Optional: compute bin centers for plotting
    E_bins = (E_edges[:-1] + E_edges[1:]) / 2
    # E_bins = np.arange(E_min-dE/2, E_max+3*dE/2, dE)

    # # print(E_min, E_max)
    # # print(E_bins[:5], E_bins[-5:])
    # # print(E_bins.shape)

    # # print(E_vals.shape)

    # E = 0
    # arr = ((E_vals >= (E - dE/2))
    #                        & (E_vals < (E + dE/2)))
    # # print(arr)
    # # print(np.sum(arr))
    # dos = np.array([np.sum((E_vals >= (E - dE/2))
    #                        & (E_vals < (E + dE/2)))/dE for E in E_bins])
    # dos /= dE
    
    # print(dos.shape)
    return dos, E_bins


def save_DoS(DoS_filename, E_filename, dE=0.01):
    dos_vals, E_bins = calc_DoS(E_filename, dE=dE)
    param_keys = {'U0', 'V0', 'V', 'N', 'basis', 'a', 'cutoff'}
    params = {}
    with np.load(E_filename) as data:
        for key in param_keys:
            if key in data:
                params[key] = data[key]
    np.savez(DoS_filename, dos_vals=dos_vals, E_bins=E_bins, dE=dE, 
             E_filename=E_filename, **params)
    

def calc_max_idx(E_vals, E_min, E_max):
    mask = (E_vals > E_min) & (E_vals < E_max)
    valid_slices = np.any(mask, axis=(1, 2))
    valid_indices = np.where(valid_slices)[0]
    dE_max = 0
    idx = 0
    for n in valid_indices[:-1]:
        dE = np.min(E_vals[n+1,:,:]) - np.max(E_vals[n,:,:])
        if dE > dE_max:
            dE_max = dE
            idx = n
    return idx

if __name__ == '__main__':
    print(sp.__version__)
    print(np.__version__)
    # a = 9
    # cutoff = 2.5
    # # basis = calc_square_basis_states(a=a, cutoff=cutoff)
    # G_vects = AV.square_approximant(a=a)
    # # print(basis[0].shape)
    # # plot_basis_states(basis)
    # U0 = 0.2
    # V0 = 0.15
    # N = 3

    # phi_vals = calc_phi(N=N)
    # U = -U0 * expi(-phi_vals)
    # Uc = np.conjugate(U)
    # V = V0 * np.ones(5)
    # Vc = np.conjugate(V)

    # f = 'Approximant/Data/Data_NonAb_a7_U0.2_N3_V0.15.npz'
    # data = np.load(f)
    # E_vals = data['E_vals']
    # idx = calc_max_idx(E_vals, E_min=-0.2, E_max=0.1)
    # print(idx)

    # q = np.array([0,0])
    # H = calc_H(q=q, basis=basis, U0=U0, N=N, V0=V0, G_vects=G_vects)

    # idx_map = generate_map(basis=basis, G_vects=G_vects)
    # print(idx_map)
    # print(idx_map[~np.isnan(idx_map)])
    # N_q = basis[0].shape[0]

    # H_map = calc_H_from_map(H=np.zeros((2*N_q,2*N_q), dtype=np.complex128), idx_map=idx_map, N_q=N_q, U=U, Uc=Uc, V=V, Vc=Vc)
    # H_map = adjust_KE(H_map, q=q, basis=basis)

    # print(H)
    # print(H_map)
    # d = H - H_map
    # print(d[np.abs(d)>1e-10])

    # q_vals = calc_q_GXMG(a=a, q_X=np.array([0,0.25]), q_M=np.array([0.25,0.25]))
    # # q_vals = calc_q_BZ_boundary(a=a)
    # E_vals = calc_BS_line(q_vals, basis=basis, U0=U0, V0=V0, N=N, G_vects=G_vects)
    # np.savez('Data/BS_a2_GXMG_U0.2_N3_V0.2.npz', q_vals=q_vals, E_vals=E_vals, 
    #          U0=U0, V0=V0, N=N, basis=basis, a=a, cutoff=cutoff, G_vects=G_vects)
    
    # q1 = AV.calc_midpoint(G[0,:], G[1,:])
    # print(q1)

    # q_vals = np.column_stack((np.linspace(0.9*q1[0],1.1*q1[0],10),
    #                           np.linspace(0.9*q1[1],1.1*q1[1],10)))

    # q_vals = np.column_stack((np.linspace(-1/3,1/3,200), np.zeros(200)))
    
    # E_vals = calc_BS_line(q_vals=q_vals, basis=basis, G_vects=G, U0=U0, V0=V0,
    #                       N=N)
    
    # np.savez('BS_GXGXG_U0.2_N3_V0.1.npz', q_vals=q_vals, E_vals=E_vals, basis=basis, G_vects=G,
    #          U0=U0, V0=V0, N=N, a=a, cutoff=cutoff)

    # f1 = 'Data/BS_approx_a3_surface_U200.0_N3_V150.0.npz'
    # f2 = 'DoS_approx_a3_U200.0_N3_V150.0_dE0.1.npz'
    # save_DoS(f2, f1, dE=0.0001)

    # U0 = 0.19
    # N = 3
    # V0 = 0.001
    # # q_vals = np.column_stack((np.linspace(0,1.5,100), np.zeros(100)))
    # q_vals = calc_q_GXMG()
    # E_vals = calc_BS_line(q_vals, basis=basis, U0=U0, V0=V0, N=N, G=G)
    # np.savez('BS_approx_a3_GXMG_U190.0_N3_V1.0.npz', q_vals=q_vals, E_vals=E_vals, 
    #          U0=U0, V0=V0, N=N, basis=basis, a=a, cutoff=cutoff, G=G)