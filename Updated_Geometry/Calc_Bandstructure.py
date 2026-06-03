import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import scipy as sp
from tqdm import tqdm


G0 = 1
R = 8
G_vects = G0 * np.column_stack((np.cos(np.arange(R)*2*np.pi/R),
                                np.sin(np.arange(R)*2*np.pi/R)))

g_vects = np.roll(G_vects, -1, axis=0) - G_vects


def calc_phi(N=1, phi0=0, R=5):
    if N == 0:
        return phi0 * np.ones(R)
    else:
        return phi0 * np.ones(R) + N * np.arange(R) * 2 * np.pi / R
    

def expi(t):
    return np.cos(t) + 1j*np.sin(t)


def remove_duplicates(arr, tol=0.001, compare_arr=None):
    # print(arr)
    arr_sorted = arr[np.lexsort(arr.T[::-1])]
    if compare_arr is None:
        unique_rows = [arr_sorted[0]]  # Start with the first row
        for row in arr_sorted[1:]:
            if not np.any(np.isclose(unique_rows, row, atol=tol).all(axis=1)):
                unique_rows.append(row)
    else:
        unique_rows = []  # Start with the first row
        # print(compare_arr)
        # compare_arr = np.concatenate((arr, compare_arr), axis=0)
        # print(compare_arr)
        for row in arr_sorted:
            if not np.any(np.isclose(compare_arr, row, atol=tol).all(axis=1)):
                unique_rows.append(row)
    # print(np.array(unique_rows))
    return np.array(unique_rows)


def remove_cutoff(arr, cutoff):
    norms = np.linalg.norm(arr, axis=1)
    mask = norms < cutoff
    return arr[mask]


def calc_basis_states(orders=2, G_vects=G_vects, g_vects=g_vects, basis=None, cutoff=1.6,
                      print_progress=True):
    if basis is None:
        prev_down = np.array([np.zeros(2)])
        prev_up = np.array([np.zeros(2)])
        b_down = np.array([np.zeros(2)])
        b_up = np.array([np.zeros(2)])

        for i in range(orders):
            if print_progress:
                print(f'Evaluating order {i+1} out of {orders}')
            # new_down = np.array([[]])
            # new_up = np.array([[]])
            new_down = np.array([prev_up[0,:]-G_vects[0,:]])
            new_up = np.array([prev_down[0,:]+G_vects[0,:]])
            # skip = True
            for j,point in enumerate(prev_down):
                if print_progress:
                    print(f'Evaluating down point {j+1} out of {prev_down.shape[0]}...' + 10*' ', end='\r')
                # Connect from previous spin-down to new spin-up
                for G in G_vects:
                    # if skip:
                    #     skip = False
                    # else:
                    # print(point)
                    # print(point + G)
                    # print(np.array([point+G]))
                    new_up = np.concatenate((new_up, np.array([point+G])), axis=0)
                # Connect from down-to-down with g couplings
                for g in g_vects:
                    new_down = np.concatenate((new_down, np.array([point+g])), axis=0)
                    new_down = np.concatenate((new_down, np.array([point-g])), axis=0)
            if print_progress: 
                print('')
            # skip = True
            for j,point in enumerate(prev_up):
                if print_progress:
                    print(f'Evaluating up point {j+1} out of {prev_up.shape[0]}...' + 10*' ', end='\r')
                # Connect from previous up to new down
                for G in G_vects:
                    # if skip:
                    #     skip = False
                    # else:
                        # print(np.array([point-G]))
                        # print(new_down)
                    new_down = np.concatenate((new_down, np.array([point-G])), axis=0)
                        # np.append(new_down, np.array([point-G]), axis=1)
                        # print(new_down)

                # Connect from up-to-up with g couplings
                for g in g_vects:
                    new_up = np.concatenate((new_up, np.array([point+g])), axis=0)
                    new_up = np.concatenate((new_up, np.array([point-g])), axis=0)
            new_up = new_up[1:,:]
            new_down = new_down[1:,:]
            if print_progress:
                print('\nRemoving duplicates... ', end='', flush=True)
            new_down = remove_duplicates(new_down)
            new_up = remove_duplicates(new_up)
            new_down = remove_duplicates(new_down, compare_arr=b_down)
            new_up = remove_duplicates(new_up, compare_arr=b_up)
            if print_progress:
                print('Done')
            
            # print(b_down)
            # print(new_down)
            b_down = np.concatenate((b_down, new_down), axis=0)
            # print(b_down)
            b_up = np.concatenate((b_up, new_up), axis=0)

            prev_down = new_down
            prev_up = new_up
        # print('Removing duplicates... ', end='', flush=True)
        # b_down_unique = remove_duplicates(b_down)
        # b_up_unique = remove_duplicates(b_up)
        # print('Done')
        b_up_unique, b_down_unique = b_up, b_down
    else:
        (b_up_unique, b_down_unique) = basis
    if print_progress:
        print('Applying cutoff... ', end='', flush=True)
    if cutoff is not None:
        b_down_unique = remove_cutoff(b_down_unique, cutoff)
        b_up_unique = remove_cutoff(b_up_unique, cutoff)
        # norms_down = np.linalg.norm(b_down_unique, axis=1)
        # mask = norms_down < cutoff
        # b_down_unique = b_down_unique[mask]
        # norms_up = np.linalg.norm(b_up_unique, axis=1)
        # mask = norms_up < cutoff
        # b_up_unique = b_up_unique[mask]
    if print_progress:
        print('Done')
    return (b_up_unique, b_down_unique)


def calc_basis_states_alt(orders=2, G_vects=G_vects, g_vects=g_vects, basis=None, cutoff=1.6):
    if basis is None:
        prev_down = np.array([np.zeros(2)])
        prev_up = np.array([np.zeros(2)])
        b_down = np.array([np.zeros(2)])
        b_up = np.array([np.zeros(2)])

        for i in range(orders):
            print(f'Evaluating order {i+1} out of {orders}')
            new_down_G_plus = prev_up[:, np.newaxis, :] + G_vects[np.newaxis, :, :]
            new_down_G_plus = new_down_G_plus.reshape(-1,2)
            new_down_G_minus = prev_up[:, np.newaxis, :] - G_vects[np.newaxis, :, :]
            new_down_G_minus = new_down_G_minus.reshape(-1,2)
            new_down_g_plus = prev_down[:, np.newaxis, :] + g_vects[np.newaxis, :, :]
            new_down_g_plus = new_down_g_plus.reshape(-1,2)
            new_down_g_minus = prev_down[:, np.newaxis, :] - g_vects[np.newaxis, :, :]
            new_down_g_minus = new_down_g_minus.reshape(-1,2)
            new_down = np.concatenate((new_down_G_plus, new_down_G_minus, new_down_g_plus, 
                                        new_down_g_minus), axis=0)
            
            new_up_G_plus = prev_down[:, np.newaxis, :] + G_vects[np.newaxis, :, :]
            new_up_G_plus = new_up_G_plus.reshape(-1,2)
            new_up_G_minus = prev_down[:, np.newaxis, :] - G_vects[np.newaxis, :, :]
            new_up_G_minus = new_up_G_minus.reshape(-1,2)
            new_up_g_plus = prev_up[:, np.newaxis, :] + g_vects[np.newaxis, :, :]
            new_up_g_plus = new_up_g_plus.reshape(-1,2)
            new_up_g_minus = prev_up[:, np.newaxis, :] - g_vects[np.newaxis, :, :]
            new_up_g_minus = new_up_g_minus.reshape(-1,2)
            new_up = np.concatenate((new_up_G_plus, new_up_G_minus, new_up_g_plus, 
                                        new_up_g_minus), axis=0)




            # new_down = np.array([[]])
            # new_up = np.array([[]])
            # G = G_vects[0,:]
            # p_up = prev_up[0,:]
            # p_down = prev_down[0,:]
            # new_down = np.array([p_up + G, p_up - G])
            # new_up = np.array([p_down + G, p_down - G])
            # new_up = np.concatenate((new_up, np.array([point+G])), axis=0)
            # # skip = True
            # for j,point in enumerate(prev_down):
            #     print(f'Evaluating down point {j+1} out of {prev_down.shape[0]}...' + 10*' ', end='\r')
            #     # Connect from previous spin-down to new spin-up
            #     for G in G_vects[1:,:]:
            #         # if skip:
            #         #     skip = False
            #         # else:
            #         # print(point)
            #         # print(point + G)
            #         # print(np.array([point+G]))
            #         new_up = np.concatenate((new_up, np.array([point+G])), axis=0)
            #         new_up = np.concatenate((new_up, np.array([point-G])), axis=0)
            #     # Connect from down-to-down with g couplings
            #     for g in g_vects:
            #         new_down = np.concatenate((new_down, np.array([point+g])), axis=0)
            #         new_down = np.concatenate((new_down, np.array([point-g])), axis=0)
            # print('')
            # # skip = True
            # for j,point in enumerate(prev_up):
            #     print(f'Evaluating up point {j+1} out of {prev_up.shape[0]}...' + 10*' ', end='\r')
            #     # Connect from previous up to new down
            #     for G in G_vects[1:,:]:
            #         # if skip:
            #         #     skip = False
            #         # else:
            #             # print(np.array([point-G]))
            #             # print(new_down)
            #         new_down = np.concatenate((new_down, np.array([point-G])), axis=0)
            #         new_down = np.concatenate((new_down, np.array([point+G])), axis=0)
            #             # np.append(new_down, np.array([point-G]), axis=1)
            #             # print(new_down)

            #     # Connect from up-to-up with g couplings
            #     for g in g_vects:
            #         new_up = np.concatenate((new_up, np.array([point+g])), axis=0)
            #         new_up = np.concatenate((new_up, np.array([point-g])), axis=0)
            # print('')
            print('Removing duplicates... ', end='', flush=True)
            new_down = remove_duplicates(new_down)
            new_up = remove_duplicates(new_up)
            new_down = remove_duplicates(new_down, compare_arr=b_down)
            new_up = remove_duplicates(new_up, compare_arr=b_up)
            print('Done')
            
            # print(b_down)
            # print(new_down)
            b_down = np.concatenate((b_down, new_down), axis=0)
            # print(b_down)
            b_up = np.concatenate((b_up, new_up), axis=0)

            prev_down = new_down
            prev_up = new_up
        # print('Removing duplicates... ', end='', flush=True)
        # b_down_unique = remove_duplicates(b_down)
        # b_up_unique = remove_duplicates(b_up)
        # print('Done')
        b_up_unique, b_down_unique = b_up, b_down
    else:
        (b_up_unique, b_down_unique) = basis

    print('Applying cutoff... ', end='', flush=True)
    if cutoff is not None:
        b_down_unique = remove_cutoff(b_down_unique, cutoff)
        b_up_unique = remove_cutoff(b_up_unique, cutoff)
        # norms_down = np.linalg.norm(b_down_unique, axis=1)
        # mask = norms_down < cutoff
        # b_down_unique = b_down_unique[mask]
        # norms_up = np.linalg.norm(b_up_unique, axis=1)
        # mask = norms_up < cutoff
        # b_up_unique = b_up_unique[mask]
    print('Done')
    return (b_up_unique, b_down_unique)


def decagon(a=1, r0=np.array([0,0]), color='k'):
    r1 = a * np.array([1,np.tan(np.pi/10)])
    t = np.pi/5
    R = np.array([[np.cos(t),-np.sin(t)],
              [np.sin(t),np.cos(t)]])
    vertices = np.zeros((10,2))
    vertices[0,:] = r1
    for i in range(1,10):
        vertices[i,:] = R @ vertices[i-1,:]

    vertices += np.outer(np.ones(10),r0)

    dec = Polygon(vertices, edgecolor=color, facecolor=(0,0,0,0))
    return dec


def octagon(a=1, r0=np.array([0,0]), color='k'):
    r1 = a * np.array([1,np.tan(np.pi/8)])
    t = np.pi/5
    R = np.array([[np.cos(t),-np.sin(t)],
              [np.sin(t),np.cos(t)]])
    vertices = np.zeros((8,2))
    vertices[0,:] = r1
    for i in range(1,8):
        vertices[i,:] = R @ vertices[i-1,:]

    vertices += np.outer(np.ones(8),r0)

    dec = Polygon(vertices, edgecolor=color, facecolor=(0,0,0,0))
    return dec


def plot_basis_states(basis, ms=5, plot_BZ=True, plot_QBZ=False, invert=False, plot_title=True, orders=3,
                      cutoff=None, inside_QBZ=False, R=5, exclude_down=False, axis_titles=True,
                      G_min_in_title=False):
    fig,ax = plt.subplots()
    (b_up, b_down) = basis
    if inside_QBZ:
        path_up = Polygon(calc_pentagon(G=1, invert=True), edgecolor='r', facecolor=(0,0,0,0)).get_path()
        mask_up = path_up.contains_points(b_up)
        b_up = b_up[mask_up]
        path_down = Polygon(calc_pentagon(G=1, invert=False), edgecolor='r', facecolor=(0,0,0,0)).get_path()
        mask_down = path_down.contains_points(b_down)
        b_down = b_down[mask_down]

    ax.plot(b_up[:,0], b_up[:,1], marker='o', color='b', ls='', ms=ms)
    if not exclude_down:
        ax.plot(b_down[:,0], b_down[:,1], marker='x', color='r', ls='', ms=ms)
    ax.set_aspect('equal')
    if axis_titles:
        ax.set_xlabel(r'$k_x$ / $|\mathbf{G}|$')
        ax.set_ylabel(r'$k_y$ / $|\mathbf{G}|$')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if plot_BZ:
        ax.add_patch(decagon(a=0.5))
    elif plot_QBZ:
        if R == 5:
            if invert != 'both':
                patch = Polygon(calc_pentagon(G=1, invert=invert), edgecolor='k', facecolor=(0,0,0,0))
                ax.add_patch(patch)
            else:
                patch1 = Polygon(calc_pentagon(G=1, invert=True), edgecolor='r', facecolor=(0,0,0,0))
                patch2 = Polygon(calc_pentagon(G=1, invert=False), edgecolor='b', facecolor=(0,0,0,0))
                ax.add_patch(patch1)
                ax.add_patch(patch2)
        elif R == 8:
            phi = np.pi * np.arange(0.5, 8.6, 1) / 4
            xy = np.column_stack((np.cos(phi), np.sin(phi))) * 0.5 / np.cos(np.pi/8)
            patch = Polygon(xy=xy, closed=True, edgecolor='k', facecolor=(0,0,0,0))
            ax.add_patch(patch)

    if plot_title:
        title_str = f'Orders = {orders}, ' + r'$k_{max}=$' + f'{cutoff}, Size = {b_up.shape[0]}x2'
        if G_min_in_title:
            G_mag = np.linalg.norm(b_up, axis=1) 
            G_min = np.min(G_mag[G_mag > 0.001])
            title_str += r', $G_{min}=$' + f'{G_min:.4g}'
        ax.set_title(title_str)
    plt.show()


def generate_map(basis, G_vects=G_vects, g_vects=g_vects):
    (b_up, b_down) = basis
    N_q = b_down.shape[0]
    N_basis = 2*N_q
    idx_map = np.full((N_basis,N_basis), np.nan)
    r = G_vects.shape[0]
    if g_vects is None:
        g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    for i in range(N_q):
        print(f'Evaluating basis state {i+1} out of {N_q}...' + 10*' ', end='\r')
        for j in range(i+1, N_q):
            # Down-to-down couplings
            dq = b_down[i] - b_down[j]
            idxs = np.where(np.isclose(-g_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j+N_q,i+N_q] = l
                idx_map[i+N_q,j+N_q] = l + r
            idxs = np.where(np.isclose(g_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j+N_q,i+N_q] = l + r
                idx_map[i+N_q,j+N_q] = l
            # Up-to-up couplings
            dq = b_up[i] - b_up[j]
            idxs = np.where(np.isclose(-g_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j,i] = l
                idx_map[i,j] = l + r
            idxs = np.where(np.isclose(g_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j,i] = l + r
                idx_map[i,j] = l
        # Spin-flip couplings
        for j in range(N_q):
            dq = b_down[i] - b_up[j]
            idxs = np.where(np.isclose(-G_vects, dq, atol=0.001).all(axis=1))[0]
            if idxs.size == 1:
                l = idxs[0]
                idx_map[j,i+N_q] = l
                idx_map[i+N_q,j] = l + r
    print('\nDone')
    return idx_map


def calc_H_from_map(H, idx_map, N_q, U, Uc, V, Vc):
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
                H[j+N_q,i+N_q] = -V_ext[l]
                H[i+N_q,j+N_q] = -np.conj(V_ext[l])
            # Up-to-up couplings
            l = idx_map[j,i]
            if not np.isnan(l):
                l = int(l)
                H[j,i] = V_ext[l]
                H[i,j] = np.conj(V_ext[l])
        # Spin-flip couplings
        for j in range(N_q):
            l = idx_map[j,i+N_q]
            if not np.isnan(l):
                l = int(l)
                H[j,i+N_q] = U_ext[l]
                H[i+N_q,j] = np.conj(U_ext[l])
    return H


def adjust_KE(H, q, basis):
    (b_up, b_down) = basis
    N_q = b_down.shape[0]
    for i in range(N_q):
        H[i,i] = np.sum((q - b_up[i,:])**2)
        H[i+N_q,i+N_q] = np.sum((q - b_down[i,:])**2)
    return H


def calc_H(q, basis, U0=0.02, N=5, R=8,
           phi_vals=None, G_vects=G_vects, g_vects=None, V0=0., V=None, 
           idx_map=None, W=0., zero_KE=False, **kwargs):
    (b_up, b_down) = basis
    N_q = b_down.shape[0]
    N_basis = 2*N_q
    if g_vects is None:
        g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    H = np.zeros((N_basis,N_basis), dtype=np.complex128)
    if phi_vals is None:
        phi_vals = calc_phi(N=N, R=R)
    U = -U0 * expi(-phi_vals)
    Uc = np.conjugate(U)
    if V is None:
        V = V0 * np.ones(R)
    Vc = np.conjugate(V)
    if idx_map is None:
        for i in tqdm(range(N_q), desc='Calculating Hamiltonian matrix elements'):
            # Kinetic energy
            if not zero_KE:
                H[i,i] = np.sum((q-b_up[i,:])**2)
                H[i+N_q,i+N_q] = np.sum((q-b_down[i,:])**2)
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
            # H_W couplings
            for j in range(N_q):
                dq = b_up[i] - b_up[j]
                idxs = np.where(np.isclose(-G_vects, dq, atol=0.001).all(axis=1))[0]
                if idxs.size == 1:
                    l = idxs[0]
                    H[j,i] += W * (-1)**l
                    H[i,j] += W * (-1)**l
                dq = b_down[i] - b_down[j]
                idxs = np.where(np.isclose(-G_vects, dq, atol=0.001).all(axis=1))[0]
                if idxs.size == 1:
                    l = idxs[0]
                    H[j+N_q,i+N_q] += W * (-1)**l
                    H[i+N_q,j+N_q] += W * (-1)**l
    else:
        for i in range(N_q):
            # Kinetic energy
            H[i,i] = np.sum((q-b_up[i,:])**2)
            H[i+N_q,i+N_q] = np.sum((q-b_down[i,:])**2)
        H = calc_H_from_map(H, idx_map, N_q, U, Uc, V, Vc)
    
    return H


# def calc_BS_point(q, return_evects=False, sparse=False, num_evals=20, **kwargs):
#     H = calc_H(q, **kwargs)
#     if sparse:
#         evals, evects = sp.sparse.linalg.eigsh(H, k=num_evals, which='SA')
#         sorted_indices = np.argsort(evals)
#         evals = evals[sorted_indices]
#         evects = evects[:, sorted_indices]
#     else:
#         evals, evects = np.linalg.eigh(H)
#     if return_evects:
#         return evals, evects
#     else:
#         return evals
    

# def calc_BS_line(q_vals, basis=calc_basis_states(), **kwargs):
#     if 'num_evals' in kwargs:
#         N = kwargs.get('num_evals')
#     else:
#         N_q_basis = basis[0].shape[0]
#         N = 2*N_q_basis
#     N_q = q_vals.shape[0]
#     E_vals = np.zeros((N,N_q))
#     for i in range(N_q):
#         print(f'Evaluating q value {i+1} of {N_q}... ' + 10*' ', end='\r')
#         E_vals[:,i] = calc_BS_point(q_vals[i,:], basis=basis, **kwargs)
#     print('\nDone')
#     return E_vals


# def calc_BS_surface(qx, qy, basis=calc_basis_states(), **kwargs):
#     nx = qx.size
#     ny = qy.size
#     if 'num_evals' in kwargs:
#         N = kwargs.get('num_evals')
#     else:
#         N_q_basis = basis[0].shape[0]
#         N = 2*N_q_basis
#     E_vals = np.zeros((N,nx,ny))
#     for i in range(nx):
#         for j in range(ny):
#             print(f'Evaluating q value {i*ny+j+1} out of {nx*ny}...' + 10*' ', 
#                   end='\r')
#             E_vals[:,i,j] = calc_BS_point(q=np.array([qx[i], qy[j]]), 
#                                           basis=basis, **kwargs)
#     print('\nDone')
#     return E_vals


def calc_BS_point(q=None, return_evects=False, num_evals=None, 
                  H=None, sparse=False, **kwargs):
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
    if num_evals is not None:
        evals = evals[:num_evals]
        evects = evects[:,:num_evals]
    if return_evects:
        return evals, evects
    else:
        return evals
    

def calc_BS_line(q_vals, basis, **kwargs):
    if 'num_evals' in kwargs:
        N = kwargs.get('num_evals')
    else:
        N_q_basis = basis[0].shape[0]
        N = 2*N_q_basis
    N_q = q_vals.shape[0]
    E_vals = np.zeros((N,N_q))
    print('Evaluating Hamiltonian... ', end='', flush=True)
    H = calc_H(q=q_vals[0,:], basis=basis, **kwargs)
    print('Done')
    for i in tqdm(range(N_q)):
        # print(f'Evaluating q value {i+1} of {N_q}... ' + 10*' ', end='\r')
        if i != 0:
            H = adjust_KE(H, q=q_vals[i,:], basis=basis)
        E_vals[:,i] = calc_BS_point(H=H, basis=basis, **kwargs)
    # print('\nDone')
    return E_vals


def calc_BS_surface(qx, qy, basis, return_evects=False, num_evals=None, **kwargs):
    nx = qx.size
    ny = qy.size
    N_basis = 2*basis[0].shape[0]
    if num_evals is not None:
        N_evals = num_evals
    else:
        N_evals = N_basis
    E_vals = np.zeros((N_evals,nx,ny))
    if return_evects:
        evects_arr = np.zeros((N_basis,N_evals,nx,ny), dtype=np.complex128)
    print('Evaluating Hamiltonian... ', end='', flush=True)
    H = calc_H(q=np.array([qx[0],qy[0]]), basis=basis, **kwargs)
    print('Done')
    print('Performing diagonalisation:')
    with tqdm(total=nx * ny, desc='Evaluating q values') as pbar:
        for i in range(nx):
            for j in range(ny):
                H = adjust_KE(H, q=np.array([qx[i], qy[j]]), basis=basis)
                if return_evects:
                    E_vals[:,i,j],evects_arr[:,:,i,j] = calc_BS_point(H=H, 
                                                                    basis=basis, 
                                                                    return_evects=return_evects, 
                                                                    num_evals=num_evals,
                                                                    **kwargs)
                else:
                    E_vals[:,i,j] = calc_BS_point(H=H, basis=basis, 
                                                return_evects=return_evects,
                                                num_evals=num_evals,
                                                **kwargs)
                pbar.update(1)
    print('Done')
    if return_evects:
        return E_vals, evects_arr
    else:
        return E_vals


def calc_DoS(filename=None, E_vals=None, dE=0.01, q_max=None, q_path=None,
             E_max=None):
    if filename is not None:
        data = np.load(filename)
        E_vals = data['E_vals']
    if q_max is not None:
        # print(q_max)
        qx_vals, qy_vals = data['qx_vals'], data['qy_vals']
        qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
        # print(qxx.shape)
        q2 = qxx**2 + qyy**2
        # print(q2)
        
        mask = q2 < q_max**2
        # print(mask)
        # print(np.count_nonzero(mask))
        # print(E_vals.shape)
        E_vals = E_vals[:, mask]
        # print(E_vals.shape)
    elif q_path is not None:
        qx_vals, qy_vals = data['qx_vals'], data['qy_vals']
        qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
        qxx, qyy = qxx.flatten(), qyy.flatten()
        mask = q_path.contains_points(np.column_stack((qxx, qyy)))
        # print(mask.shape)
        mask = mask.reshape(E_vals.shape[1:])
        # print(E_vals.shape)
        # print(E_vals.shape[1:])
        # print(mask.shape)
        E_vals = E_vals[:,mask]
    elif E_max is not None:
        E_vals = E_vals[E_vals <= E_max]
    E_vals = E_vals.flatten()

    # print(E_min, E_max)
    # print(E_bins[:5], E_bins[-5:])
    # print(E_bins.shape)

    # print(E_vals.shape)

    # E = 0
    # arr = ((E_vals >= (E - dE/2))
    #                        & (E_vals < (E + dE/2)))
    # print(arr)
    # print(np.sum(arr))
    E_min, E_max = np.min(E_vals), np.max(E_vals)
    # E_bins = np.arange(E_min-dE/2, E_max+3*dE/2, dE)
    # dos = np.array([np.sum((E_vals >= (E - dE/2))
    #                        & (E_vals < (E + dE/2)))/dE for E in E_bins])
    
    E_edges = np.arange(E_min, E_max + dE, dE)

    # Histogram: count number of states in each bin
    dos_counts, _ = np.histogram(E_vals, bins=E_edges)

    # Normalize to get DOS (per unit energy)
    dos = dos_counts / dE

    # Optional: compute bin centers for plotting
    E_bins = (E_edges[:-1] + E_edges[1:]) / 2
    # dos /= dE
    
    # print(dos.shape)
    return dos, E_bins


def calc_q_GMKG(q_M=None, q_K=None, G0=G0, R=8, N_k=100):
    if q_M is None:
        q_M = np.array([G0/2, 0.])
        q_K = np.array([G0/2, G0/2 * np.tan(np.pi/R)])
    q_vals_GM = np.column_stack((np.linspace(0, q_M[0], N_k),
                                 np.linspace(0, q_M[1], N_k)))
    q_vals_MK = np.column_stack((np.linspace(q_M[0], q_K[0], N_k)[1:],
                                 np.linspace(q_M[1], q_K[1], N_k)[1:]))
    q_vals_KG = np.column_stack((np.linspace(q_K[0], 0, N_k)[1:],
                                 np.linspace(q_K[1], 0, N_k)[1:]))
    q_vals = np.concatenate((q_vals_GM, q_vals_MK, q_vals_KG))
    return q_vals


def calc_q_GMKMKG(order=0, G0=G0):
    q_M1 = np.array([G0/2, 0.])
    c = np.cos(np.pi/5)
    s = np.sin(np.pi/5)
    R = np.array([[c,-s],
                    [s,c]])
    q_K1 = np.array([G0/2, G0/2 * np.tan(np.pi/10)])
    for i in range(order):
        q_M1 = R @ q_M1
        q_K1 = R @ q_K1

    q_M2 = R @ q_M1
    q_K2 = R @ q_K1
    q_vals_GM = np.column_stack((np.linspace(0, q_M1[0], 50),
                                 np.linspace(0, q_M1[1], 50)))
    q_vals_MK = np.column_stack((np.linspace(q_M1[0], q_K1[0], 50)[1:],
                                 np.linspace(q_M1[1], q_K1[1], 50)[1:]))
    q_vals_KM = np.column_stack((np.linspace(q_K1[0], q_M2[0], 50)[1:],
                                 np.linspace(q_K1[1], q_M2[1], 50)[1:]))
    q_vals_MK2 = np.column_stack((np.linspace(q_M2[0], q_K2[0], 50)[1:],
                                 np.linspace(q_M2[1], q_K2[1], 50)[1:]))
    q_vals_KG = np.column_stack((np.linspace(q_K2[0], 0, 50)[1:],
                                 np.linspace(q_K2[1], 0, 50)[1:]))
    q_vals = np.concatenate((q_vals_GM, q_vals_MK, q_vals_KM, q_vals_MK2,
                             q_vals_KG))
    return q_vals


def calc_distances(basis, idx):
    (b_up, b_down) = basis
    p = b_up[idx,:]
    print(f'Idx = {idx}, point p = {p}')
    norms = np.sum((b_up - p)**2, axis=1)
    norms_sorted = np.sort(norms)
    print('Minimum distances:')
    print(norms_sorted[:10])


def calc_pentagon(G=1, x=None, invert=False, rotate=None, r0=None):
    if x is None:
        x = G / (2*np.cos(np.pi/5))
    x1 = np.array([-x,0])
    if rotate is not None:
        t = rotate
        R = np.array([[np.cos(t), -np.sin(t)],
                      [np.sin(t), np.cos(t)]])
        x1 = R @ x1
    R = np.array([[np.cos(2*np.pi/5), -np.sin(2*np.pi/5)],
                  [np.sin(2*np.pi/5), np.cos(2*np.pi/5)]])
    points = np.array([x1])
    for i in range(4):
        points = np.row_stack((points, R@points[-1,:]))
    if invert:
        points *= -1
    if r0 is not None:
        points += r0
    return points


def calc_max_idx(E_vals, dos_vals, E_bins, E_min, E_max, dE=0.01):
    # Find position of largest gap in range E_min < E < E_max
    mask = np.logical_and(E_bins >= E_min, E_bins <= E_max)
    dos_vals = dos_vals[mask]
    E_bins = E_bins[mask]
    max_count = 0
    count = 0
    zero = False
    current_idx = None
    stored_idx = None
    for i in range(dos_vals.size):
        if dos_vals[i] < 1e-5:
            if zero:
                count += 1
            else:
                count += 1
                current_idx = i
            zero = True
        else:
            if zero:
                if count > max_count:
                    max_count = count
                    stored_idx = current_idx
            zero = False 
            count = 0

    E = E_bins[stored_idx] + dE

    # Find highest index of band with all eigenvalues < E
    mask = E_vals < E  # shape (N, M, K)
    all_less_than_E = np.all(mask, axis=(1, 2))  # shape (N,)
    if np.any(all_less_than_E):
        max_idx = np.where(all_less_than_E)[0].max()
    else:
        print('Error: no max index found')
    
    return max_idx


def select_physical_bands(filename=None, qx_vals=None, qy_vals=None, E_vals=None, basis=None, idx_up=0, 
                          idx_down=None, save=False, save_filename=None):
    if filename is not None:
        data = np.load(filename, allow_pickle=True)
        if 'q_vals' in data.files:
            q_vals = data['q_vals']
            qx_vals = q_vals[:,0]
            qy_vals = q_vals[:,1]
        else:
            qx_vals = data['qx_vals']
            qy_vals = data['qy_vals']
        E_vals = data['E_vals']
        basis = data['basis']
    (b_up, b_down) = basis
    selected_E_vals = np.zeros((2, *E_vals.shape[1:]))
    if idx_down is None:
        idx_down = b_up.shape[0]
    for i, qx in enumerate(qx_vals):
        for j, qy in enumerate(qy_vals):
            # calculate and order KE values
            q = np.array([qx, qy])
            K = np.concatenate((np.sum((b_up - q)**2, axis=1), np.sum((b_down - q)**2, axis=1)))
            idxs = np.argsort(K).argsort()
            # select E_vals with relevant indices
            selected_E_vals[0,i,j] = E_vals[idxs[idx_up],i,j]
            selected_E_vals[1,i,j] = E_vals[idxs[idx_down],i,j]
    if save:
        save_dict = {k:data[k] for k in data.files}
        if save_filename is None:
            save_filename = filename
        np.savez(save_filename, selected_E_vals=selected_E_vals, **save_dict)
    else:
        return selected_E_vals
    

def calc_spectral_weights(
    k_vals: np.ndarray,        # shape (N_k, 2)
    basis: tuple,              # (b_down, b_up)
    # recip_lattice: np.ndarray, # shape (2, 2)
    calc_IPR: bool = False,
    **kwargs,                  # passed through to calc_BS_point

) -> tuple[np.ndarray, np.ndarray]:
    """
    Backfold k_vals, diagonalise at each unique q, and extract spectral weights.

    Returns:
        evals_k:   shape (N_k, N_bands), eigenvalues for each k point
        weights_k: shape (N_k, N_bands), spectral weights summed over spin
    """
    b_up, b_down = basis
    N_basis = len(b_down)
    idx_down = np.where(np.isclose(b_down, np.zeros(2), atol=0.001).all(axis=1))[0][0]
    idx_up = np.where(np.isclose(b_up, np.zeros(2), atol=0.001).all(axis=1))[0][0]

    num_evals = kwargs.get('num_evals', None)
    if num_evals is None:
        num_evals = 2*N_basis

    q_vals = np.copy(k_vals)
    N_k = q_vals.shape[0]
    evals_k   = np.empty((N_k, num_evals))
    weights_up = np.empty((N_k, num_evals))
    weights_down = np.empty((N_k, num_evals))
    IPR_k = np.empty((N_k, num_evals))
    H = calc_H(q_vals[0,:], basis=basis, **kwargs)
    for i, q in enumerate(tqdm(q_vals)):
        H = adjust_KE(H, q, basis)
        evals, evects = calc_BS_point(q=q, return_evects=True, basis=basis, H=H, **kwargs)
        evals_k[i,:] = evals
        w_down = np.abs(evects[idx_down+N_basis, :])**2
        w_up   = np.abs(evects[idx_up, :])**2
        weights_up[i, :] = w_up
        weights_down[i, :] = w_down
        if calc_IPR:
            IPR_k[i,:] = np.sum(np.abs(evects)**4, axis=0)
    if calc_IPR:
        return evals_k, weights_up, weights_down, IPR_k
    else:
        return evals_k, weights_up, weights_down


def calc_spectral_function(
    w_vals: np.ndarray,    # shape (N_w,)
    sigma: float,
    evals_k: np.ndarray,   # shape (N_k, N_bands)
    weights_k: np.ndarray = None,
    weights_up: np.ndarray = None, # shape (N_k, N_bands)
    weights_down: np.ndarray = None, # shape (N_k, N_bands)
    normalise_gaussian: bool = False,
) -> np.ndarray:           # shape (N_k, N_w)
    """
    Compute A(k, w) from precomputed weights and eigenvalues.
    Can be called repeatedly with different w_vals/sigma without re-diagonalising.
    """
    # diffs shape: (N_k, N_bands, N_w)
    diffs = w_vals[np.newaxis, np.newaxis, :] - evals_k[:, :, np.newaxis]
    gaussians = np.exp(-0.5 * (diffs / sigma)**2)
    if normalise_gaussian:
        gaussians /= (sigma * np.sqrt(2 * np.pi))
    else:
        gaussians /= 2       # Account for spin so that max possible A(k,w) = 1

    if weights_up is not None and weights_down is not None:
        weights_k = weights_up + weights_down

    # weights_k shape (N_k, N_bands), gaussians shape (N_k, N_bands, N_w)
    return np.einsum('kb,kbw->kw', weights_k, gaussians)


def calc_spectral_IPR(
    evals_k: np.ndarray,   # shape (N_k, N_bands)
    weights_k: np.ndarray, # shape (N_k, N_bands)
    IPR_k: np.ndarray, # shape (N_k, N_bands)
    w_vals: np.ndarray,    # shape (N_w,)
    sigma: float,
    normalise_gaussian: bool = True,
) -> np.ndarray:           # shape (N_k, N_w)
    """
    Compute I(k, w) from precomputed weights and eigenvalues.
    Can be called repeatedly with different w_vals/sigma without re-diagonalising.
    """
    # diffs shape: (N_k, N_bands, N_w)
    diffs = w_vals[np.newaxis, np.newaxis, :] - evals_k[:, :, np.newaxis]
    gaussians = np.exp(-0.5 * (diffs / sigma)**2) 
    if normalise_gaussian:
        gaussians /= (sigma * np.sqrt(2 * np.pi))
    else:
        gaussians /= 2       # Account for spin so that max possible I(k,w) = 1

    # weights_k shape (N_k, N_bands), IPR_k shape (N_k, N_bands), gaussians shape (N_k, N_bands, N_w)
    return np.einsum('kb,kb,kbw->kw', weights_k, IPR_k, gaussians, optimize=True)


def calc_spectral_IPR_alt(
    w_vals: np.ndarray,    # shape (N_w,)
    sigma: float,
    evals_k: np.ndarray,   # shape (N_k, N_bands)
    weights_up: np.ndarray, # shape (N_k, N_bands)
    weights_down: np.ndarray, # shape (N_k, N_bands)
    normalise_gaussian: bool = True,
) -> np.ndarray:           # shape (N_k, N_w)
    """
    Compute alternative I(k, w) from precomputed weights and eigenvalues.
    Can be called repeatedly with different w_vals/sigma without re-diagonalising.
    """
    # diffs shape: (N_k, N_bands, N_w)
    diffs = w_vals[np.newaxis, np.newaxis, :] - evals_k[:, :, np.newaxis]
    gaussians = np.exp(-0.5 * (diffs / sigma)**2) 
    if normalise_gaussian:
        gaussians /= (sigma * np.sqrt(2 * np.pi))
    else:
        gaussians /= 2       # Account for spin so that max possible I(k,w) = 1

    # weights_k shape (N_k, N_bands), IPR_k shape (N_k, N_bands), gaussians shape (N_k, N_bands, N_w)
    weights_k = np.abs(weights_up)**2 + np.abs(weights_down)**2
    return np.einsum('kb,kbw->kw', weights_k, gaussians, optimize=True)


def calc_psi_r(x_vals, y_vals, psi_k, basis, k0):
    if len(psi_k.shape) == 1:
        psi_k = psi_k.reshape((psi_k.shape[0], 1))
    b_up, b_down = basis
    N_k = b_up.shape[0]
    k_up = k0 - b_up
    k_down = k0 - b_down
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    phase_up   = k_up[:, 0, None, None] * xx[None] + k_up[:, 1, None, None] * yy[None]
    phase_down = k_down[:, 0, None, None] * xx[None] + k_down[:, 1, None, None] * yy[None]
    psi_r_up   = np.sum(psi_k[:N_k, :, None, None] * np.exp(1j * phase_up[:, None, :, :]), axis=0)
    psi_r_down = np.sum(psi_k[N_k:, :, None, None] * np.exp(1j * phase_down[:, None, :, :]), axis=0)
    return psi_r_up, psi_r_down


def slice_array(arr, idxs):
    """Slice arr of shape (N_evals, Nx, Ny) down to (Nx, Ny) using index array idxs."""
    Nx, Ny = idxs.shape
    i, j = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    return arr[idxs, i, j]


def fit_ipr_powerlaw_vs_param(
    filenames,
    x_parameter,
    bin_width,
    output_filename,
    transfer_params=(),
    N_min=3,
    p0=None,
    bounds=([0, -np.inf], [np.inf, 0]),
):
    """
    For a set of .npz files, bin the IPR vs energy data into a common set
    of energy bins. Within each bin, each file contributes at most one point:
    the mean IPR (with std as uncertainty). A power law y = A * x^nu is then
    fitted across files for each bin that has at least N_min contributing files,
    weighted by the per-file IPR standard deviations.

    Results are saved to a new .npz file.

    Parameters
    ----------
    filenames : list of str
        Paths to .npz files, each containing IPR/energy data and x_parameter.
    x_parameter : str
        Key in each .npz file whose scalar value is used as the x coordinate
        for the power-law fit (e.g. "U0").
    bin_width : float
        Width of energy bins. Bins span the global energy range across all files.
    output_filename : str
        Path to the output .npz file.
    transfer_params : list or tuple of str, optional
        Keys to copy from the last loaded data file into the output .npz.
    N_min : int, optional
        Minimum number of files that must contribute a point to a bin for
        the fit to be performed (default 3).
    p0 : tuple or dict, optional
        Initial guesses for (A, nu). Dict may contain keys "A" and/or "nu".
        If None, A is auto-estimated and nu defaults to -1.
    bounds : 2-tuple of array-like, optional
        Bounds for (A, nu) passed to scipy.optimize.curve_fit.
        Default: A in [0, inf), nu in (-inf, 0].

    Returns
    -------
    dict with keys:
        bin_energies  : bin centres where fits were performed
        A_vals        : fitted prefactor A per bin
        A_errs        : 1-sigma uncertainty on A per bin
        nu_vals       : fitted exponent nu per bin
        nu_errs       : 1-sigma uncertainty on nu per bin
    """
    from scipy.optimize import curve_fit

    def power_law(x, A, nu):
        return A * np.abs(x) ** nu

    # ------------------------------------------------------------------
    # Load all files: extract x value and (E_all, IPR_all)
    # ------------------------------------------------------------------
    file_records = []   # list of dicts: x_val, E_all, IPR_all
    E_global_min =  np.inf
    E_global_max = -np.inf
    last_data    = None

    for filename in filenames:
        data = np.load(filename)
        last_data = data

        try:
            E_up   = data["E_up"].ravel()
            E_dn   = data["E_down"].ravel()
            IPR_up = data["IPR_up"].ravel()
            IPR_dn = data["IPR_down"].ravel()
        except KeyError:
            E_vals   = data["E_vals"]
            IPR_vals = data["IPR_vals"]
            idx_up   = data["k0_up_max_idx"]
            idx_dn   = data["k0_down_max_idx"]
            E_up   = slice_array(E_vals,   idx_up).ravel()
            E_dn   = slice_array(E_vals,   idx_dn).ravel()
            IPR_up = slice_array(IPR_vals, idx_up).ravel()
            IPR_dn = slice_array(IPR_vals, idx_dn).ravel()

        E_all   = np.concatenate((E_up, E_dn))
        IPR_all = np.concatenate((IPR_up, IPR_dn))

        try:
            x_val = data[x_parameter].item()
        except KeyError:
            if x_parameter == 'N_basis':
                x_val = data['basis'][0].shape[0] * 2
            else:
                raise KeyError(f"Parameter '{x_parameter}' not found in {filename}.")

        E_global_min = min(E_global_min, E_all.min())
        E_global_max = max(E_global_max, E_all.max())

        file_records.append(dict(x_val=x_val, E_all=E_all, IPR_all=IPR_all))

    # ------------------------------------------------------------------
    # Build common bin edges and centres
    # ------------------------------------------------------------------
    bin_edges   = np.arange(E_global_min, E_global_max + bin_width, bin_width)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins      = len(bin_centres)

    # ------------------------------------------------------------------
    # For each file, collapse each bin to (mean_IPR, std_IPR)
    # Shape of bin_means, bin_stds: (n_files, n_bins), NaN where no points
    # ------------------------------------------------------------------
    n_files   = len(file_records)
    bin_means = np.full((n_files, n_bins), np.nan)
    bin_stds  = np.full((n_files, n_bins), np.nan)
    x_vals    = np.array([r["x_val"] for r in file_records])

    for f_idx, record in enumerate(file_records):
        bin_indices = np.digitize(record["E_all"], bin_edges) - 1  # 0-based
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                bin_means[f_idx, b] = record["IPR_all"][mask].mean()
                bin_stds[f_idx, b]  = record["IPR_all"][mask].std()

    # ------------------------------------------------------------------
    # Auto p0 estimates
    # ------------------------------------------------------------------
    auto = {"A": 1.0, "nu": -1.0}
    if p0 is None:
        p0_base = (auto["A"], auto["nu"])
    elif isinstance(p0, dict):
        p0_base = (p0.get("A", auto["A"]), p0.get("nu", auto["nu"]))
    else:
        p0_base = tuple(p0)

    # ------------------------------------------------------------------
    # Per-bin power-law fit
    # ------------------------------------------------------------------
    result_energies = []
    result_A        = []
    result_A_err    = []
    result_nu       = []
    result_nu_err   = []

    for b in range(n_bins):
        # Which files have a valid (non-NaN) point in this bin?
        valid_mask = ~np.isnan(bin_means[:, b])
        n_valid    = valid_mask.sum()

        if n_valid < N_min:
            continue

        x_fit   = x_vals[valid_mask]
        y_fit   = bin_means[valid_mask, b]
        y_err   = bin_stds[valid_mask, b]

        # Replace any zero stds (single-point bins) with a small value
        # so curve_fit sigma doesn't break
        y_err   = np.where(y_err == 0, 1e-10, y_err)

        try:
            popt, pcov = curve_fit(
                power_law, x_fit, y_fit,
                p0=p0_base,
                sigma=y_err,
                absolute_sigma=True,
                bounds=bounds,
            )
        except RuntimeError:
            # Fit did not converge; skip this bin
            continue

        perr = np.sqrt(np.diag(pcov))

        result_energies.append(bin_centres[b])
        result_A.append(popt[0])
        result_A_err.append(perr[0])
        result_nu.append(popt[1])
        result_nu_err.append(perr[1])

    result_energies = np.array(result_energies)
    result_A        = np.array(result_A)
    result_A_err    = np.array(result_A_err)
    result_nu       = np.array(result_nu)
    result_nu_err   = np.array(result_nu_err)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    save_dict = {p: last_data[p] for p in transfer_params if p in last_data}
    save_dict.update({
        "bin_energies": result_energies,
        "A_vals":       result_A,
        "A_errs":       result_A_err,
        "nu_vals":      result_nu,
        "nu_errs":      result_nu_err,
        "bin_width":    bin_width,
    })
    np.savez(output_filename, **save_dict)

    print(f"Saved {len(result_energies)} fitted bins to '{output_filename}'.")

    return dict(
        bin_energies=result_energies,
        A_vals=result_A,
        A_errs=result_A_err,
        nu_vals=result_nu,
        nu_errs=result_nu_err,
    )
        

if __name__ == '__main__':
    U = 0.2
    left_str = 'Updated_Geometry/Data/BS_surface_R8_O'
    right_str = f'_c3.5_U{U:.4g}_V0_W0.05_N5_R8.npz'
    O_vals = [2, 3, 4, 5, 6]
    filenames = [left_str + str(O) + right_str for O in O_vals]
    f = f'Updated_Geometry/Data/Exponents_R8_U{U:.4g}_V0_W0.05_N5.npz'
    fit_ipr_powerlaw_vs_param(filenames, x_parameter='N_basis', bin_width=0.05, output_filename=f,
                              transfer_params=['U0', 'V0', 'W', 'N', 'R', 'cutoff'], N_min=3)

    # f = 'Updated_Geometry/Data/BS_Surface_b1_R8_U20.0_N5_V0.0_.npz'
    # select_physical_bands(filename=f, save=True)
    # R = 8
    # l = np.arange(R)
    # G_vects = np.column_stack((np.cos(2*np.pi*l/R),
    #                      np.sin(2*np.pi*l/R)))
    # g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    # O = 4
    # cutoff = None
    # f = f'Updated_Geometry/Data/Basis_O{O}.npz'
    # data = np.load(f)
    # basis = data['basis']
    # basis = calc_basis_states(basis=basis, orders=O, cutoff=cutoff, G_vects=G_vects, g_vects=g_vects, print_progress=False)
    # N_q = basis[0].shape[0]
    # q_K = np.array([0.5, 0.5 * np.tan(np.pi/8)])
    # qx_vals = np.linspace(-1., 1., 31)
    # qy_vals = np.copy(qx_vals)
    # qxx, qyy = np.meshgrid(qx_vals, qy_vals)
    # q_vals = np.column_stack((qxx.flatten(), qyy.flatten()))
    # dq = q_vals - q_K
    # # print(dq.shape)

    # idx_min = np.argmin(np.linalg.norm(q_vals-q_K, axis=1))
    # evals, evects = calc_BS_point(q_vals[idx_min], return_evects=True, U0=0.05, V0=0., G_vects=G_vects, g_vects=g_vects, basis=basis, N=5, R=R)
    # np.savez('Updated_Geometry/Data/NearKPoint_U0.05_V0_N5_R8.npz', evals=evals, evects=evects, U0=0.05, V0=0., N=5, R=8,
    #          orders=O, cutoff=cutoff, basis=basis)
    # data = np.load('Updated_Geometry/Data/NearKPoint_U0.05_V0_N5_R8.npz')
    # evals = data['evals']
    # evects = data['evects']
    # k0_up_mag = (np.abs(evects)**2)[0,:]
    # k0_down_mag = (np.abs(evects)**2)[N_q,:]
    # k0_up_idxs = np.argsort(k0_up_mag)[::-1]
    # k0_down_idxs = np.argsort(k0_down_mag)[::-1]
    # # evects_sorted_up = evects[:,k0_up_idxs]
    # # evects_sorted_down = evects[:,k0_down_idxs]
    # k0_up_mag_sorted = k0_up_mag[k0_up_idxs]
    # k0_down_mag_sorted = k0_down_mag[k0_down_idxs]
    # evals_up_sorted = evals[k0_up_idxs]
    # evals_down_sorted = evals[k0_down_idxs]
    # N = 10
    # print('k0, up:')
    # for i in range(N):
    #     print(f'{evals_up_sorted[i]:.3g}, {k0_up_mag_sorted[i]:.3g}')
    # print('k0, down:')
    # for i in range(N):
    #     print(f'{evals_down_sorted[i]:.3g}, {k0_down_mag_sorted[i]:.3g}')
    # print(f'evals = {evals_up_sorted[:N]}')
    # print(f'k0 magnitude = {k0_up_mag_sorted[:N]}')
    # print('k0, down:')
    # print(f'evals = {evals_down_sorted[:N]}')
    # print(f'k0 magnitude = {k0_down_mag_sorted[:N]}')
    
    # for O in orders:
    #     f = f'Updated_Geometry/Data/Basis_O{O}.npz'
    #     data = np.load(f)
    #     basis = data['basis']
    #     print(f'O = {O}, N = {basis[0].shape[0]} x 2')
        # print(f'Evaluating O = {O}...')
        # basis = calc_basis_states(basis=None, orders=O, cutoff=cutoff, G_vects=G_vects, g_vects=g_vects,
        #                         print_progress=True)
        # np.savez(f'Data/Basis_O{O}.npz', basis=basis, O=O, cutoff=cutoff)
    # G_mins = []
    # for O in orders:
    #     basis = calc_basis_states(basis=None, orders=O, cutoff=cutoff, G_vects=G_vects, g_vects=g_vects,
    #                             print_progress=True)
    #     (b_up, b_down) = basis
    #     G_mag = np.linalg.norm(b_up, axis=1) 
    #     G_min = np.min(G_mag[G_mag > 0.001])
    #     G_mins.append(G_min)
    # np.savez('Updated_Geometry/Data/G_min.npz', orders=orders, G_mins=G_mins)

    # data = np.load('Updated_Geometry/Data/G_min.npz')
    # orders = data['orders']
    # G_mins = data['G_mins']
    # fig, ax = plt.subplots()
    # ax.plot(orders, G_mins, marker='x', ls='-', color='b')
    # ax.set_xlabel(r'$O$')
    # ax.set_ylabel(r'$G_{min}$')
    # ax.set_title('Minimum Basis Vector vs Order')
    # plt.show()

    # f = f'Updated_Geometry/Data/Basis_o{orders}.npz'
    # data = np.load(f)
    # b_full = data['basis']
    # basis = calc_basis_states(orders=orders, cutoff=cutoff, basis=None,
    #                               G_vects=G, g_vects=g_vects)
    # basis = calc_basis_states(orders=orders, cutoff=cutoff, basis=b_full,
    #                             G_vects=G_vects, g_vects=g_vects)
    # G_mag = np.linalg.norm(b_up, axis=1)
    # G_min = np.min(G_mag[G_mag>0.001])
    # print(f'G_min = {G_min:.4g}')
    # print(basis[1][:10,:])
    # idx_down = np.where(np.isclose(b_down, np.zeros(2), atol=0.001).all(axis=1))[0]
    # idx_up = np.where(np.isclose(b_up, np.zeros(2), atol=0.001).all(axis=1))[0]
    # print(idx_down)
    # print(idx_up)
    # print(b_down[idx_down,:])
    # print(b_up[idx_up,:])
    # plot_basis_states(basis, orders=orders, cutoff=cutoff, R=R, plot_QBZ=False, plot_BZ=False, ms=2,
    #                   G_min_in_title=True)

    # b = {0:np.zeros(2), 3:G[0,:], 6:G[0,:]+G[3,:], 1:G[0,:]+G[3,:]+G[6,:], 
    #      4:G[0,:]+G[3,:]+G[6,:]+G[1,:], 7:G[3,:]+G[6,:]+G[1,:], 2:G[6,:]+G[1,:], 
    #      5:G[1,:]}
    # b_down = np.array(list(b.values()))
    # basis = (b_down, np.copy(b_down))
    # # plot_basis_states(basis, plot_BZ=False, plot_QBZ=True, R=8)
    # U0 = 0.03
    # N = 5
    # V0 = 0
    # q_vals = calc_q_GMKG(G0=1)
    # q_K = np.array([0.5, 0.5 * np.tan(np.pi/8)])
    # dq = np.array([np.linalg.norm(q_K - b) for b in b_down])
    # print(dq)
    # evals = calc_BS_line(q_vals, basis=basis, U0=U0, V0=V0, N=N, R=R, G_vects=G)
    # print(evals.shape)
    # np.savez('Updated_Geometry/Data/SpectrumTest.npz', evals=evals, q_vals=q_vals)
    # np.savez('Updated Geometry/Data/Basis_o4.npz', basis=basis, orders=orders, cutoff=None)
    
    # f = 'Updated Geometry/Data/Basis_o6.npz'
    # data = np.load(f)
    # b_full = data['basis']
    # l = np.arange(8)
    # G_vects = np.column_stack((np.cos(np.pi*l/4), np.sin(np.pi*l/4)))
    # g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    # orders = 1
    # cutoff = None
    # basis = calc_basis_states(basis=None, orders=orders, cutoff=cutoff, G_vects=G_vects, g_vects=g_vects)
    # # basis = calc_basis_states(basis=None, orders=orders, cutoff=cutoff)
    # # basis = calc_basis_states_alt(basis=None, orders=orders, cutoff=cutoff)

    # # print(basis[0].shape[0])
    # # # print(np.sort(np.linalg.norm(basis[0], axis=1)))
    # plot_basis_states(basis=basis, ms=3, orders=orders, cutoff=cutoff, plot_BZ=False,
    #                   plot_QBZ=True, invert='both', inside_QBZ=False, R=8, exclude_down=True)
    # points = calc_pentagon()
    # print(points)

    # f = 'Updated Geometry/Data/RQBZData_o5_c3.5_U0.2_N3_V0.0_altbasis.npz'
    # data = np.load(f)
    # E_vals = data['E_vals']
    # E_bins = data['E_bins']
    # dos_vals = data['dos_vals']
    # max_idx = calc_max_idx(E_vals=E_vals, dos_vals=dos_vals, E_bins=E_bins, 
    #                        E_min=-0.0, E_max=0.4)
    # print(max_idx)
