import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import scipy as sp
from tqdm import tqdm


G0 = 1
G_vects = G0 * np.column_stack((np.cos(np.arange(5)*2*np.pi/5),
                                np.sin(np.arange(5)*2*np.pi/5)))

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


def calc_basis_states(orders=2, G_vects=G_vects, g_vects=g_vects, basis=None, cutoff=1.6):
    if basis is None:
        prev_down = np.array([np.zeros(2)])
        prev_up = np.array([np.zeros(2)])
        b_down = np.array([np.zeros(2)])
        b_up = np.array([np.zeros(2)])

        for i in range(orders):
            print(f'Evaluating order {i+1} out of {orders}')
            # new_down = np.array([[]])
            # new_up = np.array([[]])
            new_down = np.array([prev_up[0,:]-G_vects[0,:]])
            new_up = np.array([prev_down[0,:]+G_vects[0,:]])
            # skip = True
            for j,point in enumerate(prev_down):
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
            print('')
            # skip = True
            for j,point in enumerate(prev_up):
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
            print('')
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
                      cutoff=None, inside_QBZ=False, R=5, exclude_down=False):
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
        ax.set_title(f'Orders = {orders}, Cutoff = {cutoff}, Size = {b_up.shape[0]}x2')
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
           idx_map=None, W=0., **kwargs):
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
        for i in range(N_q):
            # Kinetic energy
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
                  H=None, **kwargs):
    if H is None:
        H = calc_H(q, **kwargs)
    if num_evals is not None:
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
    

def calc_BS_line(q_vals, basis, **kwargs):
    if 'num_evals' in kwargs:
        N = kwargs.get('num_evals')
    else:
        N_q_basis = basis[0].shape[0]
        N = 2*N_q_basis
    N_q = q_vals.shape[0]
    E_vals = np.zeros((N,N_q))
    print('Evaluating Hamiltonian...')
    H = calc_H(q=q_vals[0,:], basis=basis, **kwargs)
    print('Done')
    for i in range(N_q):
        print(f'Evaluating q value {i+1} of {N_q}... ' + 10*' ', end='\r')
        if i != 0:
            H = adjust_KE(H, q=q_vals[i,:], basis=basis)
        E_vals[:,i] = calc_BS_point(H=H, basis=basis, **kwargs)
    print('\nDone')
    return E_vals


def calc_BS_surface(qx, qy, basis, return_evects=False, num_evals=None, **kwargs):
    nx = qx.size
    ny = qy.size
    N_basis = 2*basis[0].shape[0]
    if num_evals is not None:
        N_evals = num_evals
    else:
        N_evals = N_basis
        # print(N_evals)
    E_vals = np.zeros((N_evals,nx,ny))
    if return_evects:
        evects_arr = np.zeros((N_basis,N_evals,nx,ny), dtype=np.complex128)
    print('Evaluating Hamiltonian...')
    H = calc_H(q=np.array([qx[0],qy[0]]), basis=basis, **kwargs)
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
                                                                  num_evals=num_evals,
                                                                  **kwargs)
            else:
                E_vals[:,i,j] = calc_BS_point(H=H, basis=basis, 
                                              return_evects=return_evects,
                                              num_evals=num_evals,
                                              **kwargs)
    print(f'Evaluating q value {i*ny+j+1} out of {nx*ny}... Done')
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


def calc_q_GMKG(q_M=None, q_K=None, G0=G0, R=8):
    if q_M is None:
        q_M = np.array([G0/2, 0.])
        q_K = np.array([G0/2, G0/2 * np.tan(np.pi/R)])
    q_vals_GM = np.column_stack((np.linspace(0, q_M[0], 100),
                                 np.linspace(0, q_M[1], 100)))
    q_vals_MK = np.column_stack((np.linspace(q_M[0], q_K[0], 100)[1:],
                                 np.linspace(q_M[1], q_K[1], 100)[1:]))
    q_vals_KG = np.column_stack((np.linspace(q_K[0], 0, 100)[1:],
                                 np.linspace(q_K[1], 0, 100)[1:]))
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
    # print(idx_down, idx_up)

    # if 'num_bands' in kwargs:
    #     num_bands = kwargs['num_bands']
    # else:
    #     num_bands = N_basis
    num_evals = kwargs.get('num_evals', 2*N_basis)

    # q_vals, i_q_arr, i_G_arr = backfold_k_points(k_vals, basis, recip_lattice)
    # print(q_vals.shape)

    # Diagonalise at each unique q point
    q_vals = np.copy(k_vals)
    # all_evals, all_evects = [], []
    N_k = q_vals.shape[0]
    evals_k   = np.empty((N_k, num_evals))
    weights_k = np.empty((N_k, num_evals))
    H = calc_H(q_vals[0,:], basis=basis, **kwargs)
    for i,q in enumerate(tqdm(q_vals)):
        # print(q)
        H = adjust_KE(H, q, basis)
        evals, evects = calc_BS_point(q=q, return_evects=True, basis=basis, H=H, **kwargs)
        # all_evals.append(evals)
        evals_k[i,:] = evals
        w_down = np.abs(evects[idx_down+N_basis, :])**2
        w_up   = np.abs(evects[idx_up, :])**2
        weights_k[i, :] = w_down + w_up
        # all_evects.append(evects)

    # N_k = len(k_vals)
    # N_bands = len(all_evals[0])

    # for i_k in range(N_k):
    #     # i_q = i_q_arr[i_k]
    #     # i_G = i_G_arr[i_k]

    #     evects = all_evects[i_k]                         # shape (2*N_basis, N_bands)
    #     w_down = np.abs(evects[0, :])**2
    #     w_up   = np.abs(evects[N_basis, :])**2

    #     evals_k[i_k]   = all_evals[i_k]
    #     weights_k[i_k] = w_down + w_up

    return evals_k, weights_k


def calc_spectral_function(
    evals_k: np.ndarray,   # shape (N_k, N_bands)
    weights_k: np.ndarray, # shape (N_k, N_bands)
    w_vals: np.ndarray,    # shape (N_w,)
    sigma: float,
) -> np.ndarray:           # shape (N_k, N_w)
    """
    Compute A(k, w) from precomputed weights and eigenvalues.
    Can be called repeatedly with different w_vals/sigma without re-diagonalising.
    """
    # diffs shape: (N_k, N_bands, N_w)
    diffs = w_vals[np.newaxis, np.newaxis, :] - evals_k[:, :, np.newaxis]
    gaussians = np.exp(-0.5 * (diffs / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    # weights_k shape (N_k, N_bands), gaussians shape (N_k, N_bands, N_w)
    return np.einsum('kb,kbw->kw', weights_k, gaussians)


if __name__ == '__main__':

    # f = 'Updated_Geometry/Data/BS_Surface_b1_R8_U20.0_N5_V0.0_.npz'
    # select_physical_bands(filename=f, save=True)
    R = 8
    l = np.arange(R)
    G_vects = np.column_stack((np.cos(2*np.pi*l/R),
                         np.sin(2*np.pi*l/R)))
    g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    orders = 5
    cutoff = 2.5
    basis = calc_basis_states(basis=None, orders=orders, cutoff=cutoff, G_vects=G_vects, g_vects=g_vects)
    # f = f'Updated_Geometry/Data/Basis_o{orders}.npz'
    # data = np.load(f)
    # b_full = data['basis']
    # basis = calc_basis_states(orders=orders, cutoff=cutoff, basis=None,
    #                               G_vects=G, g_vects=g_vects)
    # basis = calc_basis_states(orders=orders, cutoff=cutoff, basis=b_full,
    #                             G_vects=G_vects, g_vects=g_vects)
    (b_up, b_down) = basis
    # print(basis[1][:10,:])
    # idx_down = np.where(np.isclose(b_down, np.zeros(2), atol=0.001).all(axis=1))[0]
    # idx_up = np.where(np.isclose(b_up, np.zeros(2), atol=0.001).all(axis=1))[0]
    # print(idx_down)
    # print(idx_up)
    # print(b_down[idx_down,:])
    # print(b_up[idx_up,:])
    # plot_basis_states(basis, orders=orders, cutoff=cutoff, R=R, plot_QBZ=True, plot_BZ=False)

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
