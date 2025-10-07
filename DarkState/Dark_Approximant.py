import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Approximant'))
import Approximant_Vectors as AV
import Approximant_Bandstructure as AB


def calc_g_vects(G_vects):
    N = G_vects.shape[0]
    g_vects = np.zeros((N, N, 2))
    for i in range(N):
        for j in range(N):
            g_vects[i,j] = G_vects[i,:] - G_vects[j,:]
    return g_vects


def calc_phi(R=5, p1=1., p2=0., phi1=0, phi2=0):
    l = np.arange(R)
    phi_up = 2*np.pi*p1*l/R + phi1
    phi_down = 2*np.pi*p2*l/R + phi2
    return phi_up, phi_down


def calc_V1(G_vects, phi_up, phi_down, V1=1.):
    N = G_vects.shape[0]
    g_uu = np.zeros((N, N, 2))
    g_ud = np.zeros((N, N, 2))
    g_du = np.zeros((N, N, 2))
    g_dd = np.zeros((N, N, 2))
    V_uu = np.zeros((N, N), dtype=np.complex128)
    V_ud = np.zeros((N, N), dtype=np.complex128)
    V_du = np.zeros((N, N), dtype=np.complex128)
    V_dd = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        Gi = G_vects[i,:]
        for j in range(N):
            Gj = G_vects[j,:]
            g_uu[i,j,:] = Gj - Gi
            g_ud[i,j,:] = -Gj - Gi
            g_du[i,j,:] = Gj + Gi
            g_dd[i,j,:] = Gi - Gj
            V_uu[i,j] = np.exp(1j*(phi_up[i]-phi_up[j]))
            V_ud[i,j] = np.exp(1j*(phi_up[i]-phi_down[j]))
            V_du[i,j] = np.exp(1j*(phi_down[i]-phi_up[j]))
            V_dd[i,j] = np.exp(1j*(phi_down[i]-phi_down[j]))
    V_uu *= V1
    V_ud *= V1
    V_du *= V1
    V_dd *= V1
    g_uu = g_uu.reshape(-1,2)
    g_ud = g_ud.reshape(-1,2)
    g_du = g_du.reshape(-1,2)
    g_dd = g_dd.reshape(-1,2)
    V_uu = V_uu.flatten()
    V_ud = V_ud.flatten()
    V_du = V_du.flatten()
    V_dd = V_dd.flatten()
    return g_uu, g_ud, g_du, g_dd, V_uu, V_ud, V_du, V_dd


def calc_V_const(g, V):
    mask = np.all(np.isclose(g, 0, 0.001), axis=1)
    # print(mask)
    g_filtered = g[~mask]
    V_filtered = V[~mask]
    V_const = np.sum(V[mask])
    return g_filtered, V_filtered, V_const


def calc_V0(G_vects, V0=1.):
    N = G_vects.shape[0]
    g = np.zeros((1,2))
    V = np.zeros(1)
    for i in range(N):
        for j in range(i+1,N):
            g = np.vstack((g, G_vects[i,:] - G_vects[j,:]))
            g = np.vstack((g, G_vects[j,:] - G_vects[i,:]))
            V = np.append(V, [V0, V0])
    return g[1:,:], V[1:]


def calc_H(q, basis=AB.calc_square_basis_states(a=3, cutoff=2.5), V1=1, V0=0., 
           G_vects=AV.square_approximant(a=3), phi_up=None, phi_down=None,
           R=5, p1=1, p2=0, phi1=0., phi2=0., return_V_const=True, 
           **kwargs):
    (b_up, b_down) = basis
    N_q = b_down.shape[0]
    N_basis = 2*N_q
    H = np.zeros((N_basis,N_basis), dtype=np.complex128)
    if phi_up is None or phi_down is None:
        phi_up, phi_down = calc_phi(R=R, p1=p1, p2=p2, phi1=phi1, phi2=phi2)
    g_uu, g_ud, g_du, g_dd, V_uu, V_ud, V_du, V_dd = calc_V1(G_vects, phi_up, phi_down, V1)
    # print(g_uu)
    g_uu, V_uu, V_const_up = calc_V_const(g_uu, V_uu)
    g_dd, V_dd, V_const_down = calc_V_const(g_dd, V_dd)
    g_scalar, V_scalar = calc_V0(G_vects, V0)
    for i in range(N_q):
        # Kinetic energy
        H[i,i] = np.sum((q-b_up[i,:])**2) + V_const_up
        H[i+N_q,i+N_q] = np.sum((q-b_down[i,:])**2) + V_const_down
    # if idx_map is None:
        # Same-spin couplings
        for j in range(i+1, N_q):
            # Up-to-up couplings
            dq = b_up[i] - b_up[j]
            idxs = np.where(np.isclose(g_uu, dq, atol=0.001).all(axis=1))[0]
            if idxs.size > 1:
                # print(f'Warning: duplicate at g_uu = {g_uu[idxs[0]]}')
                pass
            for l in idxs:
                H[j,i] += V_uu[l]
                H[i,j] += np.conj(V_uu[l])
            # Down-to-down couplings
            dq = b_down[i] - b_down[j]
            idxs = np.where(np.isclose(g_dd, dq, atol=0.001).all(axis=1))[0]
            if idxs.size > 1:
                # print(f'Warning: duplicate at g_dd = {g_dd[idxs[0]]}')
                pass
            for l in idxs:
                H[j+N_q,i+N_q] += V_dd[l]
                H[i+N_q,j+N_q] += np.conj(V_dd[l])
            # idxs = np.where(np.isclose(g_vects, dq, atol=0.001).all(axis=1))[0]
            # if idxs.size == 1:
            #     l = idxs[0]
            #     H[j+N_q,i+N_q] += -Vc[l]
            #     H[i+N_q,j+N_q] += -V[l]
            
        # Spin-flip couplings
        for j in range(N_q):
            dq = b_down[i] - b_up[j]
            idxs = np.where(np.isclose(g_ud, dq, atol=0.001).all(axis=1))[0]
            if idxs.size > 1:
                # print(f'Warning: duplicate at g_ud = {g_ud[idxs[0]]}')
                pass
            for l in idxs:
                H[j,i+N_q] += V_ud[l]
                H[i+N_q,j] += np.conj(V_ud[l])
        # Scalar potential
        for j in range(N_q):
            dq = b_down[i] - b_up[j]
            idxs = np.where(np.isclose(g_scalar, dq, atol=0.001).all(axis=1))[0]
            if idxs.size > 1:
                # print(f'Warning: duplicate at g_ud = {g_ud[idxs[0]]}')
                pass
            for l in idxs:
                H[j,i] += V_scalar[l]
                H[j+N_q,i+N_q] += V_scalar[l]
    if return_V_const:
        return H, V_const_up, V_const_down
    else:
        return H


def adjust_KE(H, q, basis=AB.calc_square_basis_states(a=3, cutoff=2.5), V_const_up=0., V_const_down=0.):
    (b_down, b_up) = basis
    N_q = b_down.shape[0]
    for i in range(N_q):
        H[i,i] = np.sum((q - b_up[i,:])**2) + V_const_up
        H[i+N_q,i+N_q] = np.sum((q - b_down[i,:])**2) + V_const_down
    return H


def calc_BS_surface(qx, qy, basis=AB.calc_square_basis_states(a=3, cutoff=2.5),
                    return_evects=False, **kwargs):
    nx = qx.size
    ny = qy.size
    N_basis = 2*basis[0].shape[0]
    if 'num_evals' in kwargs and 'sparse' in kwargs and kwargs.get('sparse'):
        N_evals = kwargs.get('num_evals')
    else:
        N_evals = N_basis
    E_vals = np.zeros((N_evals,nx,ny))
    if return_evects:
        evects_arr = np.zeros((N_basis,N_evals,nx,ny), dtype=np.complex128)
    print('Evaluating Hamiltonian...')
    H, V_const_up, V_const_down = calc_H(q=np.array([qx[0],qy[0]]), basis=basis, return_V_const=True, **kwargs)
    print('Done')
    for i in range(nx):
        for j in range(ny):
            print(f'Evaluating q value {i*ny+j+1} out of {nx*ny}...' + 10*' ', 
                  end='\r')
            H = adjust_KE(H, q=np.array([qx[i], qy[j]]), basis=basis, V_const_up=V_const_up, V_const_down=V_const_down)
            if return_evects:
                E_vals[:,i,j],evects_arr[:,:,i,j] = AB.calc_BS_point(H=H, 
                                                                     basis=basis, 
                                                                     return_evects=return_evects, 
                                                                     **kwargs)
            else:
                E_vals[:,i,j] = AB.calc_BS_point(H=H, basis=basis, 
                                                 return_evects=return_evects,
                                                 **kwargs)
    print('\nDone')
    return E_vals, evects_arr


def find_duplicate_rows(g):
    # unique_rows are the distinct rows, return_index maps them back,
    # return_counts tells how many times each unique row appears
    unique_rows, indices, counts = np.unique(g, axis=0, return_index=True, return_counts=True)

    # Duplicate rows are those with counts > 1
    dup_mask = counts > 1
    duplicates = unique_rows[dup_mask]

    if duplicates.size > 0:
        return duplicates
    else:
        print("No duplicate rows found.")
        return None
    

def calc_tri_basis_states(cutoff=2.5):
    a1 = np.array([1,0])
    a2 = np.array([0.5, 0.5*np.sqrt(3)])
    L = np.ceil(2*cutoff)
    xs = np.arange(-L,L+1)
    ys = np.copy(xs)
    basis = np.array([[0, 0]])
    for x in xs:
        for y in ys:
            basis = np.vstack((basis, x*a1+y*a2))
    basis = basis[np.linalg.norm(basis, axis=1) <= cutoff]
    return (basis[1:,:], basis[1:,:])


def q_inside_hex(N_q=31, a=0.5, f=1.001):
    r = f * a / np.cos(np.pi/6)
    # print(r)
    # hexagon = mpl.patches.RegularPolygon(xy=(0,0), numVertices=6, radius=r, orientation=0, fc=(0,0,0,0), ec='k')
    a1 = np.array([1, 0])
    a2 = np.array([0.5, 0.5*np.sqrt(3)])
    xs = np.arange(-2*a, 2*a+f-1, 2*a/(N_q-1))
    # xs = np.linspace(-3*a, 3*a, 3*N_q-2)
    # xs = np.linspace(-a, a, N_q)
    # print(xs)
    ys = np.copy(xs)
    r = np.array([[0,0]])
    for x in xs:
        for y in ys:
            r = np.vstack((r, x*a1 + y*a2))
    r = r[1:,:]
    xy = np.array([0.0, 0.0])
    R = f * a / np.cos(np.pi/6)
    angles = np.linspace(0, 2*np.pi, 7) + np.pi/6  # orientation=0

    verts = xy + R * np.column_stack((np.cos(angles), np.sin(angles)))
    hex_path = mpl.path.Path(verts, closed=True)

    mask = hex_path.contains_points(r)
    return r[mask], hex_path

def hex_BZ_patch(a=0.5):
    n = np.arange(6)
    t = np.pi*n/3 + np.pi/6
    r = np.column_stack((np.cos(t), np.sin(t))) * a / np.cos(np.pi/6)
    hexagon = mpl.patches.Polygon(r, fc=(0,0,0,0), ec='k')
    return hexagon


def approximant_errors(a_vals=np.arange(1,10), R=5):
    means = np.zeros(a_vals.size)
    l = np.arange(R)
    G_exact = np.column_stack((np.cos(2*np.pi*l/R), np.sin(2*np.pi*l/R)))
    g_uu, g_ud, g_du, g_dd, V_uu, V_ud, V_du, V_dd = calc_V1(G_exact, phi_up=np.zeros(R), phi_down=np.zeros(R), V1=1.)
    for i, a in enumerate(a_vals):
        G_approx = AV.square_approximant(a=a, G_vects=G_exact)
        g_uu_a, g_ud_a, g_du_a, g_dd_a, V_uu, V_ud, V_du, V_dd = calc_V1(G_approx, phi_up=np.zeros(R), phi_down=np.zeros(R), V1=1.)
        err_uu = np.linalg.norm(g_uu - g_uu_a, axis=1)**2
        err_ud = np.linalg.norm(g_ud - g_ud_a, axis=1)**2
        means[i] = np.sqrt(np.mean(np.concatenate((err_uu, err_ud))))
    fig, ax = plt.subplots()
    ax.plot(a_vals, means, marker='x', ls='-', color='b')
    ax.set_xlabel(r'$a$')
    ax.set_ylabel(r'$\epsilon_{RMS}$')
    ax.set_title('Approximant Errors')
    ax.set_ylim(bottom=0)
    plt.show()


if __name__ == '__main__':
    approximant_errors(a_vals=np.arange(3,11), R=5)
    # fig,ax = plt.subplots()
    # q, hex_path = q_inside_hex(N_q=20, a=0.5)
    # q_all = calc_tri_basis_states(c=10)
    # ax.plot(q_all[:,0], q_all[:,1], marker='o', ls='', ms=5, color='k')
    # ax.plot(q[:,0], q[:,1], marker='o', ls='', ms=5, color='r')
    # # hexagon = mpl.patches.PathPatch(hex_path, fc=(0,0,0,0), ec='k')
    # hexagon = hex_BZ_patch()
    # ax.add_patch(hexagon)
    # lim = 2.5
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.set_aspect('equal')
    # plt.show()
    # R = 3
    # l = np.arange(3)
    # a = 3
    # G_exact = np.column_stack((np.cos(2*np.pi*l/R), np.sin(2*np.pi*l/R)))
    # G_approx = AV.square_approximant(a=a, G_vects=G_exact)
    # basis = calc_tri_basis_states(c=3.5)
    # fig,ax = plt.subplots()
    # ax.set_aspect('equal')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # AV.add_square_grid(ax, a=a, L=10, color='k')
    # ax.plot(basis[:,0], basis[:,1], marker='o', ls='', color='k', ms=5)
    # # AV.plot_BZ(ax, a=a)
    # AV.plot_vectors(G_exact, ax, color='k')
    # # AV.plot_vectors(G_approx, ax, color='r')
    # # g_vects = calc_g_vects(G_approx)
    # g_uu, g_ud, g_du, g_dd, V_uu, V_ud, V_du, V_dd = calc_V1(G_exact, phi_up=np.zeros(5), phi_down=np.zeros(5), V1=1.)
    # g_scalar, V_scalar = calc_V0(G_exact, V0=1)
    # # print(g_uu)
    # # print(g_uu.shape)
    # # duplicates = find_duplicate_rows(g_uu)
    # # print(duplicates)
    # # print(g_du)
    # # print(g_du.shape)
    # # duplicates = find_duplicate_rows(g_du)
    # # print(duplicates)
    # AV.plot_vectors(g_scalar, ax, color='r')
    # AV.plot_vectors(g_uu.reshape((-1,2)), ax, color='limegreen')
    # AV.plot_vectors(g_du.reshape((-1,2)), ax, color='cyan')
    # lim = 5
    # ax.set_xlim(-lim,lim)
    # ax.set_ylim(-lim,lim)
    # ax.set_title(f'a = {a}')
    # plt.show()


