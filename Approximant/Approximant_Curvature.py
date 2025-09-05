import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import Decagon
import matplotlib.colors as mcolors
# import os
# import sys
# print(os.getcwd())
# subdir = os.path.abspath('Appoximant')
# sys.path.append('Approximant')

import Approximant_Bandstructure as ABS
from matplotlib.patches import Polygon
import scipy as sp
# import Plot_Berry_Curvature as PBC

# Calculate reciprocal lattice (G) vectors 
G0 = 1
G_vects = G0 * np.column_stack((np.cos(np.arange(10)*np.pi/5),
                                np.sin(np.arange(10)*np.pi/5)))


def calc_v(q, basis):
    N_q = basis[0].shape[0]
    N = 2*N_q
    vx = np.zeros((N,N))
    vy = np.zeros((N,N))
    for i in range(N_q):
        # Spin-up basis states
        v = 2*(q-basis[0][i])
        vx[i,i] = v[0]
        vy[i,i] = v[1]
        # Spin-down basis states
        v = 2*(q-basis[1][i])
        vx[i+N_q,i+N_q] = v[0]
        vy[i+N_q,i+N_q] = v[1]
    return vx, vy


def calc_curv_point(n=0, n_occ=None, evals=None, evects=None, q=None, vx=None, 
                    vy=None, basis=None, method='TKNN_all', a=3, cutoff=2.5,
                    tol=1e-10, n0=None, n1=None, n12=None, n2=None, **kwargs):
    if method[:4] == 'TKNN':
        if evals is None:
            if q is None:
                print('Error: must specify either bandstructure or point in k space'
                    +' at which to calculate it.')
            else:
                if basis is None:
                    basis = ABS.calc_square_basis_states(a=a, cutoff=cutoff)
                H = ABS.calc_H(q=q, basis=basis, **kwargs)
                evals, evects = np.linalg.eigh(H)
        if vx is None or vy is None:
            if basis is None:
                basis = ABS.calc_square_basis_states(a=a, cutoff=cutoff)
            vx, vy = calc_v(q=q, basis=basis)
        u_n = evects[:,n]
        E_n = evals[n]
        N = evals.size
        curv = 0
        for i in range(N):
            if method == 'TKNN_unocc':
                if n_occ is None:
                    print('Error: must specified occupied bands. Defaulting to <all> method')
                    method = 'TKNN_all'
                if i not in n_occ:
                    u_i = evects[:,i]
                    E_i = evals[i]
                    if np.abs(E_n - E_i) < tol:
                        print('Warning: skipping near-degenerate point')
                        pass
                    else:
                        curv += (1j/(E_n - E_i)**2)*(((np.conjugate(u_n) @ vx @ u_i) 
                                                    * (np.conjugate(u_i) @ vy @ u_n))
                                                    - ((np.conjugate(u_n) @ vy @ u_i) 
                                                    * (np.conjugate(u_i) @ vx @ u_n)))
            if method == 'TKNN_all':
                if i != n:
                    u_i = evects[:,i]
                    E_i = evals[i]
                    if np.abs(E_n - E_i) < tol:
                        print('Warning: skipping near-degenerate point')
                        pass
                    else:
                        curv += (1j/(E_n - E_i)**2)*(((np.conjugate(u_n) @ vx @ u_i) 
                                                    * (np.conjugate(u_i) @ vy @ u_n))
                                                    - ((np.conjugate(u_n) @ vy @ u_i) 
                                                    * (np.conjugate(u_i) @ vx @ u_n)))
    if method == 'Fukui':
        if n0 is None:
            print('Error: eigenvectors not specified')  # Won't include code for diagonalizing if eigenvectors not specified
        eps = 1e-14
        U1 = np.vdot(n0, n1) #/ (np.abs(np.vdot(n0, n1)) + eps)
        U2 = np.vdot(n1, n12) #/ (np.abs(np.vdot(n1, n12)) + eps)
        U3 = np.vdot(n12, n2) #/ (np.abs(np.vdot(n12, n2)) + eps)
        U4 = np.vdot(n2, n0) #/ (np.abs(np.vdot(n2, n0)) + eps)
        # U1 /= np.abs(U1) + eps
        # U2 /= np.abs(U2) + eps
        # U3 /= np.abs(U3) + eps
        # U4 /= np.abs(U4) + eps
        # curv = np.log(U1*U2*U3*U4) / 1j
        # curv = np.angle(U1*U2*U3*U4)
        curv = np.angle(U1) + np.angle(U2) + np.angle(U3) + np.angle(U4)
        if np.abs(curv) > np.pi:
            # print('\n\nWarning: curvature outside (-pi,pi] range. Manually adjusting.')
            curv = (curv + np.pi) % (2*np.pi) - np.pi
        elif curv == -np.pi:
            curv = np.pi
    return curv


def calc_curv_line(n_vals, q_vals, basis=ABS.calc_square_basis_states(a=3, cutoff=2.5),
                   method='TKNN_all', extend_q=True, gauge_idx=None, sparse=True,
                   **kwargs):
    N_q = q_vals.shape[0]
    N_n = n_vals.size
    curv_vals = np.zeros((N_n, N_q), dtype=np.complex128)
    if method[:4] == 'TKNN':
        for i,q in enumerate(q_vals):
            print(f'Evaluating q point {i+1} out of {N_q}...' + 10*' ', end='\r')
            H = ABS.calc_H(q=q, basis=basis, **kwargs)
            evals, evects = np.linalg.eigh(H)
            vx, vy = calc_v(q=q, basis=basis)
            for k, n in enumerate(n_vals):
                curv_vals[k,i] = calc_curv_point(n=n, n_occ=n_vals, evals=evals, 
                                                evects=evects, vx=vx, vy=vy, 
                                                method=method, **kwargs)
        print('\nDone')
    elif method == 'Fukui':
        dq = q_vals[1,:] - q_vals[0,:]
        dq_orth = np.array([-dq[1], dq[0]])
        if extend_q:
            q_vals = np.append(q_vals, [q_vals[-1,:]+dq], axis=0)
        
        N_q = q_vals.shape[0]
        N_n = n_vals.size

        evects_arr = np.zeros((2*basis[0].shape[0], N_n, N_q, 2), dtype=np.complex128)
        curv_vals = np.zeros((N_n, N_q-1), dtype=np.complex128)

        # Calculate eigenvectors at every point
        print('Calculating eigenvectors:')
        for i in range(N_q):
            print(f'Evaluating q point {i+1} out of {N_q}...', end='\r')
            q1 = q_vals[i,:] - dq_orth/2
            H = ABS.calc_H(q=q1, basis=basis, **kwargs)
            if sparse:
                evals, evects = sp.sparse.linalg.eigsh(H, which='SA', 
                                                       k=n_vals.max+1)
                sorted_indices = np.argsort(evals)
                evals = evals[sorted_indices]
                evects = evects[:, sorted_indices]
            else:
                evals, evects = np.linalg.eigh(H)
            evects_arr[:,:,i,0] = evects[:,n_vals]
            q2 = q1 + dq_orth
            H = ABS.calc_H(q=q2, basis=basis, **kwargs)
            if sparse:
                evals, evects = sp.sparse.linalg.eigsh(H, which='SA', 
                                                       k=n_vals.max() + 1)
                sorted_indices = np.argsort(evals)
                evals = evals[sorted_indices]
                evects = evects[:, sorted_indices]
            else:
                evals, evects = np.linalg.eigh(H)
            evects_arr[:,:,i,1] = evects[:,n_vals]
        if gauge_idx is not None:
            evects_arr = fix_gauge(evects_arr, gauge_idx)
        print('\nDone')

        # Calculate curvature values from eigenvectors at each point
        print('Calculating curvature:')
        for i in range(N_q-1):
            for k in range(N_n):
                print(f'Evaluating point {i*N_n + k + 1} out of '
                    + f'{(N_q-1)*N_n}...' + 10*' ', end='\r')
                n0 = evects_arr[:,k,i,0]
                n1 = evects_arr[:,k,i+1,0]
                n12 = evects_arr[:,k,i+1,1]
                n2 = evects_arr[:,k,i,1]
                curv_vals[k,i] = calc_curv_point(n0, n1, n12, n2)
        print('\nDone')

    return curv_vals


def calc_curv_surface(n_vals, qx_vals, qy_vals, method='TKNN_all',
                      basis=ABS.calc_square_basis_states(a=3, cutoff=2.5), 
                      extend_q=True, gauge_idx=None, sparse=True, **kwargs):
    N_x = qx_vals.size
    N_y = qy_vals.size
    N_q = N_x*N_y
    N_n = n_vals.size
    curv_vals = np.zeros((N_n,N_x,N_y), dtype=np.complex128)
    if method[:4] == 'TKNN':
        for i, qx in enumerate(qx_vals):
            for j, qy in enumerate(qy_vals):
                print(f'Evaluating q point {i*N_y+j+1} out of {N_q}...' 
                      + 10*' ', end='\r')
                q = np.array([qx, qy])
                H = ABS.calc_H(q=q, basis=basis, **kwargs)
                evals, evects = np.linalg.eigh(H)
                vx, vy = calc_v(q=q, basis=basis)
                for k, n in enumerate(n_vals):
                    curv_vals[k,i,j] = calc_curv_point(n=n, n_occ=n_vals, 
                                                       evals=evals, 
                                                       evects=evects, vx=vx, 
                                                       vy=vy, method=method,
                                                       **kwargs)
        print('\nDone')
    elif method == 'Fukui':
        if extend_q:
        # Add extra q points to end of supplied values. Need the eigenvectors 
        # associated with these extra points in order to calculate curvatures
        # at the supplied set of points
            dqx = qx_vals[1] - qx_vals[0]
            qx_vals = np.append(qx_vals, qx_vals[-1] + dqx)
            dqy = qy_vals[1] - qy_vals[0]
            qy_vals = np.append(qy_vals, qy_vals[-1] + dqy)
        
        N_x = qx_vals.size
        N_y = qy_vals.size
        N_q = N_x*N_y
        N_n = n_vals.size
        evects_arr = np.zeros((2*basis[0].shape[0], N_n, N_x, N_y), dtype=np.complex128)
        curv_vals = np.zeros((N_n, N_x-1, N_y-1), dtype=np.complex128)

        # Calculate eigenvectors at every point
        print('Calculating eigenvectors:')
        for i, qx in enumerate(qx_vals):
            for j, qy in enumerate(qy_vals):
                print(f'Evaluating q point {i*N_y+j+1} out of {N_q}...' + 10*' ', 
                    end='\r')
                q = np.array([qx, qy])
                H = ABS.calc_H(q=q, basis=basis, **kwargs)
                if sparse:
                    evals, evects = sp.sparse.linalg.eigsh(H, which='SA', 
                                                        k=n_vals.max() + 1)
                    sorted_indices = np.argsort(evals)
                    evals = evals[sorted_indices]
                    evects = evects[:, sorted_indices]
                else:
                    evals, evects = np.linalg.eigh(H)
                evects_arr[:,:,i,j] = evects[:,n_vals]
                # for k, n in enumerate(n_vals):
                #     evects_arr[:,k,i,j] = evects[:,n]
        if gauge_idx is not None:
            evects_arr = fix_gauge(evects_arr, gauge_idx)
        print('\nDone')

        # Calculate curvature values from eigenvectors at each point
        print('Calculating curvature:')
        for i in range(N_x-1):
            for j in range(N_y-1):
                for k in range(N_n):
                    print(f'Evaluating point {i*(N_y-1)*N_n + j*N_n + k + 1} out of ' 
                        + f'{(N_x-1)*(N_y-1)*N_n}...' + 10*' ', end='\r')
                    n0 = evects_arr[:,k,i,j]
                    n1 = evects_arr[:,k,i+1,j]
                    n12 = evects_arr[:,k,i+1,j+1]
                    n2 = evects_arr[:,k,i,j+1]
                    curv_vals[k,i,j] = calc_curv_point(n0=n0, n1=n1, n12=n12, n2=n2,
                                                       method=method)
        print('\nDone')
    return curv_vals



def fix_gauge(evects, idx=0):
    # evects /= evects[idx, :, :, :][np.newaxis, ...]  # Normalize by idx-th slice
    # norms = np.linalg.norm(evects, axis=0, keepdims=True)  # Compute norms over axis 0
    # evects /= norms
    if np.any(np.abs(evects[idx,:,:,:] < 1e-14)):
        print('Warning: small-magnitude eigenvector components.')
    phi = np.angle(evects[idx, :, :, :])
    evects *= np.exp(-1j*phi)
    return evects


def square_BZ(dq=None, a=3, r0=np.zeros(2), color='k'):
    if dq is None:
        dq = 1/a
    vertices = np.array([[dq/2,dq/2],
                         [-dq/2,dq/2],
                         [-dq/2,-dq/2],
                         [dq/2,-dq/2]])
    vertices += np.outer(np.ones(4), r0)

    return Polygon(vertices, edgecolor=color, facecolor=(0,0,0,0))



def calc_chern_number(filename=None, bands=None, inside_BZ=False, patch=None, a=3,
                      shift_q=False, truncate_q=False, curv_vals=None, qx_vals=None,
                      qy_vals=None):
    if filename is not None:
        data = np.load(filename)
        qx_vals = data['qx_vals']
        qy_vals = data['qy_vals']
        if shift_q:
            dqx = qx_vals[1] - qx_vals[0]
            qx_vals = (qx_vals + dqx/2)[:-1]
            dqy = qy_vals[1] - qy_vals[0]
            qy_vals = (qy_vals + dqy/2)[:-1]
        elif truncate_q:
            qx_vals = qx_vals[:-1]
            qy_vals = qy_vals[:-1]
        curv_vals = data['curv_vals']
    elif curv_vals is None:
        print('Error: must specify curv_vals')
        return
    if bands is None:
        bands = np.arange(curv_vals.shape[0])
    # dqx = qx[1] - qx[0]
    # dqy = qy[1] - qy[0]
    if inside_BZ:
        C = 0
        BZ_path = square_BZ(a=0.999*a).get_path()   # Enlarge infinitesimally so no boundary issues
        qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
        # print(qxx.shape)
        points = np.column_stack((qxx.ravel(), qyy.ravel()))
        mask = BZ_path.contains_points(points)
        # curv_plot = np.full_like(curv_vals, 0.)
        # print(qxx.shape)
        # print(mask.reshape(qxx.shape))
        # curv_plot[:, mask.reshape(qxx.shape)] = curv_vals[:, mask.reshape(qxx.shape)]
        # print(curv_vals.shape)
        curv_vals = curv_vals[:, mask.reshape(qxx.shape)]
        # print(curv_vals.shape)
        curv_vals = curv_vals.reshape((curv_vals.shape[0], *qxx.shape))
        # print(curv_vals.shape)
        
        # for i in range(qx_vals.size):
        #     for j in range(qy_vals.size):
        #         if BZ_path.contains_points(np.array([[qx_vals[i],qy_vals[j]]])):
        #             C += np.sum(curv_vals[bands,i,j])
        # C *= dqx * dqy
    elif patch is not None:
        C = 0
        path = patch.get_path()
        qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
        points = np.column_stack((qxx.ravel(), qyy.ravel()))
        mask = path.contains_points(points)
        # curv_plot = np.full_like(curv_vals, 0.)
        # curv_plot[:, mask.reshape(qxx.shape)] = curv_vals[:, mask.reshape(qxx.shape)]
        curv_vals = curv_vals[:, mask.reshape(qxx.shape)]
        curv_vals = curv_vals.reshape((curv_vals.shape[0], *qxx.shape))
        # for i in range(qx_vals.size):
        #     for j in range(qy_vals.size):
        #         if path.contains_points(np.array([[qx_vals[i],qy_vals[j]]])):
        #             C += np.sum(curv_vals[bands,i,j])
        # C *= dqx * dqy
    C = np.sum(curv_vals[bands,:,:]) # * dqx * dqy
    # if data['method'] != 'Fukui':
    #     dqx = qx_vals[1] - qx_vals[0]
    #     dqy = qy_vals[1] - qy_vals[0]
    #     C *= dqx * dqy
    return C / (2*np.pi)


def K_point_peak_boundary(dq=0.028, color='k'):
    q_corners = np.array([[0, -dq]])
    R = np.array([[np.cos(2*np.pi/5), -np.sin(2*np.pi/5)],
                  [np.sin(2*np.pi/5), np.cos(2*np.pi/5)]])
    for i in range(4):
        q_corners = np.append(q_corners, [R@q_corners[i,:]], axis=0)
    q0 = np.column_stack((np.zeros(5), 0.5/np.cos(np.pi/10)*np.ones(5)))
    q_corners += q0
    patch = Polygon(q_corners, edgecolor=color, facecolor=(0,0,0,0))
    return patch


def calc_curvature_fromfile(filename=None, evects_arr=None, calc_chern=True, save=True, bands=None,
                            save_filename='auto', gauge_idx=None):
    if filename is not None:
        data = np.load(filename)
        evects_arr = data['evects_arr']
    elif evects_arr is None:
        print('Error: eigenvectors not specified')
        return
    (N_n, N_x, N_y) = evects_arr.shape[1:]
    curv_vals = np.zeros((N_n, N_x-1, N_y-1), dtype=np.complex128)
    if gauge_idx is not None:
        evects_arr = fix_gauge(evects_arr, idx=gauge_idx)
    print('Calculating curvature:')
    for i in range(N_x-1):
        for j in range(N_y-1):
            for k in range(N_n):
                print(f'Evaluating point {i*(N_y-1)*N_n + j*N_n + k + 1} out of ' 
                    + f'{(N_x-1)*(N_y-1)*N_n}...' + 10*' ', end='\r')
                n0 = evects_arr[:,k,i,j]
                n1 = evects_arr[:,k,i+1,j]
                n12 = evects_arr[:,k,i+1,j+1]
                n2 = evects_arr[:,k,i,j+1]
                curv_vals[k,i,j] = np.real(calc_curv_point(n0=n0, n1=n1, n12=n12, n2=n2,
                                                    method='Fukui'))
    print('\nDone')
    
    if calc_chern:
        print('Calculating Chern number...')
        C = calc_chern_number(curv_vals=curv_vals, bands=bands)  # Assumes that curv_vals exactly tiles the BZ
        print('Done')
    
    if save:
        param_keys = {'U0', 'V0', 'V', 'N', 'basis', 'a', 'cutoff', 'qx_vals', 
                  'qy_vals'}
        params = {}
        for key in param_keys:
            if key in data:
                params[key] = data[key]
        if calc_chern:
            params['C'] = C
        if save_filename == 'auto':
            save_filename = 'Curv_approx_a3_Fukui'
            name_params = {'U0':'U', 'N':'N', 'V0':'V'}
            for key, val in name_params.items():
                if key in data:
                    save_filename += '_' + val + str(data[key])
            save_filename += '.npz'

        np.savez(save_filename, curv_vals=curv_vals, **params)
    
    else:
        if calc_chern:
            return curv_vals, C
        else:
            return curv_vals
        


def calc_curv_NonAb_point(n0, n1, n12, n2):
    A1 = np.conj(n0).T @ n1
    U1 = np.linalg.det(A1)
    A2 = np.conj(n1).T @ n12
    U2 = np.linalg.det(A2)
    A3 = np.conj(n12).T @ n2
    U3 = np.linalg.det(A3)
    A4 = np.conj(n2).T @ n0
    U4 = np.linalg.det(A4)
    curv = np.angle(U1) + np.angle(U2) + np.angle(U3) + np.angle(U4)
    if np.abs(curv) > np.pi:
        # print('\n\nWarning: curvature outside (-pi,pi] range. Manually adjusting.')
        curv = (curv + np.pi) % (2*np.pi) - np.pi
    elif curv == -np.pi:
        curv = np.pi
    return curv
        


def calc_curvature_NonAb_fromfile(filename=None, evects_arr=None, n_vals=None,
                                  calc_chern=True, save=True,
                                  save_filename='auto', gauge_idx=None):
    if filename is not None:
        data = np.load(filename)
        evects_arr = data['evects_arr']
    elif evects_arr is None:
        print('Error: eigenvectors not specified')
        return
    (N_n, N_x, N_y) = evects_arr.shape[1:]
    if n_vals is None:
        n_vals = np.arange(N_n)
    curv_vals = np.zeros((1, N_x-1, N_y-1), dtype=np.complex128)
    if gauge_idx is not None:
        evects_arr = fix_gauge(evects_arr, idx=gauge_idx)
    print('Calculating curvature:')
    for i in range(N_x-1):
        for j in range(N_y-1):
            print(f'Evaluating point {i*(N_y-1) + j + 1} out of ' 
                + f'{(N_x-1)*(N_y-1)}...' + 10*' ', end='\r')
            n0 = evects_arr[:,n_vals,i,j]
            n1 = evects_arr[:,n_vals,i+1,j]
            n12 = evects_arr[:,n_vals,i+1,j+1]
            n2 = evects_arr[:,n_vals,i,j+1]
            curv_vals[0,i,j] = calc_curv_NonAb_point(n0=n0, n1=n1, n12=n12, n2=n2)
    print('\nDone')
    
    if calc_chern:
        print('Calculating Chern number...')
        C = calc_chern_number(curv_vals=curv_vals, bands=None)  # Assumes that curv_vals exactly tiles the BZ
        print('Done')
    
    if save:
        param_keys = {'U0', 'V0', 'V', 'N', 'basis', 'a', 'cutoff', 'qx_vals', 
                  'qy_vals'}
        params = {}
        for key in param_keys:
            if key in data:
                params[key] = data[key]
        if calc_chern:
            params['C'] = C
        if save_filename == 'auto':
            save_filename = 'Curv_approx_a3_Fukui'
            name_params = {'U0':'U', 'N':'N', 'V0':'V'}
            for key, val in name_params.items():
                if key in data:
                    save_filename += '_' + val + str(data[key])
            save_filename += '.npz'

        np.savez(save_filename, curv_vals=curv_vals, **params)
    
    else:
        if calc_chern:
            return curv_vals, C
        else:
            return curv_vals


# def calc_curv_surface_NonAb(n_vals, qx_vals, qy_vals,
#                             basis=ABS.calc_square_basis_states(a=3, cutoff=2.5), 
#                             extend_q=True, gauge_idx=None, sparse=True, 
#                             **kwargs):
#     N_x = qx_vals.size
#     N_y = qy_vals.size
#     N_q = N_x*N_y
#     N_n = n_vals.size
#     curv_vals = np.zeros((N_x,N_y), dtype=np.complex128)
    
#     if extend_q:
#     # Add extra q points to end of supplied values. Need the eigenvectors 
#     # associated with these extra points in order to calculate curvatures
#     # at the supplied set of points
#         dqx = qx_vals[1] - qx_vals[0]
#         qx_vals = np.append(qx_vals, qx_vals[-1] + dqx)
#         dqy = qy_vals[1] - qy_vals[0]
#         qy_vals = np.append(qy_vals, qy_vals[-1] + dqy)
    

#     evects_arr = np.zeros((2*basis[0].shape[0], N_n, N_x, N_y), dtype=np.complex128)
#     curv_vals = np.zeros((N_n, N_x-1, N_y-1), dtype=np.complex128)

#     # Calculate eigenvectors at every point
#     print('Calculating eigenvectors:')
#     for i, qx in enumerate(qx_vals):
#         for j, qy in enumerate(qy_vals):
#             print(f'Evaluating q point {i*N_y+j+1} out of {N_q}...' + 10*' ', 
#                 end='\r')
#             q = np.array([qx, qy])
#             H = ABS.calc_H(q=q, basis=basis, **kwargs)
#             if sparse:
#                 evals, evects = sp.sparse.linalg.eigsh(H, which='SA', 
#                                                     k=n_vals.max() + 1)
#                 sorted_indices = np.argsort(evals)
#                 evals = evals[sorted_indices]
#                 evects = evects[:, sorted_indices]
#             else:
#                 evals, evects = np.linalg.eigh(H)
#             evects_arr[:,:,i,j] = evects[:,n_vals]
#             # for k, n in enumerate(n_vals):
#             #     evects_arr[:,k,i,j] = evects[:,n]
#     if gauge_idx is not None:
#         evects_arr = fix_gauge(evects_arr, gauge_idx)
#     print('\nDone')

#     # Calculate curvature values from eigenvectors at each point
#     print('Calculating curvature:')
#     for i in range(N_x-1):
#         for j in range(N_y-1):
#             for k in range(N_n):
#                 print(f'Evaluating point {i*(N_y-1)*N_n + j*N_n + k + 1} out of ' 
#                     + f'{(N_x-1)*(N_y-1)*N_n}...' + 10*' ', end='\r')
#                 n0 = evects_arr[:,k,i,j]
#                 n1 = evects_arr[:,k,i+1,j]
#                 n12 = evects_arr[:,k,i+1,j+1]
#                 n2 = evects_arr[:,k,i,j+1]
#                 curv_vals[k,i,j] = calc_curv_point(n0=n0, n1=n1, n12=n12, n2=n2, 
#                                                    method='Fukui')
#     print('\nDone')
#     return curv_vals


def calc_chern_multiband(curv_vals, **kwargs):
    C = np.zeros(curv_vals.shape[0])
    for i in range(curv_vals.shape[0]):
        C[i] = np.real(calc_chern_number(curv_vals=curv_vals, bands=[i], **kwargs))
    return C
    



if __name__ == '__main__':
    U0 = 0.001
    V0 = 0.02
    #V = V0 * np.array([1,-1,-1,-1,1,1,-1,-1,-1,1])
    N = 1
    # q_vals = np.column_stack((np.zeros(300),np.linspace(0,0.6,300)))
    # n_vals = np.arange(8)
    # curv_vals = calc_curv_line(n_vals=n_vals, q_vals=q_vals, U0=U0, V0=V0, 
    #                            N=N)
    # np.savez('TKNN_test2.npz', q_vals=q_vals, curv_vals=curv_vals, U0=U0, V0=V0, 
    #          N=N, n_vals=n_vals)




    