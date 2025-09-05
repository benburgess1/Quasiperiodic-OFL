import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import Calc_Bandstructure as CBS
from matplotlib.patches import Polygon
import scipy as sp


# Calculate reciprocal lattice (G) vectors 
G0 = 1
G_vects = G0 * np.column_stack((np.cos(np.arange(10)*np.pi/5),
                                np.sin(np.arange(10)*np.pi/5)))




def calc_curv_point(n0=None, n1=None, n12=None, n2=None, **kwargs):
    if n0 is None:
        print('Error: eigenvectors not specified')  # Won't include code for diagonalizing if eigenvectors not specified
        return
    eps = 1e-14
    U1 = np.vdot(n0, n1) #/ (np.abs(np.vdot(n0, n1)) + eps)
    U2 = np.vdot(n1, n12) #/ (np.abs(np.vdot(n1, n12)) + eps)
    U3 = np.vdot(n12, n2) #/ (np.abs(np.vdot(n12, n2)) + eps)
    U4 = np.vdot(n2, n0) #/ (np.abs(np.vdot(n2, n0)) + eps)

    curv = np.angle(U1) + np.angle(U2) + np.angle(U3) + np.angle(U4)
    if np.abs(curv) > np.pi:
        # print('\n\nWarning: curvature outside (-pi,pi] range. Manually adjusting.')
        curv = (curv + np.pi) % (2*np.pi) - np.pi
    elif curv == -np.pi:
        curv = np.pi
    return curv


def calc_curv_surface(n_vals, qx_vals, qy_vals, 
                      basis=None, orders=4, cutoff=None,
                      extend_q=True, gauge_idx=None, sparse=False, **kwargs):
    if basis is None:
        basis = CBS.calc_basis_states(orders=orders, cutoff=cutoff)

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
    if sparse:
        num_evals = np.max(n_vals) + 1
    else:
        num_evals = 2*basis[0].shape[0]
    evals, evects_arr = CBS.calc_BS_surface(qx=qx_vals, qy=qy_vals, basis=basis, 
                                            return_evects=True, num_evals=num_evals, 
                                            sparse=sparse, **kwargs)

    # for i, qx in enumerate(qx_vals):
    #     for j, qy in enumerate(qy_vals):
    #         print(f'Evaluating q point {i*N_y+j+1} out of {N_q}...' + 10*' ', 
    #             end='\r')
    #         q = np.array([qx, qy])
    #         H = CBS.calc_H(q=q, basis=basis, **kwargs)
    #         if sparse:
    #             evals, evects = sp.sparse.linalg.eigsh(H, which='SA', 
    #                                                 k=n_vals.max() + 1)
    #             sorted_indices = np.argsort(evals)
    #             evals = evals[sorted_indices]
    #             evects = evects[:, sorted_indices]
    #         else:
    #             evals, evects = np.linalg.eigh(H)
    #         evects_arr[:,:,i,j] = evects[:,n_vals]
    #         # for k, n in enumerate(n_vals):
    #         #     evects_arr[:,k,i,j] = evects[:,n]
    if gauge_idx is not None:
        evects_arr = fix_gauge(evects_arr, gauge_idx)
    print('\nDone')

    # Calculate curvature values from eigenvectors at each point
    print('Calculating curvature:')
    for i in range(N_x-1):
        for j in range(N_y-1):
            for k,n in enumerate(n_vals):
                print(f'Evaluating point {i*(N_y-1)*N_n + j*N_n + k + 1} out of ' 
                    + f'{(N_x-1)*(N_y-1)*N_n}...' + 10*' ', end='\r')
                n0 = evects_arr[:,n,i,j]
                n1 = evects_arr[:,n,i+1,j]
                n12 = evects_arr[:,n,i+1,j+1]
                n2 = evects_arr[:,n,i,j+1]
                curv_vals[k,i,j] = np.real(calc_curv_point(n0=n0, n1=n1, n12=n12, n2=n2))
    print('\nDone')
    return curv_vals


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
                curv_vals[k,i,j] = np.real(calc_curv_point(n0=n0, n1=n1, n12=n12, n2=n2))
    print('\nDone')
    
    if calc_chern:
        print('Calculating Chern number...')
        C = calc_chern_number(curv_vals=curv_vals, bands=bands)  # Assumes that curv_vals exactly tiles the BZ
        print('Done')
    
    if save:
        param_keys = {'U0', 'V0', 'V', 'N', 'basis', 'orders', 'cutoff', 'qx_vals', 
                  'qy_vals'}
        params = {}
        for key in param_keys:
            if key in data:
                params[key] = data[key]
        if calc_chern:
            params['C'] = C
        if save_filename == 'auto':
            save_filename = 'Curv_'
            name_params = {'orders':'o', 'U0':'U', 'N':'N', 'V0':'V'}
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
                                  save_filename='auto', gauge_idx=None, **kwargs):
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
        C = calc_chern_number(curv_vals=curv_vals, bands=None, **kwargs)
        print('Done')
    
    if save:
        param_keys = {'U0', 'V0', 'V', 'N', 'basis', 'orders', 'cutoff', 'qx_vals', 
                  'qy_vals'}
        params = {}
        for key in param_keys:
            if key in data:
                params[key] = data[key]
        if calc_chern:
            params['C'] = C
        if save_filename == 'auto':
            save_filename = 'Curv_approx_a3_Fukui'
            name_params = {'orders':'o', 'U0':'U', 'N':'N', 'V0':'V'}
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



def fix_gauge(evects, idx=0):
    # evects /= evects[idx, :, :, :][np.newaxis, ...]  # Normalize by idx-th slice
    # norms = np.linalg.norm(evects, axis=0, keepdims=True)  # Compute norms over axis 0
    # evects /= norms
    if np.any(np.abs(evects[idx,:,:,:] < 1e-14)):
        print('Warning: small-magnitude eigenvector components.')
    phi = np.angle(evects[idx, :, :, :])
    evects *= np.exp(-1j*phi)
    return evects


def calc_chern_number(filename=None, bands=None, inside_QBZ=False, invert=False, 
                      patch=None, shift_q=False, truncate_q=False, 
                      curv_vals=None, qx_vals=None, qy_vals=None, **kwargs):
    if filename is not None:
        data = np.load(filename)
        qx_vals = data['qx_vals']
        qy_vals = data['qy_vals']
        curv_vals = data['curv_vals']
    elif curv_vals is None:
        print('Error: must specify curv_vals')
        return
    if shift_q:
        dqx = qx_vals[1] - qx_vals[0]
        qx_vals = (qx_vals + dqx/2)[:-1]
        dqy = qy_vals[1] - qy_vals[0]
        qy_vals = (qy_vals + dqy/2)[:-1]
    elif truncate_q:
        qx_vals = qx_vals[:-1]
        qy_vals = qy_vals[:-1]
    if bands is None:
        bands = np.arange(curv_vals.shape[0])
    # dqx = qx[1] - qx[0]
    # dqy = qy[1] - qy[0]
    if inside_QBZ:
        if invert != 'both':
            QBZ_path = mpl.path.Path(vertices=CBS.calc_pentagon(G=1.0001, invert=invert))     # Enlarge infinitesimally so no boundary issues
            qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
            points = np.column_stack((qxx.ravel(), qyy.ravel()))
            mask = QBZ_path.contains_points(points).reshape(qxx.shape)
            curv_vals = np.where(mask[None, :, :], curv_vals, 0)
            # curv_vals = curv_vals[:, mask.reshape(qxx.shape)]
            # curv_vals = curv_vals.reshape((curv_vals.shape[0], *qxx.shape))
        else:
            curv_vals_2 = np.copy(curv_vals)
            qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
            points = np.column_stack((qxx.ravel(), qyy.ravel()))

            QBZ_path1 = mpl.path.Path(vertices=CBS.calc_pentagon(G=1.0001, invert=True))
            mask = QBZ_path1.contains_points(points).reshape(qxx.shape)
            curv_vals = np.where(mask[None, :, :], curv_vals, 0)
            # curv_vals = curv_vals.reshape((curv_vals.shape[0], *qxx.shape))

            QBZ_path2 = mpl.path.Path(vertices=CBS.calc_pentagon(G=1.0001, invert=False))
            mask = QBZ_path2.contains_points(points).reshape(qxx.shape)
            curv_vals_2 = np.where(mask[None, :, :], curv_vals_2, 0)
            # curv_vals_2 = curv_vals_2.reshape((curv_vals.shape[0], *qxx.shape))

    elif patch is not None:
        path = patch.get_path()
        qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
        points = np.column_stack((qxx.ravel(), qyy.ravel()))
        mask = path.contains_points(points).reshape(qxx.shape)
        curv_vals = np.where(mask[None, :, :], curv_vals, 0)
        # curv_vals = curv_vals.reshape((curv_vals.shape[0], *qxx.shape))

    if invert != 'both':
        C = np.sum(curv_vals[bands]) # * dqx * dqy
    else:
        C = np.array([np.sum(curv_vals[bands]),
                      np.sum(curv_vals_2[bands])])

    return C / (2*np.pi)


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




    