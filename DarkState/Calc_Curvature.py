import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Approximant'))
import Approximant_Vectors as AV
import Dark_Approximant as DA


N = 8
l = np.arange(N)
q8 = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
phi8 = np.zeros((N,2))

def s_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    s = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            s[i] += np.exp(1j * ((-1)**(i+1) * q.dot(r) + phi[a,i]))
    return s


def d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += (-1)**i * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]))
    return d


def del_p_d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    q_mag = np.linalg.norm(q_vals[0,:])
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += -1j * (q[0] + 1j*q[1]) * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]))
            # d[i] += -1j * q_mag * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
            #                                - phi[a,(i+1)%2]
            #                                + 2*np.pi*a/N))
    return d


def del_m_d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    q_mag = np.linalg.norm(q_vals[0,:])
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += -1j * q_mag * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]
                                           - 2*np.pi*a/N))
    return d


def del_x_d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += -1j * q[0] * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]))
    return d


def del_y_d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += -1j * q[1] * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]))
    return d


def verify_derivative(N=8, p1=0, p2=None):
    l = np.arange(N)
    q = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    q_mag = np.linalg.norm(q[0,:])
    if p2 is None:
        p2 = N - 1 - p1
    phi = np.zeros((N, 2))
    for a in range(N):
        phi[a,0] = - 2 * np.pi * a * p1 / N
        phi[a,1] = - 2 * np.pi * a * p2 / N
    # phi = np.random.rand(8,2) * 2*np.pi
    r = np.random.rand(2)
    d = d_state(r, q_vals=q, phi=phi)
    s = s_state(r, q_vals=q, phi=phi)
    del_p_d = del_p_d_state(r, q_vals=q, phi=phi)
    diff = -1j * q_mag * s - del_p_d
    print(f'Momentum transfers:\n{q}')
    print(f'Phase factors:\n{phi}')
    print(f'Position: {r}')
    print(f's = {s}')
    print(f'd = {d}')
    print(f'del_p_d = {del_p_d}')
    print(f'|-i|q|s - del_p_d| = {np.linalg.norm(diff)}')   # Should vanish


def calc_curv_point(r, **kwargs):
    d = d_state(r, **kwargs)
    dx = del_x_d_state(r, **kwargs)
    dy = del_y_d_state(r, **kwargs)
    dydx = np.vdot(dy, dx)
    ddx = np.vdot(d, dx)
    dyd = np.vdot(dy, d)
    dd = np.vdot(d, d)
    curv = np.imag(-2 * dydx / dd + 2 * ddx * dyd / dd**2)
    return curv

def verify_curvature(N=8, p1=0, p2=None, h=0.001, r=np.random.rand(2)):
    l = np.arange(N)
    q = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    if p2 is None:
        p2 = 1 - p1
    phi = np.zeros((N, 2))
    for a in range(N):
        phi[a,0] = - 2 * np.pi * a * p1 / N
        phi[a,1] = - 2 * np.pi * a * p2 / N
    # phi = np.random.rand(8,2) * 2*np.pi
    d00 = d_state(r, q_vals=q, phi=phi)
    d00 /= np.linalg.norm(d00)
    d10 = d_state(r + np.array([h,0]), q_vals=q, phi=phi)
    d10 /= np.linalg.norm(d10)
    d11 = d_state(r + np.array([h,h]), q_vals=q, phi=phi)
    d11 /= np.linalg.norm(d11)
    d01 = d_state(r + np.array([0,h]), q_vals=q, phi=phi)
    d01 /= np.linalg.norm(d01)
    # Ax0 = 1j * np.vdot(d00, d10-d00) / h
    # Ax1 = 1j * np.vdot(d01, d11-d01) / h
    # Ay0 = 1j * np.vdot(d00, d01-d00) / h
    # Ay1 = 1j * np.vdot(d10, d11-d10) / h
    # curv_num = ((Ay1 - Ay0) - (Ax1 - Ax0)) / h
    dxd = (d10 - d00) / h
    dyd = (d01 - d00) / h
    curv_num = 1j * (np.vdot(dyd, dxd) - np.vdot(dxd, dyd))

    curv_ana = calc_curv_point(r, q_vals=q, phi=phi)
    print(f'Point r = {r}')
    print(f'Numerical curvature: {curv_num}')
    print(f'Analytical curvature: {curv_ana}')


def calc_curv_surface(x_vals, y_vals, save=True, filename='Data.npz', **kwargs):
    Nx = x_vals.size
    Ny = y_vals.size
    curv_vals = np.zeros((Nx, Ny))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            print(f'Evaluating point {i*Ny + j + 1} out of {Nx * Ny}...' + 10*' ', end='\r')
            curv_vals[i,j] = calc_curv_point(r=np.array([x,y]), **kwargs)
    print('\nDone')
    if save:
        np.savez(filename, x_vals=x_vals, y_vals=y_vals, curv_vals=curv_vals, 
                 **kwargs)
    else:
        return curv_vals
    

def calc_d_surface(x_vals, y_vals, save=True, filename='Data.npz', **kwargs):
    Nx = x_vals.size
    Ny = y_vals.size
    d_vals = np.zeros((Nx, Ny))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            print(f'Evaluating point {i*Ny + j + 1} out of {Nx * Ny}...' + 10*' ', end='\r')
            d = d_state(r=np.array([x,y]), **kwargs)
            d_vals[i,j] = np.linalg.norm(d)
    print('\nDone')
    if save:
        np.savez(filename, x_vals=x_vals, y_vals=y_vals, d_vals=d_vals, 
                 **kwargs)
    else:
        return d_vals
    

def calc_U_exact(x, y, **kwargs):
    dpd = del_p_d_state(r=np.array([x,y]), **kwargs)
    d = d_state(r=np.array([x,y]), **kwargs)
    s = s_state(r=np.array([x,y]), **kwargs)
    # Q = np.eye(2) - np.outer(d, d.conj()) / np.linalg.norm(d)**2
    Q = np.outer(s, s.conj()) / np.linalg.norm(s)**2
    # U = 0
    # U += np.vdot(dpd, dpd)
    # U += np.abs(np.vdot(d, dpd))**2 / np.linalg.norm(d)**2
    # U /= np.linalg.norm(d)**2
    U = dpd.conj().T @ Q @ dpd / np.linalg.norm(d)**2
    return np.real(U)


def calc_U_exact_surface(x_vals, y_vals, save=True, filename='Data.npz', **kwargs):
    Nx = x_vals.size
    Ny = y_vals.size
    U_vals = np.zeros((Nx, Ny))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            print(f'Evaluating point {i*Ny + j + 1} out of {Nx * Ny}...' + 10*' ', end='\r')
            U_vals[i,j] = calc_U_exact(x, y, **kwargs)
    print('\nDone')
    if save:
        np.savez(filename, x_vals=x_vals, y_vals=y_vals, U_vals=U_vals, 
                 **kwargs)
    else:
        return U_vals
    

def v_state(r, q_exact=None, q_approx=None, a=3, phi=None, N=8, **kwargs):
    if q_exact is None:
        l = np.arange(N)
        q_exact = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    if q_approx is None:
        q_approx = AV.square_approximant(a=a, G_vects=q_exact)
    dq = q_approx - q_exact
    N = q_exact.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    v = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_approx):
            v[i] += (dq[a,0] + 1j*dq[a,1]) * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]))
            # d[i] += -1j * q_mag * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
            #                                - phi[a,(i+1)%2]
            #                                + 2*np.pi*a/N))
    return v


def calc_U_approx(x, y, **kwargs):
    r = np.array([x,y])
    s = s_state(r=r, **kwargs)
    d = d_state(r=r, **kwargs)
    v = v_state(r=r, **kwargs)
    U = 2*np.real(np.vdot(s, v)) / np.linalg.norm(d)**2 + 1
    return U


def calc_U_approx_surface(x_vals, y_vals, save=True, filename='Data.npz', **kwargs):
    Nx = x_vals.size
    Ny = y_vals.size
    U_vals = np.zeros((Nx, Ny))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            print(f'Evaluating point {i*Ny + j + 1} out of {Nx * Ny}...' + 10*' ', end='\r')
            U_vals[i,j] = calc_U_approx(x, y, **kwargs)
    print('\nDone')
    if save:
        np.savez(filename, x_vals=x_vals, y_vals=y_vals, U_vals=U_vals, 
                 **kwargs)
    else:
        return U_vals
    

def calc_sv_surface(x_vals, y_vals, save=True, filename='Data.npz', **kwargs):
    Nx = x_vals.size
    Ny = y_vals.size
    sv_vals = np.zeros((Nx, Ny), dtype=np.complex128)
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            print(f'Evaluating point {i*Ny + j + 1} out of {Nx * Ny}...' + 10*' ', end='\r')
            s = s_state(r=np.array([x,y]), **kwargs)
            v = v_state(r=np.array([x,y]), **kwargs)            
            sv_vals[i,j] = np.vdot(s, v)
    print('\nDone')
    if save:
        np.savez(filename, x_vals=x_vals, y_vals=y_vals, sv_vals=sv_vals, 
                 **kwargs)
    else:
        return sv_vals
    

def calc_U_FT(filename, save=True, extend=True, save_filename='U_FT.npz'):
    data = np.load(filename)
    U = data['U_vals']
    x_vals = data['x_vals']
    y_vals = data['y_vals']

    # x_vals = x_vals[:-1]
    # y_vals = y_vals[:-1]

    print('Evaluating Fourier transform...')
    U_FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U), norm='forward'))
    print('Done')

    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    kx = np.fft.fftfreq(len(x_vals), dx)
    ky = np.fft.fftfreq(len(y_vals), dy)
    kx = np.fft.fftshift(kx) * 2 * np.pi
    ky = np.fft.fftshift(ky) * 2 * np.pi

    if save:
        if extend:
            save_dict = {key: data[key] for key in data.files}
            save_dict.update({
                "U_FT": U_FT,
                "kx_vals": kx ,
                "ky_vals": ky
            })
            np.savez(filename, **save_dict)
        else:
            np.savez(save_filename, U_FT=U_FT, kx_vals=kx, ky_vals=ky)
    else:
        return U_FT, kx, ky
    

def calc_d_FT(filename, save=True, extend=True, save_filename='U_FT.npz'):
    data = np.load(filename)
    d = data['d_vals']
    x_vals = data['x_vals']
    y_vals = data['y_vals']

    # x_vals = x_vals[:-1]
    # y_vals = y_vals[:-1]

    print('Evaluating Fourier transform...')
    d_FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(d), norm='forward'))
    print('Done')

    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    kx = np.fft.fftfreq(len(x_vals), dx)
    ky = np.fft.fftfreq(len(y_vals), dy)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    if save:
        if extend:
            save_dict = {key: data[key] for key in data.files}
            save_dict.update({
                "d_FT": d_FT,
                "kx_vals": kx,
                "ky_vals": ky
            })
            np.savez(filename, **save_dict)
        else:
            np.savez(save_filename, d_FT=d_FT, kx_vals=kx, ky_vals=ky)
    else:
        return d_FT, kx, ky
    

def calc_req_x(kmax, dk=None, a=3):
    if dk is None:
        dk = (1/a) / 2
    dx = np.pi / kmax
    Lx = 2 * np.pi / dk
    # Nx = 2 * kmax / dk
    # print(Nx)
    # xmax = (Nx-1)/2 * dx
    # x_vals = np.arange(-Lx/2, Lx/2+dx/2, dx)
    # x_vals = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, int(Nx))
    x_vals = np.arange(-Lx/2 + dx/2, Lx/2, dx)
    return x_vals


def calc_dU(filename, save_filename):
    data = np.load(filename)
    U_vals = data['U_vals']
    dU = U_vals - 1
    dU_FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(dU), norm='forward'))
    save_dict = {key:data[key] for key in data.files}
    save_dict.update({
        'dU':dU,
        'dU_FT':dU_FT
    })
    save_dict.pop('U_vals')
    save_dict.pop('U_FT')
    np.savez(save_filename, **save_dict)


def calc_dV(f_dU, f_d, M=100, reconstruct=True, save_filename='dV.npz',
            save_params=['x_vals', 'y_vals', 'kx_vals', 'ky_vals', 'a', 
                         'N', 'p1', 'p2', 'phi1', 'phi2']):
    d1 = np.load(f_dU)
    dU = d1['dU']
    kx_vals, ky_vals = d1['kx_vals'], d1['ky_vals']
    d2 = np.load(f_d)
    d = d2['d_vals']

    dV = -dU / d**2
    dV_FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(dV), norm='forward'))

    kxx, kyy = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    dV_FT_flat = dV_FT.flatten()
    k_flat = np.column_stack((kxx.flatten(), kyy.flatten()))
    sort_idxs = np.argsort(np.abs(dV_FT_flat))[::-1]
    dV_FT_sorted = dV_FT_flat[sort_idxs]
    k_sorted = k_flat[sort_idxs,:]
    dV_FT_list = dV_FT_sorted[:M]
    k_list = k_sorted[:M,:]

    save_dict = {key:d1[key] for key in save_params}
    save_dict.update({
        'dV_FT_list':dV_FT_list,
        'k_list':k_list,
        'dV_FT':dV_FT,
        'dV':dV,
        'M': M,
    })
    
    if reconstruct:
        dV_FT_extracted = reconstruct_dV(dV_FT_list, k_list, kx_vals, ky_vals)
        dV_r_extracted = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(dV_FT_extracted), norm='forward'))
        save_dict.update({
            'dV_FT_extracted':dV_FT_extracted,
            'dV_r_extracted':dV_r_extracted
        })
    
    np.savez(save_filename, **save_dict)



# def calc_dU(filename=None, U_FT=None, kx_vals=None, ky_vals=None, kfactor=1, a=3, cutoff=5.5,
#             save=True, save_filename='dU.npz', M=None, 
#             save_params=['x_vals', 'y_vals', 'kx_vals', 'ky_vals', 'a', 
#                          'N', 'p1', 'p2', 'phi1', 'phi2'], 
#             reconstruct=True):
#     if filename is not None:
#         data = np.load(filename)
#         U_FT = data['U_FT']
#         kx_vals = data['kx_vals'] * kfactor
#         ky_vals = data['ky_vals'] * kfactor
#     L = np.floor(cutoff*a) / a
#     bx_vals = np.arange(-L, L, 1/a)
#     # print(bx_vals)
#     by_vals = np.copy(bx_vals)
#     N = bx_vals.shape[0]
#     # print(N)
#     U_found = np.zeros((N,N), dtype=np.complex128)
#     for i, bx in enumerate(bx_vals):
#         x_idx = np.argmin(np.abs(kx_vals - bx))
#         for j, by in enumerate(by_vals):
#             y_idx = np.argmin(np.abs(ky_vals - by))
#             U_found[i,j] = U_FT[x_idx, y_idx]
#     bxx, byy = np.meshgrid(bx_vals, by_vals, indexing='ij')
#     # print(U_found.shape)
#     # print(bxx.shape)
#     # print(bxx.flatten().shape)
#     U_found = U_found.flatten()
#     k_found = np.column_stack((bxx.flatten(), byy.flatten()))
#     # print(U_found.shape)
#     # print(k_found.shape)
#     if M is not None:
#         sort_idxs = np.argsort(np.abs(U_found))[::-1]
#         # print(sort_idxs[:5])
#         U_found = U_found[sort_idxs]
#         k_found = k_found[sort_idxs,:]
#         # print(k_found.shape)
#         U_found = U_found[1:M+1]
#         k_found = k_found[1:M+1,:]
#         # print(k_found.shape)
#     if reconstruct:
#         U_FT = reconstruct_dU(U_found, k_found, kx_vals, ky_vals)
#         k_found = np.column_stack((kx_vals, ky_vals))
#         U_vals = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_FT), norm='forward'))
#     else:
#         U_vals = None
#     if save:
#         save_dict = {key:data[key] for key in save_params}
#         save_dict.update({
#                 "U_FT": U_FT,
#                 'U_vals': U_vals,
#                 "kx_vals": k_found[:,0],
#                 "ky_vals": k_found[:,1],
#                 'M': M
#             })
#         np.savez(save_filename, **save_dict)
#     else:
#         return U_found, k_found
    

def reconstruct_dV(V_FT, k_found, target_kx, target_ky):
    Nx = target_kx.size
    Ny = target_ky.size
    V_rec = np.zeros((Nx, Ny), dtype=np.complex128)
    for i,k in enumerate(k_found):
        # print(f'k = {k}')
        x_idx = np.argmin(np.abs(target_kx-k[0]))
        y_idx = np.argmin(np.abs(target_ky-k[1]))
        # print(f'Closest k = {[target_kx[x_idx], target_ky[y_idx]]}')
        V_rec[x_idx, y_idx] = V_FT[i]
    return V_rec


def calc_compensated_U(f_dV, f_d, f_U, save=True, save_filename='U_compensated.npz'):
    d1 = np.load(f_dV)
    dV = d1['dV_r_extracted']
    d2 = np.load(f_d)
    d = d2['d_vals']
    d3 = np.load(f_U)
    U = d3['U_vals']

    # dV = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(dV_FT_extracted), norm='forward'))
    # dV = d1['dV']
    U_vals = U + dV * d**2

    if save:
        save_dict = {key: d1[key] for key in d1.files}
        save_dict.update({
            'U_vals':U_vals
        })
        np.savez(save_filename, **save_dict)


# def calc_dV(f_dU, f_d, save_params=['x_vals', 'y_vals', 'kx_vals', 'ky_vals', 'a', 
#                                  'N', 'p1', 'p2', 'phi1', 'phi2', 'M'], save_filename='dV.npz'):
#     d1 = np.load(f_dU)
#     d2 = np.load(f_d)
#     dU = d1['U_FT']
#     d_vals = d2['d_vals']
#     dV = -dU / d_vals**2
#     dV_FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(dV), norm='forward'))
#     save_dict = {key:d1[key] for key in save_params}
#     save_dict.update({
#         'dV':dV,
#         'dV_FT':dV_FT
#     })
#     np.savez(save_filename, **save_dict)


def calc_approximant_fluxes(a_vals, dx=0.05, filename='Fluxes.npz', N=5, offset=0., **kwargs):
    flux_vals = np.zeros(a_vals.size)
    l = np.arange(N)
    q_exact = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    for i, a in enumerate(a_vals):
        print(f'Evaluating a = {a} ({i+1} out of {a_vals.size})')
        x_vals = np.arange(offset, 2*np.pi*a + offset, dx)
        y_vals = np.copy(x_vals)
        print(f'Nx = {x_vals.size}')
        q_approx = AV.square_approximant(a=a, G_vects=q_exact)
        B = calc_curv_surface(x_vals, y_vals, save=False, q_vals=q_approx, N=N, a=a, **kwargs)
        flux_vals[i] = np.sum(B)
        print('')
    np.savez(filename, a_vals=a_vals, flux_vals=flux_vals, dx=dx, N=N, **kwargs)


if __name__ == '__main__':
    N = 6
    l = np.arange(N)
    a = 3
    q_exact = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    q_mag = np.linalg.norm(q_exact[0,:])
    q_approx = AV.square_approximant(a=a, G_vects=q_exact)
    p1 = 2
    p2 = -1
    phi1 = 0
    phi2 = 0
    phi = np.zeros((N, 2))
    for l in range(N):
        phi[l,0] = 2 * np.pi * l * p1 / N + phi1
        phi[l,1] = 2 * np.pi * l * p2 / N + phi2

    f_flux = 'Data/B_eff/Phi_uc_R5_a1-20_dx0.05_offset_p10.5_p20.5_phi10_phi20.npz'
    # calc_approximant_fluxes(a_vals=np.arange(1,21), filename=f_flux, N=5,
    #                         p1=p1, p2=p2, phi1=phi1, phi2=phi2, phi=phi, offset=0.01)

    Nx = 300
    # x_vals = np.linspace(0, 2*np.pi*a, Nx)
    # y_vals = np.copy(x_vals)
    # x_vals = np.linspace(9,10,100)
    # y_vals = np.linspace(2.25,3,100)
    dx = 0.05
    x_vals = np.arange(0.001, 2*2*np.pi+0.001, dx)
    y_vals = np.copy(x_vals)
    # x_vals = np.arange(0.01, 2*np.pi*a + 0.01, dx)
    # y_vals = np.copy(x_vals)

    # f_B = 'DarkState/Data/B_eff/B_eff_uc_R5_a3_dx0.05_p11_p20_phi10_phi20.npz'
    # f_B = 'DarkState/Data/B_eff/B_eff_uc_R5_a3_dx0.05_offset0.01_p10.5_p20.5_phi10_phi20.npz'
    f_B = 'DarkState/Data/B_eff/B_eff_uc_R6_dx0.05_offset0.001_p11_p20_phi10_phi20.npz'
    calc_curv_surface(x_vals=x_vals, y_vals=y_vals, save=True, filename=f_B,
                      q_vals=q_exact, p1=p1, p2=p2, phi1=phi1, phi2=phi2, phi=phi,
                      N=N, dx=dx)
    
    # f_d = 'DarkState/Data/B_eff/d_uc_R5_a3_dx0.05_p11_p20_phi10_phi20.npz'
    f_d = 'DarkState/Data/B_eff/d_uc_R6_dx0.05_offset0.001_p11_p20_phi10_phi20.npz'
    calc_d_surface(x_vals, y_vals, save=True, filename=f_d, q_vals=q_exact, phi=phi, 
                       N=N, p1=p1, p2=p2, phi1=phi1, phi2=phi2, dx=dx)


    # dx = x_vals[1] - x_vals[0]
    # kx = np.fft.fftfreq(len(x_vals), dx)
    x_vals = calc_req_x(kmax=12, dk=(1/a)/4)
    # print(x_vals.size, x_vals[1] - x_vals[0])
    kx = np.fft.fftfreq(x_vals.size, x_vals[1] - x_vals[0]) * 2 * np.pi
    # print(kx[:5])
    # print(kx.size)
    # print(np.max(kx), np.min(kx), kx[1]-kx[0])
    y_vals = np.copy(x_vals)
    # x_vals = np.linspace(8, 11, 100)
    # y_vals = np.linspace(14, 17, 100)
    f_U = 'DarkState/Data/B_eff/U_aligned_extended_R5_a3_p11_p20_phi10_phi20.npz'
    # calc_U_exact_surface(x_vals, y_vals, save=True, filename=f_U, q_vals=q_approx, phi=phi, 
    #                   N=N, p1=p1, p2=p2, a=a, phi1=phi1, phi2=phi2)
    # calc_U_FT(f_U)

    f_d = 'DarkState/Data/B_eff/d_aligned_extended_R5_a3_p11_p20_phi10_phi20.npz'
    # calc_d_surface(x_vals, y_vals, save=True, filename=f_d, q_vals=q_approx, phi=phi, 
    #                    N=N, p1=p1, p2=p2, a=a, phi1=phi1, phi2=phi2)
    
    f_dU = 'DarkState/Data/B_eff/dU_extended_R5_a3_p11_p20_phi10_phi20.npz'
    # calc_dU(f_U, save_filename=f_dU)

    M = 4000
    f_dV = 'DarkState/Data/B_eff/dV_extended_R5_a3_M4000_p11_p20_phi10_phi20.npz'
    # calc_dV(f_dU, f_d, M=M, save_filename=f_dV, save_params=['x_vals', 'y_vals', 'kx_vals', 'ky_vals', 'a', 
    #                              'N', 'p1', 'p2', 'phi1', 'phi2'])
    
    f_U_compensated = 'DarkState/Data/B_eff/U_compensated_extended_R5_a3_M4000_p11_p20_phi10_phi20.npz'
    # calc_compensated_U(f_dV, f_d, f_U, save_filename=f_U_compensated)

    # DA.calc_dU(f, M=M, save=True, save_filename=f2, 
    #            save_params={'N':5, 'p1':1, 'p2':0, 'phi1':0, 'phi2':0,
    #                         'a':3})
    # data = np.load(sf)
    # k = np.column_stack((data['kx_vals'], data['ky_vals']))
    # dU = data['U_FT']
    # for i in range(M):
    #     print(f'k = {np.round(k[i,:],3)}: dU(k) = {np.round(dU[i],3)}')
    sf = 'DarkState/Data/B_eff/U_compensated_R5_a3_M200_p11_p20_phi10_phi20.npz'
    # calc_compensated_U(f1=f, f2=f2, save=True, save_filename=sf)

    # f = 'DarkState/Data/B_eff/U_approx_R5_a5_p11_p20_phi10_phi20.npz'
    # calc_U_approx_surface(x_vals, y_vals, save=True, filename=f, q_exact=q_exact, q_approx=q_approx, q_vals=q_approx, phi=phi, 
    #                   N=N, p1=p1, p2=p2, a=a)
    # f = 'DarkState/Data/B_eff/sv_R5_a5_p11_p20_phi10_phi20.npz'
    # calc_sv_surface(x_vals, y_vals, save=True, filename=f, q_exact=q_exact, q_approx=q_approx, q_vals=q_approx, phi=phi, 
    #                   N=N, p1=p1, p2=p2, a=a)
    # f = 'DarkState/Data/B_eff/dmag_R5_a5_p11_p20_phi10_phi20.npz'
    # calc_d_surface(x_vals, y_vals, save=True, filename=f, q_vals=q_approx, phi=phi, 
    #                   N=N, p1=p1, p2=p2, a=a)
    # calc_d_FT(f)


    # phi = np.random.rand(8,2) * 2*np.pi
    # r = np.random.rand(2)
    # r = np.array([10,9.6])
    # d = d_state(r, q_vals=q_approx, phi=phi)
    # s = s_state(r, q_vals=q_approx, phi=phi)
    # v = v_state(r, q_approx=q_approx, q_exact=q_exact, phi=phi)
    # del_p_d = del_p_d_state(r, q_vals=q_approx, phi=phi)
    # U_exact = calc_U_exact(x=r[0], y=r[1], q_vals=q_approx, phi=phi)
    # U_approx = q_mag**2 + 2*q_mag*np.real(np.vdot(s,v)) / np.linalg.norm(d)**2
    # U_ana = U_approx + np.abs(np.vdot(v,s))**2 / np.linalg.norm(s)**4
    # print(f's = {np.round(s, 2)}')
    # print(f'd = {np.round(d, 2)}')
    # print(f'del_p_d = {np.round(del_p_d, 2)}')
    # print(f'-i|q||s> = {np.round(-1j*q_mag*s, 2)}')
    # print(f'-i|q||s> - i|v> = {np.round(-1j*q_mag*s + -1j*v, 2)}')
    # print(f'<d|s> = {np.round(np.vdot(d, s), 2)}')
    # print(f'<d|del_p_d> = {np.round(np.vdot(d, del_p_d), 2)}')
    # print(f'Exact U = {np.round(U_exact, 5)}')
    # print(f'Approximate U = {np.round(U_approx, 5)}')
    # print(f'Analytical U = {np.round(U_ana, 5)}')

    # del_m_d = del_m_d_state(r, q_vals=q, phi=phi)
    # del_x_d = del_x_d_state(r, q_vals=q, phi=phi)
    # del_y_d = del_y_d_state(r, q_vals=q, phi=phi)
    # diff = -1j * q_mag * s - del_p_d
    # dd = np.vdot(d,d)



