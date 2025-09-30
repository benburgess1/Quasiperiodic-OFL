import numpy as np
import matplotlib.pyplot as plt

### Useful matrices
sx = np.array([[0, 1],
               [1, 0]])
sy = np.array([[0, -1j],
               [1j, 0]])
sz = np.array([[1, 0],
               [0, -1]])
sp = np.array([[0, 1],
               [0, 0]])
sm = np.array([[0, 0],
               [1, 0]])
I2 = np.eye(2)


### Constants, matrix elements, etc.
Umag = 0.1
l = np.arange(8)
phi0 = 0
N = 5
U = -Umag * np.exp(1j * (phi0 - 2 * np.pi * N * l / 8))
dphi = np.angle(U[1]) - np.angle(U[0])
w = np.exp(1j*dphi)
G = np.column_stack((np.cos(2*np.pi*l/8), np.sin(2*np.pi*l/8)))


def calc_H(q, U=np.ones(8), G=G, V=0.):
    H = np.zeros((16, 16), dtype=np.complex128)
    # U couplings:
    for i in range(8):
        orb = np.zeros((8,8), dtype=np.complex128)
        orb[(i+3)%8, i] = U[i]
        orb[i, (i+3)%8] = U[(i+4)%8]
        dHU = np.kron(orb, sp)
        H += (dHU + dHU.conj().T)
    # V couplings:
    orb = np.zeros((8,8), dtype=np.complex128)
    for i in range(8):
        orb[(i+2)%8, i] += 2*V
        # orb[i, (i+2)%8] += V
    dHV = np.kron(orb, sz)
    H += (dHV + dHV.conj().T)
    # KE:
    H_kin = np.zeros((8, 8), dtype=np.complex128)
    b = {0:np.zeros(2), 3:G[0,:], 6:G[0,:]+G[3,:], 1:G[0,:]+G[3,:]+G[6,:], 
         4:G[0,:]+G[3,:]+G[6,:]+G[1,:], 7:G[3,:]+G[6,:]+G[1,:], 2:G[6,:]+G[1,:], 
         5:G[1,:]}
    for i, v in b.items():
        H_kin[i,i] = np.sum((q-v)**2)
    H += np.kron(H_kin, I2)
    return H


def calc_evals(qx_vals, qy_vals, **kwargs):
    evals = np.zeros((16, qx_vals.size, qy_vals.size))
    for i, qx in enumerate(qx_vals):
        for j, qy in enumerate(qy_vals):
            print(f'Evaluating q value {i*qy_vals.size + j + 1} out of {qx_vals.size * qy_vals.size}...' + 10*' ', end='\r')
            H = calc_H(q=np.array([qx,qy]), **kwargs)
            evals[:, i, j] = np.linalg.eigvalsh(H)
    return evals


def calc_evects(qx_vals, qy_vals, **kwargs):
    print('Calculating bandstructure:')
    evals = np.zeros((16, qx_vals.size, qy_vals.size))
    evects = np.zeros((16, 16, qx_vals.size, qy_vals.size), dtype=np.complex128)
    for i, qx in enumerate(qx_vals):
        for j, qy in enumerate(qy_vals):
            print(f'Evaluating q value {i*qy_vals.size + j + 1} out of {qx_vals.size * qy_vals.size}...' + 10*' ', end='\r')
            H = calc_H(q=np.array([qx,qy]), **kwargs)
            evals[:, i, j], evects[:, :, i, j] = np.linalg.eigh(H)
    print('\nDone')
    return evals, evects


def calc_curv(evects, fix_gauge=True, n_bands=np.arange(16), NonAb=False, **kwargs):
    print('Calculating curvature:')
    (Nx, Ny) = evects.shape[-2:]
    if NonAb:
        curv_vals = np.zeros((Nx-1, Ny-1))
    else:
        curv_vals = np.zeros((len(n_bands), Nx-1, Ny-1))
    if fix_gauge:
        print('Fixing gauge...', end=' ')
        for i in range(Nx-1):
            for j in range(Ny-1):
                for n in n_bands:
                    u = evects[:,n,i,j]
                    u *= np.exp(-1j * np.angle(u[0]))
                    u /= np.linalg.norm(u)
                    evects[:,n,i,j] = u
        print('Done')
    for i in range(Nx-1):
        for j in range(Ny-1):
            print(f'Evaluating q point {i*(Ny-1) + j + 1} out of {(Nx-1) * (Ny-1)}...' + 10*' ', end='\r')
            if NonAb:
                u1 = evects[:,n_bands,i,j]
                u2 = evects[:,n_bands,i+1,j]
                u3 = evects[:,n_bands,i+1,j+1]
                u4 = evects[:,n_bands,i,j+1]
                A1 = np.conj(u1).T @ u2
                U1 = np.linalg.det(A1)
                A2 = np.conj(u2).T @ u3
                U2 = np.linalg.det(A2)
                A3 = np.conj(u3).T @ u4
                U3 = np.linalg.det(A3)
                A4 = np.conj(u4).T @ u1
                U4 = np.linalg.det(A4)
                curv = -(np.angle(U1) + np.angle(U2) + np.angle(U3) + np.angle(U4))
                if np.abs(curv) > np.pi:
                    # print('\n\nWarning: curvature outside (-pi,pi] range. Manually adjusting.')
                    curv = (curv + np.pi) % (2*np.pi) - np.pi
                elif curv == -np.pi:
                    curv = np.pi
                curv_vals[i,j] = curv
            else:
                for n in n_bands:
                    u1 = evects[:,n,i,j]
                    u2 = evects[:,n,i+1,j]
                    u3 = evects[:,n,i+1,j+1]
                    u4 = evects[:,n,i,j+1]
                    U1 = np.vdot(u1, u2)
                    U2 = np.vdot(u2, u3)
                    U3 = np.vdot(u3, u4)
                    U4 = np.vdot(u4, u1)
                    curv = -(np.angle(U1) + np.angle(U2) + np.angle(U3) + np.angle(U4))
                    if np.abs(curv) > np.pi:
                        # print('\n\nWarning: curvature outside (-pi,pi] range. Manually adjusting.')
                        curv = (curv + np.pi) % (2*np.pi) - np.pi
                    elif curv == -np.pi:
                        curv = np.pi
                    curv_vals[n,i,j] = curv
    print('\nDone')
    return curv_vals


def plot_evals_surf(evals, qx_vals, qy_vals, U_mag=0.15, V_mag=0.01, n_bands=np.arange(16)):
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    fig,ax = plt.subplots(subplot_kw={'projection':'3d'})
    for n in n_bands:
        ax.plot_surface(qxx, qyy, evals[n,:,:])
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$E$ / $E_{R}$', rotation=0, labelpad=10)
    ax.set_title(f'U = {U_mag}, V = {V_mag}')
    plt.show()


def plot_curv(curv_vals, qx_vals, qy_vals, shift_q=True, U_mag=0.15, V_mag=0.01, 
              n_bands=np.arange(16), bands_in_title=True, chern_in_title=True,
              NonAb=True):
    if shift_q:
        dqx = qx_vals[1] - qx_vals[0]
        qx_vals = (qx_vals + dqx/2)[:-1]
        dqy = qy_vals[1] - qy_vals[0]
        qy_vals = (qy_vals + dqy/2)[:-1]
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    if NonAb:
        curv_plot = curv_vals
    else:
        curv_plot = np.sum(curv_vals[n_bands,:,:], axis=0)
    fig, ax = plt.subplots()
    vmax = np.max(np.abs(curv_plot))
    cmap = plt.colormaps['bwr']
    levels = np.linspace(-vmax,vmax,200)
    ticks = [-vmax,0,vmax]
    plot = ax.contourf(qxx, qyy, curv_plot, cmap=cmap, levels=levels)
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$', rotation=0)
    title_str = f'U = {U_mag}, V = {V_mag}'
    if bands_in_title:
        if len(n_bands) <= 2:
            band_tit = f'Bands = {n_bands}'
        else:
            band_tit = f'Bands = [{n_bands[0]}-{n_bands[-1]}]'
        title_str += ', ' + band_tit
    if chern_in_title:
        C = np.sum(curv_plot) / (2*np.pi)
        C_str = str(np.round(np.real(C), 5))
        title_str += ', ' + r'$C = $' + C_str
    ax.set_title(title_str)
    cbar = fig.colorbar(plot, ticks=ticks)
    cbar.ax.set_ylabel(r'$\Omega$', rotation=0)
    ax.set_xlim(np.min(qx_vals), np.max(qx_vals))
    ax.set_ylim(np.min(qy_vals), np.max(qy_vals))
    plt.show()


def calc_q_GMKG():
    qG = np.zeros(2)
    qM = np.array([0.5, 0])
    qK = np.array([0.5, 0.5*np.tan(np.pi/8)])
    qGM = np.column_stack((np.linspace(qG[0], qM[0], 100),
                           np.linspace(qG[1], qM[1], 100)))
    qMK = np.column_stack((np.linspace(qM[0], qK[0], 100)[1:],
                           np.linspace(qM[1], qK[1], 100)[1:]))
    qKG = np.column_stack((np.linspace(qK[0], qG[0], 100)[1:],
                           np.linspace(qK[1], qG[1], 100)[1:]))
    return np.row_stack((qGM, qMK, qKG))


def calc_BS_path(q_vals, **kwargs):
    evals = np.zeros((q_vals.shape[0],16))
    for i in range(q_vals.shape[0]):
        print(f'Evaluating q point {i+1} out of {q_vals.shape[0]}...' + 10*' ', end='\r')
        q = q_vals[i,:]
        H = calc_H(q, **kwargs)
        evals[i,:] = np.linalg.eigvalsh(H)
    return evals


def plot_BS_GMKG(evals, U_mag=None, V_mag=0., alternating=True):
    fig,ax = plt.subplots()
    x = np.arange(298)
    for i in range(16):
        if alternating:
            styles = ['-', ':']
            colours = ['k', 'r']
            ax.plot(x, evals[:,i], ls=styles[i%2], c=colours[i%2])
    ax.set_xticks([0, 99, 198, 297])
    ax.set_xticklabels([r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])
    ax.set_ylabel(r'$E/E_R$')
    if U_mag is not None:
        ax.set_title(f'U = {U_mag}, V = {V_mag}')
    plt.show()


if __name__ == '__main__':
    ### Constants, parameters etc.
    U_mag = 0.05
    l = np.arange(8)
    phi0 = 0
    N = 5
    U = -U_mag * np.exp(1j * (phi0 - 2 * np.pi * N * l / 8))
    V_mag = 0.0005

    # q_vals = calc_q_GMKG()
    # evals = calc_BS_path(q_vals, U=U, V=V_mag)
    # plot_BS_GMKG(evals=evals, U_mag=U_mag, V_mag=V_mag)

    dq = 0.1*np.linspace(-1,1,50)
    qx_vals = dq + 0.5
    qy_vals = dq + 0.5*np.tan(np.pi/8)

    evals, evects = calc_evects(qx_vals, qy_vals, U=U, V=V_mag)
    plot_evals_surf(evals, qx_vals, qy_vals, U_mag=U_mag, V_mag=V_mag)
    n_bands = np.arange(6)
    curv = calc_curv(evects, n_bands=n_bands, NonAb=True)
    plot_curv(curv, qx_vals, qy_vals, U_mag=U_mag, V_mag=V_mag, n_bands=n_bands)
    

