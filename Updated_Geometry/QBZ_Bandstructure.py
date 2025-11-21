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


def calc_H(q, U=np.ones(8), G=G, V=0., Dx=0., Dy=0., Dz=0., W=0.):
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
    # W couplings:
    orb = np.zeros((8,8), dtype=np.complex128)
    # for i in range(4):
    #     orb[(i+2)%8, i] += 2*W*1j
    #     orb[(i+6)%8, i+4] += -2*W*1j
    for i in range(3,7):
        orb[(i+2)%8, i] += 2*W*1j
        orb[(i+6)%8, (i+4)%8] += -2*W*1j
    dHW = np.kron(orb, I2)
    H += (dHW + dHW.conj().T)
    # D couplings:
    H += Dx * np.kron(np.eye(8), sx)
    H += Dy * np.kron(np.eye(8), sy)
    H += Dz * np.kron(np.eye(8), sz)
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


def plot_evals_surf(evals, qx_vals, qy_vals, U_mag=0.15, V_mag=0.01, n_bands=np.arange(16), 
                    Dx=0., Dy=0., Dz=0., W=0.):
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    fig,ax = plt.subplots(subplot_kw={'projection':'3d'})
    for n in n_bands:
        ax.plot_surface(qxx, qyy, evals[n,:,:])
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$E$ / $E_{R}$', rotation=0, labelpad=10)
    ax.set_title(f'U = {U_mag}, V = {V_mag}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}, W = {W}')
    plt.show()


def plot_curv(curv_vals, qx_vals, qy_vals, shift_q=True, U_mag=0.15, V_mag=0.01, 
              n_bands=np.arange(16), bands_in_title=True, chern_in_title=True,
              NonAb=True, N=N, Dx=0., Dy=0., Dz=0., W=0.,
              title_params={'R':r'$R$', 'U0':r'$U$', 'V0':r'$V$', 'N':r'$N$'},
              dp=3):
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
    title_str = ''
    for k,v in title_params.items():
        title_str += ', ' + k + r'$=$' + str(np.round(v, dp))
    # title_str = f'U = {U_mag}, V = {V_mag}, N = {N}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}, W = {W}'
    title_str = title_str[2:]
    if bands_in_title:
        if len(n_bands) <= 2:
            band_tit = f'Bands = {n_bands}'
        else:
            band_tit = f'Bands = [{n_bands[0]}-{n_bands[-1]}]'
        title_str += ', ' + band_tit
    if chern_in_title:
        C = np.sum(curv_plot) / (2*np.pi)
        C_str = str(np.round(np.real(C), dp))
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
    return np.vstack((qGM, qMK, qKG))


def calc_BS_path(q_vals, **kwargs):
    evals = np.zeros((q_vals.shape[0],16))
    for i in range(q_vals.shape[0]):
        print(f'Evaluating q point {i+1} out of {q_vals.shape[0]}...' + 10*' ', end='\r')
        q = q_vals[i,:]
        H = calc_H(q, **kwargs)
        evals[i,:] = np.linalg.eigvalsh(H)
    return evals


def plot_BS_GMKG(evals, U_mag=None, V_mag=0., alternating=True, N=5, Dx=0., Dy=0., Dz=0., W=0.):
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
        ax.set_title(f'U = {U_mag}, V = {V_mag}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}, N = {N}, W = {W}')
    plt.show()


def plot_BS_path(evals, q_vals, x='qx', U_mag=None, V_mag=0., alternating=True, N=5, Dx=0.,
                 Dy=0., Dz=0., Elim=None, W=0.):
    fig,ax = plt.subplots()
    if x == 'qx':
        x_vals = q_vals[:,0]
        xlab = r'$q_x$'
    elif x == 'qy':
        x_vals = q_vals[:,1]
        xlab = r'$q_y$'
    else:
        x_vals = np.arange(q_vals.shape[0])
        xlab = 'N'
    for i in range(16):
        if alternating:
            styles = ['-', ':']
            colours = ['k', 'r']
            ax.plot(x_vals, evals[:,i], ls=styles[i%2], c=colours[i%2])
    # ax.set_xticks([0, 99, 198, 297])
    # ax.set_xticklabels([r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'$E/E_R$')
    if U_mag is not None:
        ax.set_title(f'U = {U_mag}, V = {V_mag}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}, N = {N}, W = {W}')
    if Elim is not None:
        ax.set_ylim(*Elim)
    plt.show()


def calc_dE_vs_N(N_vals, U_mag=0.05, V_mag=0.001, R=8, phi0=0, 
                 progress=True, **kwargs):
    dE_vals = np.zeros(N_vals.shape)
    l = np.arange(R)
    G = np.column_stack((np.cos(2*np.pi*l/R), np.sin(2*np.pi*l/R)))
    q_K = np.array([0.5, 0.5 * np.tan(np.pi/R)])
    for i, N in enumerate(N_vals):
        if progress:
            print(f'Evaluating N value {i+1} out of {N_vals.size}...' + 10*' ', end='\r')
        U = -U_mag * np.exp(1j * (phi0 - 2 * np.pi * N * l / R))
        H = calc_H(q=q_K, U=U, G=G, V=V_mag)
        evals, evects = np.linalg.eigh(H)
        dE_vals[i] = evals[6] - evals[5]
    return dE_vals


def plot_dE_vs_N(N_vals, dE_vals, U_mag=0.05, V_mag=0.001):
    fig, ax = plt.subplots()
    ax.plot(N_vals, dE_vals, color='b', ls='-', marker='o', ms=2)
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$\Delta E$', rotation=0)
    ax.set_title(f'U = {np.round(U_mag,4)}, V = {np.round(V_mag,4)}, V/U = {np.round(V_mag/U_mag, 4)}')
    plt.show()


def calc_dN_vs_r(r_vals, Nerr=0.01, U_mag=0.05, 
                 save=True, filename='Data.npz', **kwargs):
    N_crit = []
    N_vals = np.arange(0, R + Nerr, Nerr)
    for i, r in enumerate(r_vals):
        print(f'Evaluating r value {i+1} out of {r_vals.size}...' + 10*' ', end='\r')
        dE = calc_dE_vs_N(N_vals=N_vals, U_mag=U_mag, V_mag=r*U_mag,
                          R=R, progress=False, **kwargs)
        N_found = []
        for j in range(N_vals.size):
            if j == 0:
                if dE[j] < dE[j+1]:
                    N_found.append(N_vals[j])
            elif j == N_vals.size-1:
                if dE[j] < dE[j-1]:
                    N_found.append(N_vals[j])
            else:
                if dE[j] < dE[j+1] and dE[j] < dE[j-1]:
                    N_found.append(N_vals[j])
        N_crit.append(N_found)

        # for j in range(R):
        #     mask = (N_vals > j) & (N_vals < j+1)
        #     idx = np.argmin(dE[mask])
        #     N_crit[i,j] = N_vals[mask][idx]
    if save:
        max_N = max([len(N) for N in N_crit])
        N_save = np.full((r_vals.size, max_N), np.nan)
        for i,N in enumerate(N_crit):
            N_save[i,:len(N)] = N
        np.savez(filename, r_vals=r_vals, N_crit=N_save, Nerr=Nerr, U_mag=U_mag,
                 **kwargs)
    else:
        return N_crit
    

def plot_dN_vs_r(filename):
    data = np.load(filename)
    r_vals = data['r_vals']
    N_crit = data['N_crit']
    Nerr = data['Nerr']

    print(N_crit)
    print(r_vals)

    fig, ax = plt.subplots()
    # for i in range(N_crit.shape[1]):
    #     ax.errorbar(N_crit[:,i], r, xerr=Nerr, color='b', marker='x')
    for i, r in enumerate(r_vals):
        ax.errorbar(N_crit[i], r*np.ones(len(N_crit[i])), color='b', marker='x', ls='')

    
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$V/U$')
    ax.set_title(f'U = {data['U_mag']}')

    ax.set_ylim(bottom=0)

    plt.show()


def calc_r_max_vs_N(N_vals, dr=0.05, U_mag=0.05, R=8, phi0=0,
                    save=True, filename='Data.npz', **kwargs):
    r_vals = np.zeros(N_vals.shape)
    l = np.arange(R)
    G = np.column_stack((np.cos(2*np.pi*l/R), np.sin(2*np.pi*l/R)))
    q_G = np.zeros(2)
    q_M = np.array([0.5, 0])
    q_K = np.array([0.5, 0.5 * np.tan(np.pi/R)])
    for i, N in enumerate(N_vals):
        print(f'Evaluating N value {i+1} out of {N_vals.size}...' + 10*' ', end='\r')
        U = -U_mag * np.exp(1j * (phi0 - 2 * np.pi * N * l / R))
        r = 0
        while True:
            V_mag = U_mag * r
            H_G = calc_H(q=q_G, U=U, G=G, V=V_mag)
            H_M = calc_H(q=q_M, U=U, G=G, V=V_mag)
            H_K = calc_H(q=q_K, U=U, G=G, V=V_mag)
            evals_G, evects_G = np.linalg.eigh(H_G)
            evals_M, evects_M = np.linalg.eigh(H_M)
            evals_K, evects_K = np.linalg.eigh(H_K)
            evals_5 = np.array([evals_G[5], evals_M[5], evals_K[5]])
            evals_6 = np.array([evals_G[6], evals_M[6], evals_K[6]])
            dE = np.min(evals_6) - np.max(evals_5)
            if dE < 0:
                r_vals[i] = r
                break
            else:
                r += dr
    if save:
        np.savez(filename, N_vals=N_vals, r_vals=r_vals, U_mag=U_mag, dr=dr,
                 R=R, phi0=phi0, **kwargs)
    else:
        return r_vals
    

def plot_r_max_vs_N(filenames, colors=['b','r','skyblue','limegreen','gold','navy']):
    fig,ax = plt.subplots()
    for i,f in enumerate(filenames):
        data = np.load(f)
        N_vals = data['N_vals']
        r_vals = data['r_vals']
        dr = data['dr']
        U = data['U_mag']
        ax.errorbar(N_vals, r_vals, yerr=dr/2, color=colors[i], marker='x', ls='', label=U)
    ax.legend(title=r'$U$')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$r_{max}$')
    ax.set_title(r'Critical $r_{max} = V/U$')
    plt.show()
    




if __name__ == '__main__':
    ### Constants, parameters etc.
    U_mag = 0.01
    r = 0.6
    V_mag = r * U_mag
    phi0 = 0
    R = 8
    N_vals = np.arange(0, 8.01, 0.01)
    
    # dE_vals = calc_dE_vs_N(N_vals=N_vals, U_mag=U_mag, V_mag=V_mag, 
    #                        R=R, phi0=phi0)
    
    # plot_dE_vs_N(N_vals, dE_vals, U_mag=U_mag, V_mag=V_mag)

    r_vals = np.arange(0.02, 1.02, 0.02)
    Nerr = 0.01
    f = 'Updated Geometry/Data/Ncrit_R8_U0.05.npz'
    # calc_dN_vs_r(r_vals=r_vals, Nerr=Nerr, U_mag=U_mag, save=True,
    #              filename=f)
    # plot_dN_vs_r(f)

    N_vals = np.arange(0, 8.01, 0.05)
    f1 = 'Updated Geometry/Data/rmax_R8_U0.05.npz'
    f2 = 'Updated Geometry/Data/rmax_R8_U0.025.npz'
    f3 = 'Updated Geometry/Data/rmax_R8_U0.01.npz'
    f4 = 'Updated Geometry/Data/rmax_R8_U0.075.npz'
    # calc_r_max_vs_N(N_vals=N_vals, dr=0.05, U_mag=0.075, save=True, 
    #                 filename=f4)
    # plot_r_max_vs_N(filenames=[f3, f2, f1, f4])


    ### Constants, parameters etc.
    U_mag = 0.05
    l = np.arange(8)
    phi0 = 0
    N = 5
    U = -U_mag * np.exp(1j * (phi0 - 2 * np.pi * N * l / 8))
    V_mag = 0.0
    Dx = 0.0
    Dy = 0.0
    Dz = 0.0
    W = 0.001

    # q_vals = calc_q_GMKG()
    # evals = calc_BS_path(q_vals, U=U, V=V_mag, Dx=Dx, Dy=Dy, Dz=Dz, W=W)
    # plot_BS_GMKG(evals=evals, U_mag=U_mag, V_mag=V_mag, N=N, Dx=Dx, Dy=Dy, Dz=Dz, W=W)

    q_K = np.array([0.5, 0.5*np.tan(np.pi/8)])
    # dq = np.column_stack((np.linspace(-0.01, 0.01, 1000), np.zeros(1000)))
    # dq = np.column_stack((np.zeros(1000), np.linspace(-0.01, 0.01, 1000)))
    dq = np.column_stack((np.linspace(-0.01, 0.01, 1000), np.linspace(-0.01, 0.01, 1000)))
    q_vals = q_K + dq
    # e = q_K / np.linalg.norm(q_K)
    # q_vals = np.column_stack((e[0] * np.linspace(-0.01, 0.01, 1000), e[1] * np.linspace(-0.01, 0.01, 1000))) + q_K
    evals = calc_BS_path(q_vals, U=U, V=V_mag, Dx=Dx, Dy=Dy, Dz=Dz, W=W)
    plot_BS_path(evals=evals, q_vals=q_vals, x='qx', U_mag=U_mag, V_mag=V_mag, N=N, Dx=Dx, Dy=Dy, Dz=Dz, W=W,
                 Elim=(0.24,0.27))

    dq = 0.01*np.linspace(-1,1,50)
    qx_vals = dq + 0.5
    qy_vals = dq + 0.5*np.tan(np.pi/8)

    # qx_vals = np.linspace(0.5032, 0.5036, 200)
    # qy_vals = np.linspace(0.2081, 0.2085, 200)

    evals, evects = calc_evects(qx_vals, qy_vals, U=U, V=V_mag, Dx=Dx, Dy=Dy, Dz=Dz, W=W)
    # plot_evals_surf(evals, qx_vals, qy_vals, U_mag=U_mag, V_mag=V_mag, Dx=Dx, Dy=Dy, Dz=Dz)
    n_bands = np.arange(6)
    curv = calc_curv(evects, n_bands=n_bands, NonAb=True)
    plot_curv(curv, qx_vals, qy_vals, U_mag=U_mag, V_mag=V_mag, n_bands=n_bands, N=N, Dx=Dx, Dy=Dy, Dz=Dz, W=W,
              title_params={'U':U_mag, 'N':N, 'V':V_mag, 'Dx':Dx, 'Dy':Dy, 'W':W},
              bands_in_title=False, dp=5)
    

