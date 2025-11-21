import numpy as np
import matplotlib.pyplot as plt
import scipy

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
q_K = np.array([0.5, 0.5*np.tan(np.pi/8)])


def calc_H(q=np.zeros(2), U=np.ones(8), G=G, V=0., Dx=0., Dy=0., Dz=0., **kwargs):
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


def calc_K_eigenstates(q=np.array([0.5, 0.5*np.tan(np.pi/8)]), **kwargs):
    H = calc_H(q=q, **kwargs)
    evals, evects = np.linalg.eigh(H)
    psi = evects[:,4:8]
    return psi


def visualise_K_eigenstates(psi, title_str=None,
                            title_params={'U_mag':'U', 'N':'N', 'V':'V', 
                                          'Dx':'Dx', 'Dy':'Dy', 'Dz':'Dz'},
                                                 **kwargs):
    fig, axs = plt.subplots(1,3)
    # box0 = axs[0].get_position()
    # print(box0)
    # box0.x0 -= 0.15
    # box0.x1 -= 0.15
    # print(box0)
    # axs[0].set_position(box0)
    # box1 = axs[1].get_position()
    # box1.x0 -= 0.05
    # box1.x1 -= 0.05
    # axs[1].set_position(box1)
    im1 = axs[0].imshow(np.abs(psi), cmap='hot', origin='upper')
    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar1.set_label(r'$|\psi|$')
    im2 = axs[1].imshow(np.real(psi), cmap='bwr', origin='upper')
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar2.set_label(r'$Re(\psi)$')
    im3 = axs[2].imshow(np.imag(psi), cmap='bwr', origin='upper')
    cbar3 = fig.colorbar(im3, ax=axs[2])
    cbar3.set_label(r'$Im(\psi)$')
    if title_str is None:
        title_str = ''
        for k, v in title_params.items():
            if k in kwargs:
                title_str += v + '=' + str(np.round(np.real(kwargs[k]),5)) + ', '
        title_str = title_str[:-2]
    fig.suptitle(title_str, y=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def project_M(M, psi):
    return psi.conj().T @ M @ psi


def calc_c8(N=5):
    dphi = N * np.pi / 4
    c8_orb = np.zeros((8, 8), dtype=np.complex128)
    for i in range(8):
        c8_orb[(i+1)%8, i] = 1
    c8_spin = np.array([[np.exp(1j*dphi/2), 0],
                        [0, np.exp(-1j*dphi/2)]])
    U_c8 = np.kron(c8_orb, c8_spin)
    return U_c8


def calc_Pz(N=5, factor=1.):
    U_c8 = calc_c8(N=N)
    Pz = np.linalg.matrix_power(U_c8, 4) * factor
    return Pz


def calc_sd():
    sd_orb = np.zeros((8,8), dtype=np.complex128)
    for i in range(8):
        sd_orb[7-i, i] = 1
    U_sd = np.kron(sd_orb, sy)
    return U_sd

def calc_sd2():
    sd_orb = np.zeros((8,8), dtype=np.complex128)
    for i in range(8):
        sd_orb[(3-i)%8, i] = 1
    U_sd = np.kron(sd_orb, sx)
    return U_sd


def calc_U_Tx():
    U_orb = np.zeros((8, 8), dtype=np.complex128)
    for i in range(8):
        U_orb[(i+4)%8, i] = 1
    U_Tx = np.kron(U_orb, sx)
    return U_Tx

def left_Tx(v, U_Tx=calc_U_Tx()):
    return U_Tx @ v.conj()

def inner_Tx(left_v, right_v, U_Tx=calc_U_Tx()):
    return left_v.conj().T @ left_Tx(right_v, U_Tx=U_Tx)


def calc_S1(psi, U_Pz=calc_Pz(), U_Tx=calc_U_Tx()):
    return psi.conj().T @ U_Pz @ left_Tx(psi, U_Tx=U_Tx)

def calc_Dx(D=1.):
    return D*np.kron(np.eye(8), sx)

def calc_Dy(D=1.):
    return D*np.kron(np.eye(8), sy)

def calc_HV(V=1.):
    orb = np.zeros((8,8), dtype=np.complex128)
    for i in range(8):
        orb[(i+2)%8, i] += 2*V
        # orb[i, (i+2)%8] += V
    dHV = np.kron(orb, sz)
    visualise_matrix(dHV)
    return (dHV + dHV.conj().T)


def visualise_matrix(M, title_str=''):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    
    for ax, data, cmap, label in zip(
        axs,
        [np.abs(M), np.real(M), np.imag(M)],
        ['hot', 'bwr', 'bwr'],
        [r'$|M|$', r'$Re(M)$', r'$Im(M)$']
    ):
        im = ax.imshow(data, cmap=cmap, origin='upper')
        # Add a colorbar that matches the axis height
        cax = ax.inset_axes([1.02, 0.0, 0.1, 1.])  # [x0, y0, width, height]
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(label)

    fig.suptitle(title_str, y=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def investigate_H_U(N=5, dq=np.zeros(2), subtract_Eav=False, block_diagonalise=False, **kwargs):
    q_K = np.array([0.5, 0.5*np.tan(np.pi/8)])
    H = calc_H(q=q_K + dq, N=N, **kwargs)
    evals, evects = np.linalg.eigh(H)
    psi_K = evects[:, 4:8]
    print(np.round(evals[4:8],5))
    Eav = np.mean(evals[4:8])

    # Behaviour for 'random' numerical eigenstates of H
    # U_c8 = calc_c8(N=N)
    U_Pz = calc_Pz(N=N, factor=1j)
    U_sd = calc_sd()
    U_Tx = calc_U_Tx()
    H_proj = project_M(H, psi_K)
    if subtract_Eav:
        H_proj -= Eav * np.eye(4)
    U_Pz_proj = project_M(U_Pz, psi_K)
    U_sd_proj = project_M(U_sd, psi_K)
    Tx_proj = inner_Tx(psi_K, psi_K, U_Tx=U_Tx)
    S1_proj = calc_S1(psi_K, U_Pz=U_Pz, U_Tx=U_Tx)
    ## Uncomment to instead view full 16x16 matrices in the energy eigenbasis
    # H_proj = project_M(H, evects)
    # U_Pz_proj = project_M(U_Pz, evects)
    # U_sd_proj = project_M(U_sd, evects)
    # Tx_proj = inner_Tx(evects, evects, U_Tx=U_Tx)
    # S1_proj = calc_S1(evects, U_Pz=U_Pz, U_Tx=U_Tx)
    # visualise_K_eigenstates(psi=psi_K, title_str='Numerical Eigenstates')
    visualise_matrix(H_proj, title_str=r'$H$, $H$ Eigenstates')
    visualise_matrix(U_Pz_proj, title_str=r'$P_z$, $H$ Eigenstates')
    visualise_matrix(U_sd_proj, title_str=r'$\sigma_d$, $H$ Eigenstates')
    visualise_matrix(Tx_proj, title_str=r'$T_x$, $H$ Eigenstates')
    visualise_matrix(S1_proj, title_str=r'$S_1$, $H$ Eigenstates')

    # Behaviour for eigenstates of Pz
    psi_Pz = np.zeros_like(psi_K)
    T, evects = scipy.linalg.schur(U_Pz_proj)
    psi_Pz = psi_K @ evects

    H_proj_Pz = project_M(H, psi_Pz)
    U_Pz_proj_Pz = project_M(U_Pz, psi_Pz)
    U_sd_proj_Pz = project_M(U_sd, psi_Pz)
    Tx_proj_Pz = inner_Tx(psi_Pz, psi_Pz, U_Tx=U_Tx)
    S1_proj_Pz = calc_S1(psi_Pz, U_Pz=U_Pz, U_Tx=U_Tx)
    # visualise_K_eigenstates(psi=psi_Pz, title_str=r'$P_z$ Eigenstates')
    # visualise_matrix(H_proj_Pz, title_str=r'$H$, $P_z$ Eigenstates')
    # visualise_matrix(U_Pz_proj_Pz, title_str=r'$P_z$, $P_z$ Eigenstates')
    # visualise_matrix(U_sd_proj_Pz, title_str=r'$\sigma_d$, $P_z$ Eigenstates')
    # visualise_matrix(Tx_proj_Pz, title_str=r'$T_x$, $P_z$ Eigenstates')
    # visualise_matrix(S1_proj_Pz, title_str=r'$S_1$, $P_z$ Eigenstates')
    
    # Behaviour for eigenstates of sd
    # psi_sd = np.zeros_like(psi_K)
    if block_diagonalise:
        evals, evects = diagonalise_within_blocks(U_sd_proj, block_idxs=[[0,1],[2,3]],
                                                return_evals=True)
    else:
        T, evects = scipy.linalg.schur(U_sd_proj)
        # evals, evects = fix_order(evects, H_proj, return_evals=True)
    # print(np.round(evals,5))
    # print(np.round(evects,3))

    psi_sd = psi_K @ evects

    # H_proj_sd = project_M(H_proj, evects)
    # U_Pz_proj_sd = project_M(U_Pz_proj, evects)
    # U_sd_proj_sd = project_M(U_sd_proj, evects)
    H_proj_sd = project_M(H, psi_sd)
    if subtract_Eav:
        H_proj_sd -= Eav * np.eye(4)
    U_Pz_proj_sd = project_M(U_Pz, psi_sd)
    U_sd_proj_sd = project_M(U_sd, psi_sd)
    Tx_proj_sd = inner_Tx(psi_sd, psi_sd, U_Tx=U_Tx)
    S1_proj_sd = calc_S1(psi_sd, U_Pz=U_Pz, U_Tx=U_Tx)
    visualise_matrix(H_proj_sd, title_str=r'$H$, $\sigma_d$ Eigenstates')
    visualise_matrix(U_Pz_proj_sd, title_str=r'$P_z$, $\sigma_d$ Eigenstates')
    visualise_matrix(U_sd_proj_sd, title_str=r'$\sigma_d$, $\sigma_d$ Eigenstates')
    visualise_matrix(Tx_proj_sd, title_str=r'$T_x$, $\sigma_d$ Eigenstates')
    visualise_matrix(S1_proj_sd, title_str=r'$S_1$, $\sigma_d$ Eigenstates')

    # Behaviour for eigenstates of Tx
    psi_Tx = np.zeros_like(psi_K)
    if block_diagonalise:
        evals, evects = diagonalise_within_blocks(Tx_proj, block_idxs=[[0,1],[2,3]],
                                                  return_evals=True)
    else:
        T, evects = scipy.linalg.schur(Tx_proj)
        evects = fix_order(evects, H_proj)
    psi_Tx = psi_K @ evects

    H_proj_Tx = project_M(H, psi_Tx)
    U_Pz_proj_Tx = project_M(U_Pz, psi_Tx)
    U_sd_proj_Tx = project_M(U_sd, psi_Tx)
    Tx_proj_Tx = inner_Tx(psi_Tx, psi_Tx, U_Tx=U_Tx)
    S1_proj_Tx = calc_S1(psi_Tx, U_Pz=U_Pz, U_Tx=U_Tx)
    # visualise_matrix(H_proj_Tx, title_str=r'$H$, $T_x$ Eigenstates')
    # visualise_matrix(U_Pz_proj_Tx, title_str=r'$P_z$, $T_x$ Eigenstates')
    # visualise_matrix(U_sd_proj_Tx, title_str=r'$\sigma_d$, $T_x$ Eigenstates')
    # visualise_matrix(Tx_proj_Tx, title_str=r'$T_x$, $T_x$ Eigenstates')
    # visualise_matrix(S1_proj_Tx, title_str=r'$S_1$, $T_x$ Eigenstates')
    

def fix_order(evects, H, return_evals=False):
    H_diag = evects.conj().T @ H @ evects
    evals = np.real_if_close(np.diag(H_diag))
    sort_idx = np.argsort(evals)
    evects_sorted = evects[:, sort_idx]
    if return_evals:
        evals_sorted = evals[sort_idx]
        return evals_sorted, evects_sorted
    else:
        return evects_sorted
    

def diagonalise_within_blocks(U, block_idxs=[[0,1],[2,3]], return_evals=False):
    Z_blocks = []
    T_blocks = []
    # print(np.round(U,3))
    for idxs in block_idxs:
        U_block = U[np.ix_(idxs,idxs)]
        # print(np.round(U_block,3))
        T, Z = scipy.linalg.schur(U_block)
        Z_blocks.append(Z)
        T_blocks.append(T)
    evects = scipy.linalg.block_diag(*Z_blocks)
    evals = np.concatenate([np.diag(T) for T in T_blocks])
    if return_evals:
        return evals, evects
    else:
        return evects
    

def left_S1(U1, A):
    return U1 @ A.conj()
    

def verify_commutations():
    Hx = np.kron(sz, I2)
    Hy = np.kron(sx, sx)
    sd = np.kron(I2, sz)
    a = 1
    b = -1
    U1 = np.kron(np.array([[a,0],[0,b]]), sy)        # S1 = U1 K
    sd2 = np.kron(0.5*(a+b)*sx + 0.5*(a-b)*sy, sy)
    # Dx = np.kron(sy, sy)
    Dx = np.kron(sz, sx)
    # Dx = np.kron(I2, sy)
    Dy = np.kron(sx, I2)
    Pz = 1j*np.kron(sy, sx)

    # Commutators of symmetry operators:
    print('Commutators of symmetry operators:')
    acomm_sdS1 = np.linalg.norm(sd @ U1 + left_S1(U1, sd))
    print(f'{{sd, S1}} = {np.round(acomm_sdS1, 3)}')
    acomm_sd2S1 = np.linalg.norm(sd2 @ U1 + left_S1(U1, sd2))
    print(f'{{sd2, S1}} = {np.round(acomm_sd2S1, 3)}')
    acomm_sdsd2 = np.linalg.norm(sd @ sd2 + sd2 @ sd)
    print(f'{{sd, sd2}} = {np.round(acomm_sdsd2, 3)}')
    acomm_Pzsd = np.linalg.norm(Pz @ sd + sd @ Pz)
    print(f'{{Pz, sd}} = {np.round(acomm_Pzsd, 3)}')
    acomm_Pzsd2 = np.linalg.norm(Pz @ sd2 + sd2 @ Pz)
    print(f'{{Pz, sd2}} = {np.round(acomm_Pzsd2, 3)}')
    comm_PzS1 = np.linalg.norm(Pz @ U1 - left_S1(U1, Pz))
    print(f'[Pz, S1] = {np.round(comm_PzS1, 3)}')


    # Commutators with Hx:
    print('\nCommutators with Hx:')
    comm_Hxsd = np.linalg.norm(Hx @ sd - sd @ Hx)
    print(f'[Hx, sd] = {np.round(comm_Hxsd, 3)}')
    comm_HxS1 = np.linalg.norm(Hx @ U1 - left_S1(U1, Hx))
    print(f'[Hx, S1] = {np.round(comm_HxS1, 3)}')
    acomm_Hxsd2 = np.linalg.norm(Hx @ sd2 + sd2 @ Hx)
    print(f'{{Hx, sd2}} = {np.round(acomm_Hxsd2, 3)}')
    acomm_HxPz = np.linalg.norm(Hx @ Pz + Pz @ Hx)
    print(f'{{Hx, Pz}} = {np.round(acomm_HxPz, 3)}')

    # Commutators with Dx:
    print('\nCommutators with Dx:')
    acomm_Dxsd = np.linalg.norm(Dx @ sd + sd @ Dx)
    print(f'{{Dx, sd}} = {np.round(acomm_Dxsd, 3)}')
    acomm_DxS1 = np.linalg.norm(Dx @ U1 + left_S1(U1, Dx))
    print(f'{{Dx, S1}} = {np.round(acomm_DxS1, 3)}')
    comm_Dxsd2 = np.linalg.norm(Dx @ sd2 - sd2 @ Dx)
    print(f'[Dx, sd2] = {np.round(comm_Dxsd2, 3)}')
    # acomm_DxHx = np.linalg.norm(Dx @ Hx + Hx @ Dx)
    # print(f'{{Dx, Hx}} = {np.round(acomm_DxHx, 3)}')
    # acomm_DxHy = np.linalg.norm(Dx @ Hy + Hy @ Dx)
    # print(f'{{Dx, Hy}} = {np.round(acomm_DxHy, 3)}')

    # Commutators with Hy:
    print('\nCommutators with Hy:')
    acomm_Hysd = np.linalg.norm(Hy @ sd + sd @ Hy)
    print(f'{{Hy, sd}} = {np.round(acomm_Hysd, 3)}')
    comm_HyS1 = np.linalg.norm(Hy @ U1 - left_S1(U1, Hy))
    print(f'[Hy, S1] = {np.round(comm_HyS1, 3)}')
    comm_Hysd2 = np.linalg.norm(Hy @ sd2 - sd2 @ Hy)
    print(f'[Hy, sd2] = {np.round(comm_Hysd2, 3)}')
    acomm_HyPz = np.linalg.norm(Hy @ Pz + Pz @ Hy)
    print(f'{{Hy, Pz}} = {np.round(acomm_HyPz, 3)}')

    # Commutators with Dy:
    print('\nCommutators with Dy:')
    comm_Dysd = np.linalg.norm(Dy @ sd - sd @ Dy)
    print(f'[Dy, sd] = {np.round(comm_Dysd, 3)}')
    acomm_DyS1 = np.linalg.norm(Dy @ U1 + left_S1(U1, Dy))
    print(f'{{Dy, S1}} = {np.round(acomm_DyS1, 3)}')
    acomm_Dysd2 = np.linalg.norm(Dy @ sd2 + sd2 @ Dy)
    print(f'{{Dy, sd2}} = {np.round(acomm_Dysd2, 3)}')
    # comm_DyHx = np.linalg.norm(Dy @ Hx - Hx @ Dy)
    # print(f'[Dy, Hx] = {np.round(comm_DyHx, 3)}')
    # comm_DyHy = np.linalg.norm(Dy @ Hy - Hy @ Dy)
    # print(f'[Dy, Hy] = {np.round(comm_DyHy, 3)}')
    acomm_DxDy = np.linalg.norm(Dx @ Dy + Dy @ Dx)
    print(f'{{Dx, Dy}} = {np.round(acomm_DxDy, 3)}')
    acomm_DxDyHx = np.linalg.norm(Dx @ Dy @ Hx + Hx @ Dx @ Dy)
    print(f'{{Dx Dy, Hx}} = {np.round(acomm_DxDyHx, 3)}')
    acomm_DxDyHy = np.linalg.norm(Dx @ Dy @ Hy + Hy @ Dx @ Dy)
    print(f'{{Dx Dy, Hy}} = {np.round(acomm_DxDyHy, 3)}')


def construct_representation(dq=0.001, N=5, subtract_Eav=True, block_diagonalise=True, 
                             D=1., V=1., **kwargs):
    q_K = np.array([0.5, 0.5*np.tan(np.pi/8)])
    dqx = dq * np.array([1, 0])
    Hx = calc_H(q=q_K + dqx, **kwargs)
    evals, evects = np.linalg.eigh(Hx)
    psi_K = evects[:, 4:8]
    Eav = np.mean(evals[4:8])

    # Behaviour for 'random' numerical eigenstates of H
    U_Pz = calc_Pz(N=N, factor=1)
    U_sd = calc_sd()
    U_Tx = calc_U_Tx()
    Hx_proj = project_M(Hx, psi_K)
    if subtract_Eav:
        Hx_proj -= Eav * np.eye(4)
    U_sd_proj = project_M(U_sd, psi_K)
    S1_proj = calc_S1(psi_K, U_Pz=U_Pz, U_Tx=U_Tx)
    visualise_matrix(Hx_proj, title_str=r'$H_x$, $H$ Eigenstates')
    visualise_matrix(U_sd_proj, title_str=r'$\sigma_d$, $H$ Eigenstates')
    visualise_matrix(S1_proj, title_str=r'$S_1$, $H$ Eigenstates')
    
    # Since [Hx, sd] = 0, have already diagonalised Hx and sd simultaneously
    # Now, only need to perform gauge transformations to get other operators in desired form

    # Get S1 in form tz x sy by gauge-transforming tau basis states:
    # print(S1_proj[:2,:2])
    a = (S1_proj[:2,:2] @ sy)[0,0]
    phi_a = np.angle(a)
    b = (S1_proj[2:,2:] @ sy)[0,0]
    phi_b = np.angle(b)
    U_tau = np.kron(np.diag([np.exp(-1j*phi_a/2), 1j*np.exp(-1j*phi_b/2)]), I2)
    # U = np.kron(Ut, I2)
    S1_proj = U_tau @ S1_proj @ U_tau
    visualise_matrix(S1_proj, title_str=r'$S_1$, Gauge-Transformed Eigenstates')

    # Construct sd2:
    U_sd2 = calc_sd2()
    U_sd2_proj = U_tau @ project_M(U_sd2, psi_K) @ U_tau.conj().T
    # Get in form ty x sy by gauge-transforming sigma basis states:
    c = U_sd2_proj[0,3]
    d = U_sd2_proj[1,2]
    a = (c+d)/(2*1j)
    b = (c-d)/2
    v = a + 1j*b
    phi = np.angle(1j / v)
    U_sigma = np.kron(I2, np.diag([1, np.exp(1j*phi)]))
    U_sd2_proj = U_sigma @ U_sd2_proj @ U_sigma.conj().T
    visualise_matrix(U_sd2_proj, title_str=r'$\sigma_d^\prime$, Gauge-Transformed Eigenstates')

    U_tot = U_sigma @ U_tau
    ## Uncomment to view unitary transformations of basis states
    # visualise_matrix(U_sigma, title_str=r'$U_\sigma$')
    # visualise_matrix(U_tau, title_str=r'$U_\tau$')
    # visualise_matrix(U_tot, title_str=r'$U_{tot}$')

    # Construct Hy; check it has expected tx x sx form:
    dqy = dq * np.array([0, 1])
    Hy = calc_H(q=q_K + dqy, **kwargs)
    Hy_proj = U_tot @ project_M(Hy, psi_K) @ U_tot.conj().T
    if subtract_Eav:
        Hy_proj -= Eav * np.eye(4)
    visualise_matrix(Hy_proj, title_str=r'$H_y$, Gauge-Transformed Eigenstates')

    # Construct Pz and Tx representations individually, instead of just combined S1
    U_Pz_proj = U_tot @ project_M(U_Pz, psi_K) @ U_tot.conj().T
    visualise_matrix(U_Pz_proj, title_str=r'$P_z$, Gauge-Transformed Eigenstates')

    Tx_proj = U_tot @ inner_Tx(psi_K, psi_K, U_Tx) @ U_tot      # NB different form of gauge-transformation as Tx is anti-unitary
    visualise_matrix(Tx_proj, title_str=r'$T_x$, Gauge-Transformed Eigenstates')

    # Construct Dx
    Dx = calc_Dx(D=D)
    Dx_proj = U_tot @ project_M(Dx, psi_K) @ U_tot.conj().T
    visualise_matrix(Dx_proj, title_str=r'$D_x$, Gauge-Transformed Eigenstates')
    
    # Construct Dy
    Dy = calc_Dy(D=D)
    Dy_proj = U_tot @ project_M(Dy, psi_K) @ U_tot.conj().T
    visualise_matrix(Dy_proj, title_str=r'$D_y$, Gauge-Transformed Eigenstates')

    # Construct HV
    HV = calc_HV(V=V)
    visualise_matrix(HV, title_str=r'$H_V$, Full Basis')
    HV_proj = U_tot @ project_M(HV, psi_K) @ U_tot.conj().T
    visualise_matrix(HV_proj, title_str=r'$H_V$, Gauge-Transformed Eigenstates')







if __name__ == '__main__':
    U_mag = 0.05
    l = np.arange(8)
    phi0 = 0
    N = 5
    U = -U_mag * np.exp(1j * (phi0 - 2 * np.pi * N * l / 8))
    V_mag = 0.0
    Dx = 0.0
    Dy = 0.00
    Dz = 0.0
    q_K = np.array([0.5, 0.5*np.tan(np.pi/8)])
    dq = np.array([0.00, 0.001])

    # U_c8 = calc_c8(N=5)
    # visualise_matrix(U_c8, title_str=r'$C_8$')
    # visualise_matrix(U_c8.conj().T, title_str=r'$C_8^{\dagger}$')
    # U_Pz = calc_Pz(N=5)
    # visualise_matrix(U_Pz, title_str=r'$P_z$')
    # visualise_matrix(U_Pz.conj().T, title_str=r'$P_z^{\dagger}$')
    # visualise_matrix(U_Pz.conj().T@U_Pz, title_str=r'$P_z^{\dagger} P_z$')
    

    # investigate_H_U(dq=dq, U=U, N=N, V=V_mag, Dx=Dx, Dy=Dy, Dz=Dz, subtract_Eav=True,
    #                 block_diagonalise=False, P=np.eye(4))

    # verify_commutations()

    # construct_representation(dq=0.001, U=U, N=N, V=V_mag, D=1)
    HV = calc_HV(V=1.)
    visualise_matrix(HV, title_str=r'$H_V$')
    # ## Calculate all 'component' matrices
    # Dx_mat = Dx * np.kron(np.eye(8), sx)
    # Dy_mat = Dy * np.kron(np.eye(8), sx)
    # Dz_mat = Dz * np.kron(np.eye(8), sx)


    # q = q_K - 0.0025 * np.array([1,0])
    # psi_K = calc_K_eigenstates(q=q, U=U, V=V_mag, Dx=Dx, Dy=Dy, Dz=Dz)
    # print(psi_K.shape)
    # # print(psi_K)
    # visualise_K_eigenstates(psi_K, U_mag=U_mag, V=V_mag, Dx=Dx, Dy=Dy, Dz=Dz)
