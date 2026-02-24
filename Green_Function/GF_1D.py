import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def E_free(k, **kwargs):
    return k**2


def v_free(E, **kwargs):
    return 2*np.sqrt(np.abs(E))


def E_tb(k, t=1, a=1, **kwargs):
    return -2*t*np.cos(k*a)


def v_tb(E, t=1, a=1, **kwargs):
    k = np.arccos(E/(-2*t)) / a
    return 2*t*a*np.sin(k*a)


def G0(k, E, E0=E_free, eps=1e-4, **kwargs):
    return 1 / (E - E0(k, **kwargs) + 1j*eps)


def G(k, E, Sigma_func=None, **kwargs):
    if Sigma_func is None:
        Sigma = 0
    else:
        Sigma = Sigma_func(k, E, **kwargs)
    return G0(k, E - Sigma, **kwargs)


def Sigma_2(k, E, V=0.05, q=1, **kwargs):
    return np.abs(V)**2 * (G0(k-q, E, eps=0) + G0(k+q, E, eps=0))


def Sigma_4(k, E, V=0.05, q=1, **kwargs):
    S2 = Sigma_2(k, E, V=V, q=q, **kwargs)
    S4 = np.abs(V)**4 * (G0(k-q, E, eps=0)**2 * G0(k-2*q, E, eps=0)
                         + G0(k+q, E, eps=0)**2 * G0(k+2*q, E, eps=0))
    # S4 += np.abs(V)**4 * (G0(k-G, E, eps=0)**2 * G0(k, E, eps=0)
    #                       + G0(k+G, E, eps=0)**2 * G0(k, E, eps=0))
    return S2 + S4


def Sigma_6(k, E, V=0.05, q=1, **kwargs):
    S4 = Sigma_4(k, E, V=V, q=q, **kwargs)
    S6 = np.abs(V)**6 * (G0(k-q, E, eps=0)**2 * G0(k-2*q, E, eps=0)**2 * G0(k-3*q, E, eps=0)
                         + G0(k+q, E, eps=0)**2 * G0(k+2*q, E, eps=0)**2 * G0(k+3*q, E, eps=0))
    return S4 + S6


def calc_dos(E_vals, k_vals, save=False, save_filename='Data.npz', **kwargs):
    G_vals = G(k_vals[:, None], E_vals[None, :], **kwargs)
    dos_vals = np.sum(np.imag(G_vals), axis=0) * (-1/np.pi)
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, **kwargs)
    return dos_vals


def Sigma_2_2Component(k, E, V1=0.05, V2=0.025, q1=1, q2=1/np.sqrt(2), **kwargs):
    S2_1 = np.abs(V1)**2 * (G0(k-q1, E, eps=0) + G0(k+q1, E, eps=0))
    S2_2 = np.abs(V2)**2 * (G0(k-q2, E, eps=0) + G0(k+q2, E, eps=0))
    return S2_1 + S2_2


def Sigma_4_2Component(k, E, V1=0.05, V2=0.025, q1=1, q2=1/np.sqrt(2), **kwargs):
    # Second order terms
    S2 = Sigma_2_2Component(k, E, V1=V1, V2=V2, q1=q1, q2=q2, **kwargs)
    S4 = 0
    # Cross-coupling terms
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            S4 += (np.abs(V1)**2 * np.abs(V2)**2 
                   * G0(k+s1*q1, E, eps=0)**2 * G0(k+s1*q1+s2*q2, E, eps=0))
            S4 += (np.abs(V1)**2 * np.abs(V2)**2 
                   * G0(k+s1*q2, E, eps=0)**2 * G0(k+s1*q2+s2*q1, E, eps=0))
            S4 += (2 * np.abs(V1)**2 * np.abs(V2)**2 
                   * G0(k+s1*q1, E, eps=0) * G0(k+s1*q1+s2*q2, E, eps=0) * G0(k+s2*q2, E, eps=0))
    # 2G coupling terms
    S4 += np.abs(V1)**4 * (G0(k-q1, E, eps=0)**2 * G0(k-2*q1, E, eps=0)
                         + G0(k+q1, E, eps=0)**2 * G0(k+2*q1, E, eps=0))
    S4 += np.abs(V2)**4 * (G0(k-q2, E, eps=0)**2 * G0(k-2*q2, E, eps=0)
                         + G0(k+q2, E, eps=0)**2 * G0(k+2*q2, E, eps=0))
    return S2 + S4


def adaptive_params(E, v0=v_free, A=0.1, B=1, eps_min=1e-5, L_max=1e5, print_warnings=True, **kwargs):
    E = np.abs(E)
    # Choose epsilon based on upper bound set by E
    eps = A * E
    if eps < eps_min:
        if print_warnings:
            print(f'\nWarning: eps = {eps:.3g} < eps_min = {eps_min:.3g}. Setting eps = eps_min.')
        eps = eps_min
    # Choose L based on lower bound set by E and epsilon
    L = B * 2 * np.pi * v0(E, **kwargs) / eps
    if L > L_max:
        if print_warnings:
            print(f'Warning: L = {L:.3g} > L_max = {L_max:.3g}. Setting L = L_max.\n')
        L = L_max
    elif L < 1:
        if print_warnings:
            print(f'Warning: low value L = {L:.3g} detected. Setting L = L_max.\n')
        L = L_max
    return eps, L
        

def generate_k_vals(L, kmax=None, n=2, C=0.1, q=1, **kwargs):
    dk = 2 * np.pi / L
    if kmax is None:
        kmax = (n/2 + C) * q
    k_vals = np.arange(-kmax, kmax, dk)
    return k_vals


def calc_dos_adaptive(E_vals, save=False, save_filename='Data.npz', **kwargs):
    dos_vals = np.zeros_like(E_vals)
    for i, E in tqdm(enumerate(E_vals)):
        eps, L = adaptive_params(E, **kwargs)
        k_vals = generate_k_vals(L, **kwargs)
        print(f'eps = {eps}, L = {L}')
        print(f'N_k = {k_vals.size}')
        G_vals = G(k_vals, E, eps=eps, **kwargs)
        dos_vals[i] = np.sum(np.imag(G_vals)) * (-1/np.pi) / L
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, **kwargs)
    return dos_vals




if __name__ == '__main__':
    E_vals = np.linspace(-2, 2, 1000)
    kmax = 1.1 * np.pi
    L = 1e5
    dk = 2*np.pi / L
    k_vals = np.arange(-kmax, kmax, dk)
    # k_vals = np.array([0])
    # f = 'Green_Function/Data/DoS_1D_AA_S4_V10.05_V20.025_GF.npz'
    # calc_dos(E_vals, k_vals, Sigma_func=Sigma_4_AA, L=L, V1=0.05, V2=0.025, eps=1e-4, save=True, save_filename=f)
    # f = 'Green_Function/Data/1D/DoS_1D_free_GF_adaptive_updated.npz'
    # f = 'Green_Function/Data/1D/DoS_1D_S2_V0.1_GF_adaptive_updated.npz'
    # f = 'Green_Function/Data/1D/DoS_1D_S2_V0.01_GF_adaptive.npz'
    # calc_dos_adaptive(E_vals=E_vals, Sigma_func=Sigma_2, save=True, save_filename=f, A=0.001, B=10, C=0.1, n=2, 
    #                   V=0.01, kmax=1.1, L_max=2e5)
    f = 'Green_Function/Data/1D/TB/DoS_1D_TB_t1_GF.npz'
    calc_dos_adaptive(E_vals=E_vals, 
                      E0=E_tb, v0=v_tb, t=1, a=1, 
                      Sigma_func=None, 
                      save=True, save_filename=f, 
                      A=0.001, B=10, kmax=1.*np.pi, L_max=1e6)

    

