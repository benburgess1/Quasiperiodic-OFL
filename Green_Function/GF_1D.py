import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def G0(k, E, eps=1e-4, **kwargs):
    return 1 / (E - k**2 + 1j*eps)


def G(k, E, Sigma_func=None, **kwargs):
    if Sigma_func is None:
        Sigma = 0
    else:
        Sigma = Sigma_func(k, E, **kwargs)
    return G0(k, E - Sigma, **kwargs)


def Sigma_2(k, E, V=0.05, G=1, **kwargs):
    return np.abs(V)**2 * (G0(k-G, E, eps=0) + G0(k+G, E, eps=0))


def Sigma_4(k, E, V=0.05, G=1, **kwargs):
    S2 = Sigma_2(k, E, V=V, G=G, **kwargs)
    S4 = np.abs(V)**4 * (G0(k-G, E, eps=0)**2 * G0(k-2*G, E, eps=0)
                         + G0(k+G, E, eps=0)**2 * G0(k+2*G, E, eps=0))
    return S2 + S4


def calc_dos(E_vals, k_vals, save=False, save_filename='Data.npz', **kwargs):
    G_vals = G(k_vals[:, None], E_vals[None, :], **kwargs)
    dos_vals = np.sum(np.imag(G_vals), axis=0) * (-1/np.pi)
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, **kwargs)
    return dos_vals


def Sigma_2_AA(k, E, V1=0.05, V2=0.025, G1=1, G2=1/np.sqrt(2), **kwargs):
    S2_1 = np.abs(V1)**2 * (G0(k-G1, E, eps=0) + G0(k+G1, E, eps=0))
    S2_2 = np.abs(V2)**2 * (G0(k-G2, E, eps=0) + G0(k+G2, E, eps=0))
    return S2_1 + S2_2


def Sigma_4_AA(k, E, V1=0.05, V2=0.025, G1=1, G2=1/np.sqrt(2), **kwargs):
    # Second order terms
    S2 = Sigma_2_AA(k, E, V1=V1, V2=V2, G1=G1, G2=G2, **kwargs)
    S4 = 0
    # Cross-coupling terms
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            S4 += (np.abs(V1)**2 * np.abs(V2)**2 
                   * G0(k+s1*G1, E, eps=0)**2 * G0(k+s1*G1+s2*G2, E, eps=0))
            S4 += (np.abs(V1)**2 * np.abs(V2)**2 
                   * G0(k+s1*G2, E, eps=0)**2 * G0(k+s1*G2+s2*G1, E, eps=0))
            S4 += (2 * np.abs(V1)**2 * np.abs(V2)**2 
                   * G0(k+s1*G1, E, eps=0) * G0(k+s1*G1+s2*G2, E, eps=0) * G0(k+s2*G2, E, eps=0))
    # 2G coupling terms
    S4 += np.abs(V1)**4 * (G0(k-G1, E, eps=0)**2 * G0(k-2*G1, E, eps=0)
                         + G0(k+G1, E, eps=0)**2 * G0(k+2*G1, E, eps=0))
    S4 += np.abs(V2)**4 * (G0(k-G2, E, eps=0)**2 * G0(k-2*G2, E, eps=0)
                         + G0(k+G2, E, eps=0)**2 * G0(k+2*G2, E, eps=0))
    return S2 + S4


def adaptive_params(E, a=0.1, b=1, eps_min=1e-5, L_max=1e5, print_warnings=True, **kwargs):
    # Choose epsilon based on upper bound set by E
    eps = a * E
    if eps < eps_min:
        if print_warnings:
            print(f'\nWarning: eps = {eps:.3g} < eps_min = {eps_min:.3g}. Setting eps = eps_min.')
        eps = eps_min
    L = b * 4 * np.pi * np.sqrt(E) / eps
    if L > L_max:
        if print_warnings:
            print(f'Warning: L = {L:.3g} < L_max = {L_max:.3g}. Setting L = L_max.\n')
        L = L_max
    elif L < 1:
        if print_warnings:
            print(f'Warning: low value L = {L:.3g} detected. Setting L = L_max.\n')
        L = L_max
    return eps, L
        

def generate_k_vals(E, L, c=2, kmin=0.1, **kwargs):
    dk = 2 * np.pi / L
    if E == 0:
        kmax = kmin
    else:
        kmax = np.sqrt(c*E)
    return np.arange(-kmax, kmax, dk)


def calc_dos_adaptive(E_vals, save=False, save_filename='Data.npz', **kwargs):
    dos_vals = np.zeros_like(E_vals)
    for i, E in tqdm(enumerate(E_vals)):
        eps, L = adaptive_params(E, **kwargs)
        k_vals = generate_k_vals(E, L, **kwargs)
        print(eps, L)
        print(k_vals.size)
        G_vals = G(k_vals, E, eps=eps, **kwargs)
        dos_vals[i] = np.sum(np.imag(G_vals)) * (-1/np.pi) / L
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, **kwargs)
    return dos_vals




if __name__ == '__main__':
    E_vals = np.linspace(0, 1, 200)
    kmax = 1.5
    L = 1e5
    dk = 2*np.pi / L
    k_vals = np.arange(-kmax, kmax, dk)
    # k_vals = np.array([0])
    # f = 'Green_Function/Data/DoS_1D_AA_S4_V10.05_V20.025_GF.npz'
    # calc_dos(E_vals, k_vals, Sigma_func=Sigma_4_AA, L=L, V1=0.05, V2=0.025, eps=1e-4, save=True, save_filename=f)
    f = 'Green_Function/Data/1D/DoS_1D_S2_V0.05_GF_adaptive.npz'
    calc_dos_adaptive(E_vals=E_vals, Sigma_func=Sigma_2, save=True, save_filename=f, a=0.01, b=3, c=10, V=0.05)

    

