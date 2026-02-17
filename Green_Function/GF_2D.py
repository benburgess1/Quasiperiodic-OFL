import numpy as np
from tqdm import tqdm

def G0(k, E, eps=1e-4, R=True, **kwargs):
    if not R:
        eps = -np.abs(eps)
    return 1 / (E - np.sum(k**2, axis=1) + 1j*eps)


def G(k, E, Sigma_func=None, **kwargs):
    if Sigma_func is None:
        Sigma = 0
    else:
        Sigma = Sigma_func(k, E, **kwargs)
    return G0(k, E - Sigma, **kwargs)


def calc_dos(E_vals, k_vals, L=1e4, save=False, save_filename='Data.npz', **kwargs):
    G_vals = G(k_vals[:, :, None], E_vals[None, :], **kwargs)
    dos_vals = np.sum(np.imag(G_vals), axis=0) * (-1/np.pi) / L**2
    # G_R_vals = G(k_vals[:, :, None], E_vals[None, :], **kwargs)
    # G_A_vals = G(k_vals[:, :, None], E_vals[None, :], R=False, **kwargs)
    # dG = G_R_vals - G_A_vals
    # dos_vals = np.sum(np.imag(dG), axis=0) * (-1/(2*np.pi))
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, L=L, **kwargs)
    return dos_vals


def adaptive_params(E, a=0.1, b=1, eps_min=1e-4, L_max=1e4, print_warnings=True, **kwargs):
    # Combine a and b into new parameters a-prime and b-prime
    # As in 1D case, original a corresponds to upper bound condition on epsilon;
    # i.e. making the peak sharp enough
    # Likewise, original b corresponds to resolvability of peak by choosing L large enough
    # However, acceptable L and epsilon depend on one another here, so we need 
    # to form the new parameters a-prime and b-prime
    # ap = (2 * b * a**2)**(1/3)
    # bp = (b / a)**(2/3) * 2**(5/3) * np.pi
    # eps = ap * E
    # L = bp / np.sqrt(E)
    eps = a * np.sqrt(E)
    if eps < eps_min:
        if print_warnings:
            print(f'\nWarning: eps = {eps:.3g} < eps_min = {eps_min:.3g}. Setting eps = eps_min.')
        eps = eps_min
    L = b * 4 * np.pi * np.sqrt(E) / eps
    if L > L_max:
        if print_warnings:
            print(f'Warning: L = {L:.3g} > L_max = {L_max:.3g}. Setting L = L_max.\n')
        L = L_max
    elif L < 1:
        if print_warnings:
            print(f'Warning: low value L = {L:.3g} detected. Setting L = L_max.\n')
        L = L_max
    return eps, L


def generate_k_vals(L, kmax=None, n=2, c=0.1, G=1, **kwargs):
    dk = 2 * np.pi / L
    if kmax is None:
        kmax = (n/2 + c) * G
    kx = np.arange(-kmax, kmax, dk)
    kxx, kyy = np.meshgrid(kx, kx)
    k_vals = np.column_stack((kxx.flatten(), kyy.flatten()))
    return k_vals


def calc_dos_adaptive(E_vals, save=False, save_filename='Data.npz', **kwargs):
    dos_vals = np.zeros_like(E_vals)
    for i, E in tqdm(enumerate(E_vals)):
        eps, L = adaptive_params(E, **kwargs)
        k_vals = generate_k_vals(L, **kwargs)
        print(f'E = {E}')
        print(f'eps = {eps}, L = {L}')
        print(f'N_k = {k_vals.shape[0]}')
        G_vals = G(k_vals, E, eps=eps, **kwargs)
        dos_vals[i] = np.sum(np.imag(G_vals)) * (-1/np.pi) / L**2
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, **kwargs)
    return dos_vals


if __name__ == '__main__':
    E_vals = np.linspace(0, 1, 200)[1:]
    kmax = 1.1
    L = 3e3
    dk = 2*np.pi / L
    kx_vals = np.arange(-kmax, kmax, dk)
    ky_vals = np.copy(kx_vals)
    print(kx_vals.shape)
    kxx, kyy = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    k_vals = np.column_stack((kxx.flatten(), kyy.flatten()))
    print(k_vals.shape)
    f = 'Green_Function/Data/2D/DoS_2D_free_GF_test.npz'
    # calc_dos(E_vals, k_vals, Sigma_func=None, L=L, eps=1e-3, save=True, save_filename=f)
    f = 'Green_Function/Data/2D/DoS_2D_free_GF_adaptive.npz'
    calc_dos_adaptive(E_vals, Sigma_func=None, save=True, save_filename=f, a=0.001, b=1, kmax=1.1, L_max=1e4)