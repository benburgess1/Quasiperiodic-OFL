import numpy as np
import matplotlib.pyplot as plt

# Calculate bandstructure and DoS from perturbation theory

def E_free(k):
    return k**2


def E_1D(k, V=0.1, G=1):
    if np.abs(k) < G/2:
        s1 = -1
    else:
        s1 = 1
    if k < 0:
        s2 = 1
    else:
        s2 = -1
    return 0.5 * (E_free(k) + E_free(k+s2*G)) + s1 * np.sqrt(0.25 * (E_free(k) - E_free(k+s2*G))**2 + V**2)


def dEdk_1D(k, V=0.1, G=1):
    if np.abs(k) < G/2:
        s1 = -1
    else:
        s1 = 1
    if k < 0:
        s2 = 1
    else:
        s2 = -1
    return 2*k + s2*G + s1 * 0.5 * -s2 * G * (E_free(k) - E_free(k+s2*G)) / np.sqrt(0.25 * (E_free(k) - E_free(k+s2*G))**2 + V**2)



def calc_dos_1D(k_vals=np.linspace(0, 1, 1000), L=1e5, L_normalise=False,
                save=True, save_filename='Data.npz', 
                eps=1e-6, **kwargs):
    E_vals = np.array([E_1D(k, **kwargs) for k in k_vals])
    dos_vals = np.array([1/(np.abs(dEdk_1D(k, **kwargs)) + eps) for k in k_vals]) * L / np.pi
    if L_normalise:
        dos_vals /= L
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, L=L, **kwargs)
    else:
        return dos_vals


def calc_dos_free_1D(E_vals, L=1e5, L_normalise=False, save=True, save_filename='Data.npz', 
                     eps=1e-6, **kwargs):
    dos_vals = L / (2*np.pi*np.sqrt(E_vals + eps))
    if L_normalise:
        dos_vals /= L
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, L=L, **kwargs)
    else:
        return dos_vals
    
def calc_dos_free_2D(E_vals, L=1e5, L_normalise=False, save=True, save_filename='Data.npz', **kwargs):
    dos_vals = np.ones_like(E_vals) * L**2 / (4*np.pi)
    if L_normalise:
        dos_vals /= L**2
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, L=L, **kwargs)
    else:
        return dos_vals




if __name__ == '__main__':
    k_vals = np.linspace(0, 1.5, 1000)
    # E_vals = np.array([E_1D(k, V=0.0) for k in k_vals])
    # fig, ax = plt.subplots()
    # ax.plot(k_vals, E_vals, ls='', color='b', marker='.', ms=2)
    # plt.show()
    f = 'Green_Function/Data/2D/DoS_2D_free_theory.npz'
    calc_dos_free_2D(E_vals=np.linspace(0, 0.5, 250), L=5e3, save=True, save_filename=f, L_normalise=True)
    # f = 'Green_Function/Data/1D/DoS_1D_S2_V0.02_theory_normalised.npz'
    # calc_dos_free_1D(E_vals=np.linspace(0, 1, 250), L=5e3, L_normalise=True, save=True, save_filename=f)
    # calc_dos_1D(k_vals=k_vals, L=1e5, L_normalise=True, save=True, save_filename=f, V=0.02)

