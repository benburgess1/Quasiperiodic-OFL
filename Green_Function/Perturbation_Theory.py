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



def calc_dos_1D(k_vals=np.linspace(0, 1, 1000), L=1e5, save=True, save_filename='Data.npz', 
                eps=1e-6, **kwargs):
    E_vals = np.array([E_1D(k, **kwargs) for k in k_vals])
    dos_vals = np.array([1/(np.abs(dEdk_1D(k, **kwargs)) + eps) for k in k_vals]) * L / np.pi
    # dos_vals = np.zeros_like(E_vals)
    # dk = k_vals[1] - k_vals[0]
    # for i in range(len(E_vals)):
    #     if i == 0:
    #         dEdk = (E_vals[i+1] - E_vals[i]) / (dk)
    #     elif i == len(E_vals) - 1:
    #         dEdk = (E_vals[i] - E_vals[i-1]) / (dk)
    #     else:
    #         dEdk = (E_vals[i+1] - E_vals[i-1]) / (2*dk)
    #     dos_vals[i] = L / (np.pi * np.abs(dEdk))
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, L=L, **kwargs)
    else:
        return dos_vals


def calc_dos_free(E_vals, L=1e5, save=True, save_filename='Data.npz', 
                eps=1e-6, **kwargs):
    dos_vals = L / (2*np.pi*np.sqrt(E_vals + eps))
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
    f = 'Green_Function/Data/DoS_1D_S2_V0.15_theory_extended.npz'
    # calc_dos_free(E_vals=np.linspace(0, 1, 100), L=1e5, save=True, save_filename=f)
    calc_dos_1D(k_vals=k_vals, L=1e5, save=True, save_filename=f, V=0.15)

