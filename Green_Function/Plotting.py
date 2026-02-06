import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.ticker import ScalarFormatter


def plot_dos(filename, plot_theory=False, filename_theory='', L=1e5, ylim=None, 
             plot_title=True, title_params={}):
    data = np.load(filename)
    E_vals = data['E_vals']
    dos_vals = data['dos_vals']
    fig, ax = plt.subplots()
    ax.plot(E_vals, dos_vals, color='b', ls='-', marker=None, label='Numerics')
    ax.set_xlabel(r'$E/E_R$')
    ax.set_ylabel(r'$\rho$ (arb.)')
    if plot_theory:
        # def sqrt_fit(x, A):
        #     eps = 1e-4
        #     return A / np.sqrt(x + eps)
        # popt, pcov = sp.optimize.curve_fit(f, E_vals, dos_vals, p0=[1e3])
        # eps = 1e-4
        # y = L / (2*np.pi*np.sqrt(E_vals + eps))
        # ax.plot(E_vals, y, color='r', ls='--', marker=None)
        data_theory = np.load(filename_theory)
        E_vals = data_theory['E_vals']
        dos_vals = data_theory['dos_vals']
        ax.plot(E_vals, dos_vals, color='r', ls='', marker='.', ms=2, label='Theory')
        ax.legend()

    if ylim is not None:
        ax.set_ylim(*ylim)
    formatter = ScalarFormatter(useOffset=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    if plot_title:
        title_str = make_title_str(title_params, data, base_str=r'$\rho (E)$')
        ax.set_title(title_str)


    plt.show()


def make_title_str(title_params, data, base_str='', dp=5):
    for k, v in title_params.items():
        val = data[k]
        if len(base_str) > 0:
            base_str += ', '
        base_str += v + r'$=$' + f'{val:.{dp}g}'
    return base_str


if __name__ == '__main__':
    # f = 'Green_Function/Data/DoS_1D_S4_V0.15_GF.npz'
    f2 = 'Green_Function/Data/DoS_1D_S2_V0.15_theory_extended.npz'
    f = 'Green_Function/Data/DoS_1D_AA_S4_V10.05_V20.025_GF.npz'
    plot_dos(f, ylim=(-0.1e5, 2.5e5), plot_title=True, 
             title_params={'V1':r'$|V_1|$', 'V2':r'$|V_2|$', 'L':r'$L$', 'eps':r'$\epsilon$'},
             plot_theory=False, filename_theory=f2)