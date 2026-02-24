import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.ticker import ScalarFormatter


def plot_dos(filenames, colors=None, labels=None, plot_theory=False, filename_theory='', L=1e5, 
             ylim=None, xlim=None,
             plot_title=True, title_params={}, plot_legend=True,
             xlab=r'$E/E_R$'):
    fig, ax = plt.subplots()
    if colors is None:
        colors = ['b' for i in range(len(filenames))]
    if labels is None:
        labels = ['Numerics' for i in range(len(filenames))]
    for i,f in enumerate(filenames):
        data = np.load(f)
        E_vals = data['E_vals']
        dos_vals = data['dos_vals']
        ax.plot(E_vals, dos_vals, color=colors[i], ls='-', marker=None, label=labels[i])
    ax.set_xlabel(xlab)
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
    if plot_legend:
        ax.legend()

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    
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
    # f2 = 'Green_Function/Data/DoS_1D_S2_V0.15_theory_extended.npz'
    # f = 'Green_Function/Data/DoS_1D_AA_S4_V10.05_V20.025_GF.npz'
    # f = 'Green_Function/Data/2D/DoS_2D_free_GF_adaptive.npz'
    # f = 'Green_Function/Data/2D/DoS_2D_free_GF_test.npz'
    # f2 = 'Green_Function/Data/2D/DoS_2D_free_theory.npz'
    # f = 'Green_Function/Data/1D/DoS_1D_S2_V0.1_GF_adaptive_updated.npz'
    f1 = 'Green_Function/Data/1D/DoS_1D_S2_V0.01_GF_adaptive.npz'
    f2 = 'Green_Function/Data/1D/DoS_1D_S4_V0.05_GF_adaptive.npz'
    f3 = 'Green_Function/Data/1D/DoS_1D_S6_V0.05_GF_adaptive.npz'
    f4 = 'Green_Function/Data/1D/DoS_1D_iterative_n4_V0.05.npz'
    filenames = [f1, f2, f3]
    filenames = [f3]
    filenames = [f1, f4]
    # ft = 'Green_Function/Data/1D/Dos_1D_S4_V0.05_theory.npz'
    ft = 'Green_Function/Data/1D/Dos_1D_S2_V0.01_theory_normalised.npz'
    # f2 = 'Green_Function/Data/1D/DoS_1D_S2_V0.02_theory_normalised.npz'
    # f = 'Green_Function/Data/1D/DoS_1D_free_GF_adaptive_updated.npz'
    # f2 = 'Green_Function/Data/1D/DoS_1D_free_theory_normalised.npz'
    # plot_dos(filenames=filenames, 
    #         #  colors=['b','r','cyan'], labels=['S2', 'S4', 'S6'], 
    #         colors = ['b', 'cyan'], labels=['Self-energy', 'Iterative'],
    #          ylim=(-0.5,10), xlim=(-0.1,1), plot_title=True, 
    #         #  title_params = {'a':r'$a$', 'b':r'$b$', 'kmax':r'$k_{max}$'},
    #         #  title_params={'L':r'$L$', 'eps':r'$\epsilon$'},
    #         title_params = {'V':r'$|V|$'},
    #          plot_theory=True, filename_theory=ft)
    f1 = 'Green_Function/Data/1D/TB/DoS_1D_TB_t1_GF.npz'
    filenames = [f1]
    ft = 'Green_Function/Data/1D/TB/DoS_1D_t1_theory.npz'
    plot_dos(filenames=filenames, 
             colors = ['b'], labels=['GF'],
             ylim=(-0.1,3), xlim=None, plot_title=True, 
             title_params = {'t':r'$t$', 'a':r'$a$'},
             xlab=r'$E$ / $t$',
             plot_theory=True, filename_theory=ft)
