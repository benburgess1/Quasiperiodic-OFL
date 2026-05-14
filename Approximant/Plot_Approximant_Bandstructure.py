import numpy as np
import matplotlib.pyplot as plt
import os
import Plot_Approximant_Curvature as PAC
import Approximant_Bandstructure as ABS

def plot_BS_line(filename, bands=None, plot_title=True, title_str=None,
                 x='auto', xticks=None, xticklabels=None, plot_legend=False,
                 E_R=1, E_lim=None, params_in_title=True, plot_BZ=False, ax=None,
                 plot=True, color='b', idx=None, idx_color='gold', lw=2):
    data = np.load(filename)
    if 'q_vals_GXMG' in data:
        q_vals = data['q_vals_GXMG']
    elif 'q_vals' in data:
        q_vals = data['q_vals']
    else:
        print('Error: no q values data')
        return
    if 'E_vals_GXMG' in data:
        E_vals = data['E_vals_GXMG']
    elif 'E_vals' in data:
        E_vals = data['E_vals']
    E_vals = np.sort(E_vals, axis=0)
    N = E_vals.shape[0]
    N_q = E_vals.shape[1]
    if ax is None:
        fig,ax = plt.subplots()
    if x == 'qx':
        x_vals = q_vals[:,0]
        xlab = r'$q_{x}$'
    elif x == 'qy':
        x_vals = q_vals[:,1]
        xlab = r'$q_{y}$'
    else:
        x_vals = np.arange(N_q)
        xlab = r'$q$'
        if xticks is not None:
            if xticks == 'GMKMKG':
                xticks = [0,49,98,147,196,245]
                xticklabels = [r'$\Gamma$', r'$M$', r'$K$', r'$M^{\prime}$', 
                               r'$K^{\prime}$', r'$\Gamma$']
            elif xticks == 'GMKG':
                xticks = [0,99,148,247], 
                xticklabels = [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$']
            elif xticks == 'GXMG':
                xticks = [0,49,98,147]
                xticklabels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$']
            elif xticks == 'boundary':
                xticks = [0,49,98,147,196]
                xticklabels = [r'$M_1$',r'$M_2$',r'$M_3$',r'$M_4$',r'$M_1$']
            ax.set_xticks(xticks)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels)
    if plot_title:
        if title_str is None:
            title_str = PAC.title_params(filename, bands=bands)
        elif params_in_title:
            title_str += ', ' + PAC.title_params(filename, bands=bands)
        # if title_str is None:
        #     U0 = data['U0']
        #     V0 = data['V0']
        #     N_wind = data['N']
        #     title_str = (r'$|U| = $' + str(U0) + ', ' 
        #                  + r'$|V| = $' + str(V0) + ', '
        #                  + r'$N = $' + str(N_wind) + ', ' 
        #                  + bandtit)
        ax.set_title(title_str)
    if bands is None:
        bands = np.arange(N)
    #     bandtit = 'All Bands'
    # elif len(bands) == 1:
    #     bandtit = f' Bands = {bands}'
    # else:
    #     bandtit = f'Bands = [{np.min(bands)}-{np.max(bands)}]'
    for n in bands:
        ax.plot(x_vals, E_vals[n,:]/E_R, label=f'n = {n}', color=color, lw=lw)
    if idx is not None:
        ax.plot(x_vals, E_vals[idx,:]/E_R, label=f'n = {n}', color=idx_color, lw=lw)

    if plot_BZ:
        if 'a' in data:
            a = data['a']
        else:
            a = 3
        ax.axvline(0.5/a, ls='--', color='k')
        ax.axvline(-0.5/a, ls='--', color='k')
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'$E$ / $E_{R}$')
    
    if plot_legend:
        ax.legend()
    if E_lim is not None:
        ax.set_ylim(*E_lim)
    if plot:
        plt.show()


def plot_DoS(filename, plot_title=True, scalefactor=1., xlim=None,
             calc_new=False, dE=0.01, yticks=[0], chern_in_title=True,
             plot_E_max=True, n_occ=16):
    data = np.load(filename)
    if calc_new:
        dos_vals, E_bins = ABS.calc_DoS(filename, dE=dE)
    else:
        dos_vals = data['dos_vals'] / scalefactor
        if 'E_bins' in data:
            E_bins = data['E_bins']
        elif 'E_vals' in data:
            E_bins = data['E_vals']
        else:
            E_bins = np.arange(dos_vals.size)

    fig, ax = plt.subplots()
    ax.plot(E_bins, dos_vals, color='b', ls='-', marker='', label='DoS')
    ax.set_xlabel(r'$E$ / $E_R$')
    ax.set_ylabel(r'$\rho(E)$')
    if plot_E_max:
        if 'E_vals_surf' in data:
            E_vals = data['E_vals_surf']
        elif 'E_vals' in data:
            E_vals = data['E_vals']
        E_max = np.max(E_vals[np.arange(n_occ),:,:])
        ax.axvline(x=E_max, color='r', ls='--', label=r'$E_{max}$')
        ax.legend()
    if plot_title:
        title_str = ''
        if 'a' in data:
            a = data['a']
            title_str += r'$a = $' + str(a) + ', '
        title_str += PAC.title_params(filename, include_bands=False)
        if chern_in_title:
            C = np.round(np.real(data['C']),4)
            title_str += ', ' + r'$C = $' + str(C)
        # title_str = 'DoS'
        # params = {'U0':r'$|U|$', 'V0':r'$|V|$', 'N':r'$N$'}
        # for key, val in params.items():
        #     if key in data:
        #         title_str += ', ' + val + r'$ = $' + str(data[key])
        ax.set_title(title_str)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_yticks(yticks)
    plt.show()


def plot_BS_surface(filename, E_R=1, bands=None, plot_title=True, 
                    title_str=None):
    data = np.load(filename)
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    E_vals = data['E_vals']
    fig,ax = plt.subplots(subplot_kw={'projection':'3d'})
    if bands is None:
        bands = np.arange(E_vals.shape[0])
    #     bandtit = 'All Bands'
    # elif len(bands) == 1:
    #     bandtit = f' Bands = {bands}'
    # else:
    #     bandtit = f'Bands = [{np.min(bands)}-{np.max(bands)}]'
    for n in bands:
        ax.plot_surface(qxx, qyy, E_vals[n,:,:]/E_R)
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$E$ / $E_{R}$', rotation=0, labelpad=10)
    if plot_title:
        title_str = ''
        if 'a' in data:
            a = data['a']
            title_str += r'$a = $' + str(a) + ', '
        title_str += PAC.title_params(f, bands=bands)
        # if title_str is None:
        #     U0 = data['U0']
        #     V0 = data['V0']
        #     N_wind = data['N']
        #     title_str = (r'$|U| = $' + str(U0) + ', ' 
        #                  + r'$|V| = $' + str(V0) + ', '
        #                  + r'$N = $' + str(N_wind) + ', ' 
        #                  + bandtit)
        ax.set_title(title_str)
    plt.show()


def compare_DoS(filenames, colors=None, linestyles=None, legend_params={}, 
                plot_legend=True, plot_title=True, title_params={}):
    if colors is None:
        colors = ['b'] * len(filenames)
    if linestyles is None:
        linestyles = ['-'] * len(filenames)
    fig, ax = plt.subplots()
    for i,f in enumerate(filenames):
        data = np.load(f)
        dos = data['dos_vals']
        E_bins = data['E_bins']
        label = ''
        for k,v in legend_params.items():
            label += v + ' = ' + str(np.round(data[k],2)) + ', '
        if len(label) > 2:
            label = label[:-2]
        ax.plot(E_bins, dos, color=colors[i], ls=linestyles[i], label=label)
    ax.set_xlabel(r'$E$ / $E_R$')
    ax.set_ylabel(r'$\rho(E)$')
    if plot_legend:
        ax.legend()
    if plot_title:
        title_str = ''
        for k,v in title_params.items():
            title_str += v + ' = ' + str(np.round(data[k],2)) + ', '
        title_str = title_str[:-2]
        ax.set_title(title_str)
    ax.set_yticks([0])
    plt.show()


def plot_spectral_function(
    filename: str,
    calc_new: bool = False,
    w_vals: np.ndarray = None,
    sigma: float = None,
    sym_points: list[tuple] = None,         # e.g. [(k_index, 'Γ'), (k_index, 'M'), ...]
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None,
    ax: plt.Axes = None,
    figsize: tuple = (6, 5),
    plot_planewave: bool = False,
    filename_planewave: str = None,
    plot_title = True,
    title_params = {}
) -> plt.Axes:
    """
    Plot the spectral function A(k, w) loaded from a .npz file.

    The .npz file should contain:
        k_vals:   shape (N_k, 2)
        evals_k:  shape (N_k, N_bands)
        weights_k: shape (N_k, N_bands)
    """
    data = np.load(filename)
    k_vals    = data['k_vals']
    evals_k   = data['evals_k']
    weights_k = data['weights_k']
    
    if calc_new:
        A = ABS.calc_spectral_function(evals_k, weights_k, w_vals, sigma)
    else:
        A = data['A']
        w_vals = data['w_vals']
        sigma = data['sigma']


    # dk = np.diff(k_vals, axis=0)
    # k_dist = np.concatenate([[0], np.cumsum(np.linalg.norm(dk, axis=1))])
    k_dist = np.arange(k_vals.shape[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    mesh = ax.pcolormesh(k_dist, w_vals, A.T, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_ticks([0])
    cbar.set_ticklabels(['0'])
    cbar.set_label(r'$A(k, \omega)$ (arb.)')

    if sym_points is not None:
        tick_positions, tick_labels = [], []
        for i_k, label in sym_points:
            x = k_dist[i_k]
            ax.axvline(x, color='white', linewidth=0.8, linestyle='--', alpha=0.7, zorder=3)
            tick_positions.append(x)
            tick_labels.append(label)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticks([])

    if plot_planewave:
        data_pw = np.load(filename_planewave)
        evals = data_pw['evals']
        ax.plot(k_dist, evals, color='r', ls='-', marker=None)


    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(w_vals[0], w_vals[-1])
    ax.set_ylabel(r'$\omega$')
    ax.set_xlabel(r'$k$')

    if plot_title:
        title_str = make_title_str(title_params, data, base_str=r'$A(k, \omega)$', dp=5)
        ax.set_title(title_str)

    plt.show()

    return ax

def make_title_str(title_params, data, base_str='', dp=5):
    for k, v in title_params.items():
        val = data[k]
        if len(base_str) > 0:
            base_str += ', '
        base_str += v + r'$=$' + f'{val:.{dp}g}'
    return base_str



if __name__ == '__main__':
    f = 'Approximant/Data/8Fold/Localisation/IPR_surface_a3_c3.5_R8_U0.15_V0.05_W0_N5.npz'
    plot_DoS(f, calc_new=True, dE=0.01, n_occ=14, xlim=(None, 1))
    # f = 'Approximant/Data/8Fold/SpectralFunction/Data_GMKG_R8_a7_c2.5_U0.03_N5_V0.0.npz'
    # U = 0.03
    # V = 0.00
    # # f = f'Updated_Geometry/Data/SpectralFunction_GMKG_O3_U{U}_V{V:.3g}_W0.005_N5_R8.npz'
    # # data = np.load(f)
    # # evals_k = data['evals_k']
    # # print(evals_k.shape)
    # f2 = f'Updated_Geometry/Data/QBZ_BS_GMKG_U{U}_V{V:.3g}_N5_R8.npz'
    # # w_vals = np.linspace(0.24, 0.26, 1000)
    # w_vals = np.linspace(-0.05, 0.4, 1000)
    # sigma = 0.001
    # plot_spectral_function(f, calc_new=True, w_vals=w_vals, sigma=sigma,
    #                     #    sym_points=None,
    #                        sym_points=[(0, r'$\Gamma$'), (99, r'$M$'), (198, r'$K$'), (297, r'$\Gamma$')],
    #                        plot_planewave=True, filename_planewave=f2,
    #                     #    title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$', 'W':r'$W$', 'orders':r'$O$'}
    #                        title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$', 'a':r'$N_a$'}
    #                        )
    # f = 'Data/BS_approx_a4_GXMG_U0.2_N3_V0.15.npz'
    # f = 'Data/BS_approx_a3_GXMG_U200.0_N1_V150.0.npz'
    # f = 'Data/BS_approx_a3_GXMG_U200.0_N3_V200.0.npz'
    # f = 'Approximant/Data/Data_NonAb_a4_U0.2_N4_V0.15.npz'
    # f = 'Approximant/Data/8Fold/Data_R8_a3_c2.5_U0.4_N5_V0.22.npz'
    # f = 'Approximant/Data/8Fold/Data_R8_a7_c2.5_U0.15_N5_V0.0.npz'
    # f = 'Approximant/Data/5Fold/Irregular/Data_R5_a4_c2.5_U0.2_N3_V0.15_l2_phi1.4.npz'
    # f = 'Approximant/Data/5Fold/AllCoherent/Data_R5_a4_c2.5_U0.2_N3_V0.1_corrected.npz'
    # f = 'Approximant/Data/8Fold/AllCoherent/Data_R8_a3_c2.5_U0.15_N5_V0.04_TEST7.npz'
    # f = 'DarkState/Data/5Fold/Data_R5_a3_c5.5_V110.0_V00.0_p11_p20_phi10_phi20.npz'
    # f = 'DarkState/Data/5Fold/Updated/Data_R5_a3_c5.5_V110.0_V00.0_p11_p20_phi10_phi20.npz'
    # f = 'DarkState/Data/5Fold/Data_R5_a6_c5.5_V110.0_V00.0_p11_p20_phi10_phi20.npz'
    # f = 'DarkState/Data/5Fold/Detailed_Chern_Data_R5_a3_c5.5_V110.0_V00.0_p11_p20_phi11_phi21.npz'
    # f = 'DarkState/Data/8Fold/Detailed_Chern_Data_R8_a3_c5.5_V110.0_V00.0_p11_p20_phi10_phi20.npz'
    # f = 'DarkState/Data/5Fold/Detailed_Chern_Data_R5_a3_c11.5_V1200.0_V00.0_p11_p20_phi10_phi20.npz'
    # f = 'Approximant/Data/8Fold/Old_Data/Data_R8_a3_c2.5_U0.28_N5_V0.1.npz'
    f = 'Approximant/Data/8Fold/Eigenvectors/Data_Multiband_R8_a3_c2.5_U0.3_N5_V0.12.npz'
    filenames = ['DarkState/Data/5Fold/Data_R5_a3_c5.5_V110.0_V00.0_p11_p20_phi10_phi20.npz',
                 'DarkState/Data/5Fold/Data_R5_a3_c7.5_V110.0_V00.0_p11_p20_phi10_phi20.npz']
    # f = 'DarkState/Data/5Fold/Detailed_Chern_Data_R5_a3_c3.5_V10.2_V00.0_p11_p20_phi10_phi20.npz'
    # f = 'DarkState/Data/3Fold/Data_R3_c5.5_V1100.0_V00.0_p1-1_p2-1_phi10_phi20.npz'
    # filenames = ['DarkState/Data/3Fold/Data_R3_c5.5_V1100.0_V00.0_p1-1_p2-1_phi10_phi20.npz',
    #              'DarkState/Data/3Fold/Data_R3_c7.5_V1100.0_V00.0_p1-1_p2-1_phi10_phi20.npz',
    #              'DarkState/Data/3Fold/Data_R3_c9.5_V1100.0_V00.0_p1-1_p2-1_phi10_phi20.npz',
    #              'DarkState/Data/3Fold/Data_R3_c11.5_V1100.0_V00.0_p1-1_p2-1_phi10_phi20.npz',
    #              'DarkState/Data/3Fold/Data_R3_c13.5_V1100.0_V00.0_p1-1_p2-1_phi10_phi20.npz']
    # f = 'Data/BS_a2_GXMG_U0.2_N3_V0.2.npz'
    # data = np.load(f)
    # idx = data['max_idx']
    # print(idx)
    # C = data['C']
    # print('C = ', np.round(-np.real(C),2))
    # dE = np.min(data['E_vals'][idx+1,:,:]) - np.max(data['E_vals'][idx,:,:])
    # print('dE = ' + str(np.round(dE,4)))
    # print(np.sum(np.real(C)[:13]))
    # E_vals = data['E_vals']
    # print(np.min(E_vals[14,:,:]) - np.max(E_vals[13,:,:]))
    # idx = data['max_idx']
    # print(idx)
    # C = np.round(np.real(data['C']),4)
    # print(C)
    # E_vals = data['E_vals']
    # E_max = np.max(E_vals[np.arange(16),:,:])
    # print(E_max)
    # plot_BS_line(f, xticks='GXMG', E_lim=None, bands=None, lw=1)

    # # f = 'DoS_approx_a3_U200.0_N3_V150.0_dE0.1.npz'
    # plot_BS_surface(f, bands=np.arange(24))
    # plot_DoS(f, scalefactor=1., xlim=None, calc_new=False, dE=0.01, n_occ=9, plot_E_max=True,
    #         chern_in_title=False)
    # for i in range(30,50):
    #     print(f'i = {i}')
    #     print(f'C = {np.round(np.sum(C[:i]), 2)}')
    #     plot_DoS(f, scalefactor=1., xlim=None, calc_new=False, dE=0.01, n_occ=i, plot_E_max=True,
    #         chern_in_title=False)
    # compare_DoS(filenames, colors=['b','r','c','orange','limegreen'], legend_params={'cutoff':r'$c$'},
    #             title_params={'V1':r'$|V_1|$', 'V0':r'$|V_0|$'})
    # print(os.getcwd())
    
    # f = 'Data/BS_approx_a3_surface_U200.0_N3_V150.0.npz'

    # f = 'BS_GXGXG_U0.2_N3_V0.1.npz'
    # plot_BS_line(f, x='qx', E_lim=(-0.28, 0.17), bands=None, 
    #              title_str=r'$GXGXG$', plot_BZ=True)
