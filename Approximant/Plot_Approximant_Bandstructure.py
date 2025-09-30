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



if __name__ == '__main__':
    # f = 'Data/BS_approx_a4_GXMG_U0.2_N3_V0.15.npz'
    # f = 'Data/BS_approx_a3_GXMG_U200.0_N1_V150.0.npz'
    # f = 'Data/BS_approx_a3_GXMG_U200.0_N3_V200.0.npz'
    # f = 'Approximant/Data/Data_NonAb_a4_U0.2_N4_V0.15.npz'
    # f = 'Approximant/Data/8Fold/Data_R8_a3_c2.5_U0.4_N5_V0.22.npz'
    # f = 'Approximant/Data/8Fold/Data_R8_a7_c2.5_U0.15_N5_V0.0.npz'
    # f = 'Approximant/Data/5Fold/Irregular/Data_R5_a4_c2.5_U0.2_N3_V0.15_l2_phi1.4.npz'
    # f = 'Approximant/Data/5Fold/AllCoherent/Data_R5_a4_c2.5_U0.2_N3_V0.1_corrected.npz'
    f = 'Approximant/Data/8Fold/AllCoherent/Data_R8_a3_c2.5_U0.15_N5_V0.04_TEST7.npz'
    # f = 'Data/BS_a2_GXMG_U0.2_N3_V0.2.npz'
    data = np.load(f)
    E_vals = data['E_vals']
    print(np.min(E_vals[14,:,:]) - np.max(E_vals[13,:,:]))
    # idx = data['max_idx']
    # print(idx)
    # C = np.round(np.real(data['C']),4)
    # print(C)
    # E_vals = data['E_vals']
    # E_max = np.max(E_vals[np.arange(16),:,:])
    # print(E_max)
    # plot_BS_line(f, xticks='GXMG', E_lim=None, bands=None, lw=1)

    # # f = 'DoS_approx_a3_U200.0_N3_V150.0_dE0.1.npz'
    plot_DoS(f, scalefactor=1., xlim=None, calc_new=False, dE=0.01, n_occ=14, plot_E_max=True,
             chern_in_title=False)
    # print(os.getcwd())
    
    # f = 'Data/BS_approx_a3_surface_U200.0_N3_V150.0.npz'
    # plot_BS_surface(f, bands=np.arange(75, 85))

    # f = 'BS_GXGXG_U0.2_N3_V0.1.npz'
    # plot_BS_line(f, x='qx', E_lim=(-0.28, 0.17), bands=None, 
    #              title_str=r'$GXGXG$', plot_BZ=True)
