import numpy as np
import matplotlib.pyplot as plt
import os
# import Plot_Approximant_Curvature as PAC
import Calc_Bandstructure as CBS
import matplotlib as mpl


def title_params(filename, include_bands=True, bands=None, include_orders=True):
    data = np.load(filename)
    title_str = ''
    if include_orders:
        o = data['orders']
        title_str += f'Orders = {o}, '
    params = {'U0':r'$|U|$', 'V0':r'$|V|$', 'N':r'$N$'}
    for key, val in params.items():
        if key in data:
            title_str += val + r'$ = $' + str(np.round(data[key],4)) + ', '
    if include_bands:
        if bands is None:
            bandtit = 'All Bands'
        elif len(bands) == 1:
            bandtit = f' Bands = {bands}'
        else:
            bandtit = f'Bands = [{np.min(bands)}-{np.max(bands)}]'
        title_str += bandtit
    else:
        title_str = title_str[:-2]
    return title_str


def plot_BS_line(filename, bands=None, plot_title=True, title_str=None,
                 x='auto', xticks=None, xticklabels=None, plot_legend=False,
                 E_R=1, E_lim=None, params_in_title=True, plot_BZ=False):
    data = np.load(filename)
    q_vals = data['q_vals']
    E_vals = data['E_vals']
    E_vals = np.sort(E_vals, axis=0)
    N = E_vals.shape[0]
    N_q = E_vals.shape[1]
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
            title_str = title_params(data, bands=bands)
        elif params_in_title:
            title_str += ', ' + title_params(data, bands=bands)
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
        ax.plot(x_vals, E_vals[n,:]/E_R, label=f'n = {n}')

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
    plt.show()


def plot_DoS(filename, plot_title=True, scalefactor=1., xlim=None,
             calc_new=False, dE=0.01, yticks=[0], chern_in_title=True,
             plot_E_max=True, n_occ=16, q_max=None, q_max_in_title=False,
             q_path=None, ax=None, plot=True, E_scale=None, color='b', 
             label='DoS', plot_legend=False, plot_val=None):
    data = np.load(filename)
    if calc_new:
        if xlim is not None:
            E_max = xlim[1]
        else:
            E_max = None
        dos_vals, E_bins = CBS.calc_DoS(filename, dE=dE, q_max=q_max, 
                                        q_path=q_path, E_max=E_max)
    else:
        dos_vals = data['dos_vals']
        if 'E_bins' in data:
            E_bins = data['E_bins']
        elif 'E_vals' in data:
            E_bins = data['E_vals']
        else:
            E_bins = np.arange(dos_vals.size)
    if ax is None:
        fig, ax = plt.subplots()
    if E_scale is not None:
        E_vals = data['E_vals']
        if q_path is not None:
            qx_vals, qy_vals = data['qx_vals'], data['qy_vals']
            qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
            qxx, qyy = qxx.flatten(), qyy.flatten()
            mask = q_path.contains_points(np.column_stack((qxx, qyy)))
            mask = mask.reshape(E_vals.shape[1:])
            E_vals = E_vals[:,mask]
        elif q_max is not None:
            qx_vals, qy_vals = data['qx_vals'], data['qy_vals']
            qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
            q2 = qxx**2 + qyy**2
            mask = q2 < q_max**2
            # mask = mask.reshape(E_vals.shape[1:])
            E_vals = E_vals[:, mask]
        scalefactor = np.sum(E_vals <= E_scale)

    ax.plot(E_bins, dos_vals/scalefactor, color=color, ls='-', marker='', label=label)
    ax.set_xlabel(r'$E$ / $E_R$')
    ax.set_ylabel(r'$\rho(E)$')
    if plot_E_max:
        E_vals = data['E_vals']
        if q_path is not None:
            qx_vals, qy_vals = data['qx_vals'], data['qy_vals']
            qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
            qxx, qyy = qxx.flatten(), qyy.flatten()
            mask = q_path.contains_points(np.column_stack((qxx, qyy)))
            mask = mask.reshape(E_vals.shape[1:])
            E_vals = E_vals[:,mask]
        elif q_max is not None:
            qx_vals, qy_vals = data['qx_vals'], data['qy_vals']
            qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
            q2 = qxx**2 + qyy**2
            mask = q2 < q_max**2
            # mask = mask.reshape(E_vals.shape[1:])
            E_vals = E_vals[:, mask]

        E_max = np.max(E_vals[np.arange(n_occ)])
        ax.axvline(x=E_max, color='r', ls='--', label=r'$E_{max}$')
    if plot_val is not None:
        ax.axhline(y=plot_val, color='r', ls=':', label='Predicted')
    
    if plot_legend:
        ax.legend()

    

    if plot_title:
        title_str = title_params(filename, include_bands=False, include_orders=True)
        if chern_in_title:
            C = np.round(np.real(data['C']),4)
            title_str += ', ' + r'$C = $' + str(C)
        if q_max_in_title:
            title_str += ', ' + r'$|q| < $' + str(np.round(q_max, 2))
        if n_occ is not None:
            title_str += f', max_idx = {n_occ-1}'
        # title_str = 'DoS'
        # params = {'U0':r'$|U|$', 'V0':r'$|V|$', 'N':r'$N$'}
        # for key, val in params.items():
        #     if key in data:
        #         title_str += ', ' + val + r'$ = $' + str(data[key])
        ax.set_title(title_str)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if plot:
        plt.show()


def plot_BS_surface(filename, E_R=1, bands=None, plot_title=True, 
                    title_str=None, q_max=None, q_max_in_title=False,
                    q_path=None):
    data = np.load(filename)
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    E_vals = data['E_vals']
    if q_max is not None:
        q2 = qxx**2 + qyy**2
        # print(q2.shape)
        mask = q2 < q_max**2
        # print(mask.shape)
        # qxx = qxx[mask]
        # qyy = qyy[mask]
        # E_vals = E_vals[:, mask]
        qxx = np.where(mask, qxx, np.nan)
        qyy = np.where(mask, qyy, np.nan)
        E_vals = np.where(mask[None, :, :], E_vals, np.nan)
        # print(qxx.shape)
        # print(E_vals.shape)
    elif q_path is not None:
        # qx_vals, qy_vals = data['qx_vals'], data['qy_vals']
        # qxx, qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
        # qxx, qyy = qxx.flatten(), qyy.flatten()
        points = np.column_stack((qxx.ravel(), qyy.ravel()))
        mask = q_path.contains_points(points)
        # print(mask.shape)
        mask = mask.reshape(qxx.shape)
        # print(E_vals.shape)
        # print(E_vals.shape[1:])
        # print(mask.shape)
        qxx = np.where(mask, qxx, np.nan)
        qyy = np.where(mask, qyy, np.nan)
        E_vals = np.where(mask[None, :, :], E_vals, np.nan)
    fig,ax = plt.subplots(subplot_kw={'projection':'3d'})
    if plot_title:
        title_str = title_params(filename, bands=bands)
        if q_max_in_title and q_max is not None:
            title_str += ', ' + r'$|q| < $' + str(np.round(q_max,2))
        ax.set_title(title_str)
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
    
        # if title_str is None:
        #     U0 = data['U0']
        #     V0 = data['V0']
        #     N_wind = data['N']
        #     title_str = (r'$|U| = $' + str(U0) + ', ' 
        #                  + r'$|V| = $' + str(V0) + ', '
        #                  + r'$N = $' + str(N_wind) + ', ' 
        #                  + bandtit)
        
    plt.show()


def compare_DoS(filenames, E_scale=0.25, colors=None, labels=None, orders=None,
                RQBZ=True, dE=0.01, xlim=None, cmap=plt.get_cmap('gnuplot'),
                title=None):
    fig,ax = plt.subplots()
    if colors is None:
        N = len(filenames)
        colors = [cmap(i / (N - 1 if N > 1 else 1)) for i in range(N)]
    if labels is None and orders is not None:
        labels = []
        for O in orders:
            labels.append(r'$O = $' + str(O))
    for i,f in enumerate(filenames):
        if RQBZ:
            calc_new = True
            x = (np.sqrt(5)-1)/(2*orders[i])
            patch = mpl.patches.Polygon(xy=CBS.calc_pentagon(G=q, x=x, invert=True), 
                                            edgecolor='k', facecolor=(0,0,0,0))
            q_path = patch.get_path()
        else:
            calc_new = False
            q_path = None
        plot_DoS(f, ax=ax, plot_title=False, plot=False, E_scale=E_scale, 
                 plot_E_max=False, color=colors[i], label=labels[i], 
                 calc_new=calc_new, q_path=q_path, dE=dE, xlim=xlim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    plt.show()




if __name__ == '__main__':
    # f = 'Data/BS_approx_a4_GXMG_U0.2_N3_V0.15.npz'
    # f = 'Data/BS_approx_a3_GXMG_U200.0_N1_V150.0.npz'
    # f = 'Data/BS_approx_a3_GXMG_U200.0_N3_V200.0.npz'
    # f = 'Data/BS_a2_GXMG_U0.2_N3_V0.2.npz'
    # c = data['cutoff']
    # print(c)
    # C = np.round(np.real(data['C']),4)
    # print(C)
    # E_vals = data['E_vals']
    # E_max = np.max(E_vals[np.arange(16),:,:])
    # print(E_max)
    # plot_BS_line(f, xticks='GXMG', E_lim=None, bands=None)

    # # f = 'DoS_approx_a3_U200.0_N3_V150.0_dE0.1.npz'
    # q_path = mpl.path.Path(vertices=CBS.calc_pentagon(G=1., invert=True))
    f = 'Updated Geometry/Data/RQBZData_o1_U0.0_N3_V0.0_finer.npz'
    f = 'Updated Geometry/Data/RQBZData_o8_c3.5_U0.2_N3_V0.15_extended.npz'
    data = np.load(f)
    q_max = np.max(data['qx_vals'])
    print(f'q_max = {q_max}')
    N = data['qx_vals'].size
    print(f'Size: {N} x {N}')
    q = 0.14589803
    N_bands = 202
    # x = (np.sqrt(5)-1)/(2*1)
    x = (np.sqrt(5)-1)/(2*np.sqrt(N_bands/2))
    print(f'RQBZ radius = {x}')
    patch = mpl.patches.Polygon(xy=CBS.calc_pentagon(G=q, x=x, invert=False), 
                                        edgecolor='k', facecolor=(0,0,0,0))
    q_path = patch.get_path()
    dE = 0.01
    dk = 2*q_max / (N-1)
    g = (2*np.pi / dk**2)
    # print(f'Predicted DoS: g = {g}')
    # plot_BS_surface(f, bands=np.arange(1), q_max=None, q_max_in_title=False, q_path=q_path)
    # plot_DoS(f, scalefactor=1., xlim=None, calc_new=True, dE=dE, n_occ=126, plot_E_max=True,
    #          chern_in_title=False, q_max=0.01, q_max_in_title=False, q_path=None, 
    #          plot_val=None, plot_legend=True, yticks=None)
    

    # f2 = 'Updated Geometry/Data/RQBZData_o2_U0.02_N3_V0.01.npz'
    # f3 = 'Updated Geometry/Data/RQBZData_o3_U0.02_N3_V0.01.npz'
    # f4 = 'Updated Geometry/Data/RQBZData_o4_U0.02_N3_V0.01.npz'
    # f5 = 'Updated Geometry/Data/RQBZData_o5_c3.5_U0.02_N3_V0.01.npz'
    # f6 = 'Updated Geometry/Data/RQBZData_o6_c3.5_U0.02_N3_V0.01.npz'
    f7_1 = 'Updated Geometry/Data/RQBZData_o7_c3.5_U0.02_N3_V0.01.npz'
    f8_1 = 'Updated Geometry/Data/RQBZData_o8_c3.5_U0.02_N3_V0.01.npz'
    # f9 = 'Updated Geometry/Data/RQBZData_o9_c3.5_U0.02_N3_V0.01.npz'
    # f2 = 'Updated Geometry/Data/RQBZData_o2_U0.0_N3_V0.0.npz'
    # f3 = 'Updated Geometry/Data/RQBZData_o3_U0.0_N3_V0.0.npz'
    f4 = 'Updated Geometry/Data/RQBZData_o4_U0.0_N3_V0.0.npz'
    f5 = 'Updated Geometry/Data/RQBZData_o5_c3.5_U0.0_N3_V0.0.npz'
    f5_old = 'Updated Geometry/Data/RQBZData_o5_c3.5_U0.0_N3_V0.0_old.npz'
    f6 = 'Updated Geometry/Data/RQBZData_o6_c3.5_U0.0_N3_V0.0.npz'
    f6_old = 'Updated Geometry/Data/RQBZData_o6_c3.5_U0.0_N3_V0.0_old.npz'
    f7 = 'Updated Geometry/Data/RQBZData_o7_c3.5_U0.0_N3_V0.0.npz'
    f8 = 'Updated Geometry/Data/RQBZData_o8_c3.5_U0.0_N3_V0.0.npz'
    f9 = 'Updated Geometry/Data/RQBZData_o9_c3.5_U0.0_N3_V0.0.npz'

    f2 = 'Updated Geometry/Data/TestData_o2_U0.2_N3_V0.15.npz'
    f3 = 'Updated Geometry/Data/TestData_o3_U0.2_N3_V0.15.npz'
    f4 = 'Updated Geometry/Data/TestData_o4_U0.2_N3_V0.15.npz'
    f5 = 'Updated Geometry/Data/TestData_o5_c3.5_U0.2_N3_V0.15.npz'
    f6 = 'Updated Geometry/Data/TestData_o6_c3.5_U0.2_N3_V0.15.npz'
    f7 = 'Updated Geometry/Data/TestData_o7_c3.5_U0.2_N3_V0.15.npz'
    f8 = 'Updated Geometry/Data/TestData_o8_c3.5_U0.2_N3_V0.15.npz'
    f9 = 'Updated Geometry/Data/TestData_o9_c3.5_U0.2_N3_V0.15.npz'
    f10 = 'Updated Geometry/Data/TestData_o10_c3.5_U0.2_N3_V0.15_updated.npz'

    filenames = [f4, f5, f6, f7, f8, f9, f10]
    orders = [4, 5, 6, 7, 8, 9, 10]
    filenames = [f7, f8, f9, f10]
    orders = [7, 8, 9, 10]
    # filenames = [f6, f6_old]
    # orders = [6, 6]
    # colors = ['b', 'r']
    # labels = ['New', 'Old']
    # colors = ['darkblue', 'mediumblue', 'royalblue', 'dodgerblue', 'lightblue']

    # compare_DoS(filenames, E_scale=0.25, colors=None, orders=orders, RQBZ=True,
    #             xlim=(-0.03, 0.53), labels=None, title='Intermediate Coupling')

    # print(os.getcwd())
    
    # f = 'Data/BS_approx_a3_surface_U200.0_N3_V150.0.npz'

    # f = 'BS_GXGXG_U0.2_N3_V0.1.npz'
    # plot_BS_line(f, x='qx', E_lim=(-0.28, 0.17), bands=None, 
    #              title_str=r'$GXGXG$', plot_BZ=True)
