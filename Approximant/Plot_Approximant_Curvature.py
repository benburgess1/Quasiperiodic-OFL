import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import Decagon
import matplotlib.colors as mcolors
import Approximant_Curvature as AC
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap


# Generate colormap to use later when plotting
bwr = plt.cm.get_cmap('bwr')
upper_half_bwr = mcolors.LinearSegmentedColormap.from_list(
    'upper_half_bwr', bwr(np.linspace(0.5, 1, 256)))


max_idx_dict = {1:1, 2:3, 3:15, 4:27}


def plot_curv_line(filename, bands=None, plot_title=True, title_str=None, 
                   real=True, x='auto', xticks=None, xticklabels=None,
                   plot_abs=False, plot_legend=True, plot_sum=False,
                   scale_factor=1., ax=None, plot=True):
    data = np.load(filename)
    q_vals = data['q_vals']
    curv_vals = data['curv_vals'] / scale_factor
    # print(curv_vals)
    if ax is None:
        fig,ax = plt.subplots()
    if bands is None:
        bands = np.arange(curv_vals.shape[0])
    if real:
        curv_vals = np.real(curv_vals)
        ylab = r'$\Omega_{n}$'
    else:
        curv_vals = np.imag(curv_vals)
        ylab = r'$Im(\Omega_{n})$'
    if plot_abs:
        curv_vals = np.abs(curv_vals)
        ylab = r'$|$' + ylab + r'$|$'
    if x == 'auto':
        x_vals = np.arange(q_vals.shape[0])
        xlab = r'$q$'
    elif x == 'qx':
        x_vals = q_vals[:,0]
        xlab = r'$q_x$'
    elif x == 'qy':
        x_vals = q_vals[:,1]
        xlab = r'$q_y$'
    if not plot_sum:
        for n in bands:
            ax.plot(x_vals, curv_vals[n,:], label=r'$n = $' + str(n))
    else:
        curv_sum = np.zeros(curv_vals.shape[1])
        for n in bands:
            curv_sum += curv_vals[n,:]
        ax.plot(x_vals, curv_sum)
        ylab = r'$\Sigma_{n}$' + ylab

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab, rotation=0, labelpad=10)
    if plot_legend:
        ax.legend()
    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    if plot_title:
        if title_str is None:
            U0 = data['U0']
            V0 = data['V0']
            N = data['N']
            title_str = (r'$|U| = $' + str(U0) + ', ' + r'$|V| = $' + str(V0) 
                         + ', ' + r'$N = $' + str(N) 
                         + f', Bands = {bands}')
        ax.set_title(title_str)
    if plot:
        plt.show()



def plot_curv_surface(filename, bands=None, plot_title=True, 
                      title_str=None, real=True, plot_abs=False, 
                      plot_log=False):
    data = np.load(filename)
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    curv_vals = data['curv_vals']
    fig,ax = plt.subplots(subplot_kw={'projection':'3d'})
    if bands is None:
        bands = range(curv_vals.shape[0])
    if real:
        curv_vals = np.real(curv_vals)
        zlab = r'$\Omega$'
    else:
        curv_vals = np.imag(curv_vals)
        zlab = r'$Im(\Omega)$'
    for i in bands:
        ax.plot_surface(qxx, qyy, curv_vals[i,:,:])
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlab, rotation=0, labelpad=10)
    # ax.zaxis.label.set_rotation(90)
    if plot_title:
        if title_str is None:
            U = data['U']
            N = data['N']
            title_str = (r'$|U| = $' + str(U) + ', ' + r'$N = $' + str(N) 
                         + f', Bands = {bands}')
        ax.set_title(title_str)
    plt.show()


def plot_curv_contour(filename, bands=None, plot_title=True, 
                      title_str=None, real=True, cmap='default',
                      plot_cbar=True, plot_BZ=True, plot_abs=False, 
                      plot_log=False, patch=None, chern=False, inside_BZ=False,
                      scale_factor=1., a=3, axlim=None, shift_q=False,
                      max_idx_dict=max_idx_dict, ax=None, plot_fig=True,
                      levels=None):
    data = np.load(filename)
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    if 'a' in data:
        a = data['a']
    if shift_q:
        dqx = qx_vals[1] - qx_vals[0]
        qx_vals = (qx_vals + dqx/2)[:-1]
        dqy = qy_vals[1] - qy_vals[0]
        qy_vals = (qy_vals + dqy/2)[:-1]
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    curv_vals = data['curv_vals'] / scale_factor
    if curv_vals.shape[1:] != qxx.shape:
        curv_vals = curv_vals[:, :qxx.shape[0], :qxx.shape[1]]
    if inside_BZ:
        BZ_path = AC.square_BZ(a=0.999*a).get_path()
        points = np.column_stack((qxx.ravel(), qyy.ravel()))
        mask = BZ_path.contains_points(points)
        curv_plot = np.full_like(curv_vals, 0.)
        curv_plot[:, mask.reshape(qxx.shape)] = curv_vals[:, mask.reshape(qxx.shape)]
        # print(curv_plot.shape)
        # curv_plot = curv_plot.reshape((curv_plot.shape[0], *qxx.shape))
    else:
        curv_plot = curv_vals
    if ax is None:
        fig,ax = plt.subplots()
    if bands is None:
        bands = np.arange(curv_plot.shape[0])
    #     bandtit = 'All Bands'
    # elif len(bands) == 1:
    #     bandtit = f' Bands = {bands}'
    # else:
    #     bandtit = f'Bands = [{np.min(bands)}-{np.max(bands)}]'
    if real:
        curv_plot = np.real(curv_plot)
        zlab = r'$\Omega_{n}$'
    else:
        curv_plot = np.imag(curv_plot)
        zlab = r'$Im(\Omega_{n})$'
        #cmap = plt.colormaps['hot']
    # curv_sum = np.zeros((curv_plot.shape[1],curv_plot.shape[2]), 
    #                     dtype=np.float64)
    # for i in bands:
    #     curv_sum += curv_plot[i,:,:]
    curv_sum = np.sum(curv_plot[bands,:,:], axis=0)
    if len(bands) > 1 or 'NonAb' in filename:
        zlab = r'$\Sigma_{n}$' + zlab
    if plot_abs:
        curv_sum = np.abs(curv_sum)
        zlab = r'$|$' + zlab + r'$|$'
        if cmap == 'default':
            cmap = upper_half_bwr
    else:
        if cmap == 'default':
            cmap = plt.colormaps['bwr']
    if plot_log:
        curv_sum = np.log10(curv_sum)
        zlab = r'$log_{10}($' + zlab + r'$)$'
        vmin = np.min(curv_sum)
        vmax = np.max(curv_sum)
        if levels is None:
            levels = np.linspace(vmin,vmax,200)
            ticks = [vmin,vmax]
    else:
        vmax = np.max(np.abs(curv_sum))
        if plot_abs:
            if levels is None:
                levels = np.linspace(0,vmax,200)
                ticks = [0,vmax]
        else:
            if levels is None:
                levels = np.linspace(-vmax,vmax,200)
                ticks = [-vmax,0,vmax]
    #print(levels)
    plot = ax.contourf(qxx, qyy, curv_sum, cmap=cmap, levels=levels)
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$', rotation=0)
    if plot_title:
        title_str = title_params(filename, bands=bands, max_idx_dict=max_idx_dict, include_bands=False)
        if chern:
            # print(curv_vals.shape)
            C = AC.calc_chern_number(curv_vals=curv_vals, qx_vals=qx_vals, 
                                     qy_vals=qy_vals, bands=bands, 
                                     inside_BZ=inside_BZ, a=a, shift_q=shift_q)
            C_str = str(np.round(np.real(C), 5))
            title_str += ', ' + r'$C = $' + C_str
        ax.set_title(title_str)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab, rotation=0)
    if plot_BZ:
        ax.add_patch(AC.square_BZ(a=a, color='k'))
    if patch is not None:
        ax.add_patch(patch)
    if axlim is None:
        ax.set_xlim(np.min(qx_vals), np.max(qx_vals))
        ax.set_ylim(np.min(qy_vals), np.max(qy_vals))
    else:
        ax.set_xlim(*axlim)
        ax.set_ylim(*axlim)
    if plot_fig:
        plt.show()
    else:
        return plot


def title_params(filename, include_bands=True, bands=None, max_idx_dict=max_idx_dict):
    data = np.load(filename)
    title_str = ''
    params = {'U0':r'$|U|$', 'V0':r'$|V|$', 'N':r'$N$', 'R':r'$R$'}
    for key, val in params.items():
        if key in data:
            title_str += val + r'$ = $' + str(np.round(data[key],4)) + ', '
    if include_bands:
        if 'NonAb' in filename:
            a = int(data['a'])
            bandtit = f'Bands = [0-{max_idx_dict[a]}]'
        elif bands is None:
            bandtit = 'All Bands'
        elif len(bands) == 1:
            bandtit = f' Bands = {bands}'
        else:
            bandtit = f'Bands = [{np.min(bands)}-{np.max(bands)}]'
        title_str += bandtit
    else:
        title_str = title_str[:-2]
    return title_str


def plot_C_multiband(filename, calc_new=False, cmap=plt.colormaps['bwr'], 
                     scalefactor=1., plot_title=True, plot_cbar=True):
    data = np.load(filename)
    if calc_new:
        C = AC.calc_chern_multiband(curv_vals=data['curv_vals'])
    else:
        C = data['C']

    vmax = np.max(np.abs(C))
    levels = np.arange(-np.round(vmax), np.round(vmax) + 2)
    fig, ax = plt.subplots()

    im = ax.imshow(scalefactor * C[np.newaxis, :], cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(C.size))
    ax.set_yticks([])
    if plot_title:
        title_str = title_params(filename, include_bands=False)
        title_str += f', vmax = {np.round(vmax)}'
        ax.set_title(title_str)
    if plot_cbar:
        boundaries = np.arange(-np.round(vmax) - 0.5, np.round(vmax) + 1.5)
        norm_for_cbar = BoundaryNorm(boundaries=boundaries, ncolors=256)  # used only for colorbar

        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation='horizontal',
            pad=0.15,
            boundaries=boundaries,
            ticks=np.arange(-np.round(vmax), np.round(vmax) + 1),
            spacing='proportional'
        )
        # cbar.ax.set_xticklabels([str(i) for i in range(-np.round(vmax), np.round(vmax) + 1)])
    plt.show()

    


if __name__ == '__main__':
    # f = -(0.6/300)**2
    # f = 'Data/Curv_approx_a3_Fukui_U200.0_N4_V150.0.npz'
    # f = 'Data/Data_a3_U0.125_N3_V0.05.npz'

    # f = 'Data/Data_a3_U0.2_N3_V0.1.npz'
    # f = 'Approximant/Data/Data_NonAb_a4_U0.2_N4_V0.15.npz'
    # f = 'Approximant/Data/8Fold/Data_R8_a3_c2.5_U0.15_N5_V0.1061_corrected.npz'
    # f = 'Approximant/Data/8Fold/Data_Multiband_R8_a3_c2.5_U0.36_N5_V0.26.npz'
    # f = 'Approximant/Data/8Fold/Data_R8_a7_c2.5_U0.15_N5_V0.001.npz'
    # f = 'Approximant/Data/5Fold/Irregular/Data_R5_a4_c2.5_U0.2_N3_V0.15_l2_phi1.4.npz'
    # f = 'Approximant/Data/5Fold/AllCoherent/Data_R5_a4_c2.5_U0.2_N3_V0.24_corrected.npz'
    # f = 'Approximant/Data/8Fold/AllCoherent/Data_R8_a3_c2.5_U0.15_N5_V0.0353_TEST6.npz'
    f = 'Approximant/Data/8Fold/Data_R8_a3_c2.5_U0.15_N5_V0.0354_TEST.npz'
    # data = np.load(f)
    # print(data['max_idx'])
    # curv_vals = data['curv_vals']
    # C = AC.calc_chern_multiband(curv_vals)
    # for i, val in enumerate(-np.round(C,5)):
    #     print(f"{i}: {val}")
    # plot_C_multiband(f, calc_new=False, scalefactor=-1)
    
    # C = data['C']
    # print(C)
    # qx = data['qx_vals']
    # print(qx)
    # curv = data['curv_vals']
    # print(curv.shape)


    plot_curv_contour(f, bands=None, 
                    plot_abs=False, plot_log=False, patch=None, chern=True,
                    inside_BZ=False, scale_factor=-1., shift_q=True, a=3)
    # for i in range(16):
    #     plot_curv_contour(f, bands=[i], 
    #                   plot_abs=False, plot_log=False, patch=None, chern=True,
    #                   inside_BZ=False, scale_factor=1., shift_q=True, a=1)
    
    # for i in range(12,21):
    #     plot_curv_contour('Curv_FullSurface_U200.0_V5.0_N3.npz', bands=[i], chern=True,
    #                       inside_BZ=True)
    
    # plot_curv_contour('Curv_FullSurface_Fukui.npz', bands=np.arange(8), 
    #                   plot_abs=False, plot_log=False, patch=None, chern=True)
    # for i in range(8):
        # plot_curv_contour('Curv_FullSurface_TKNN_unocc.npz', bands=[i], 
        #               plot_abs=False, plot_log=False, patch=None, chern=True)
        # plot_curv_contour('Curv_FullSurface_Fukui.npz', bands=[i], 
        #               plot_abs=False, plot_log=False, patch=None, chern=True)