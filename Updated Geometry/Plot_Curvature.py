import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import Calc_Bandstructure as CBS
import Calc_Curvature as CC



# Generate colormap to use later when plotting
bwr = plt.cm.get_cmap('bwr')
upper_half_bwr = mcolors.LinearSegmentedColormap.from_list(
    'upper_half_bwr', bwr(np.linspace(0.5, 1, 256)))


max_idx_dict = {1:1, 2:3, 3:15, 4:27}


def plot_curv_contour(filename, bands=None, plot_title=True, 
                      title_str=None, real=True, cmap='default',
                      plot_cbar=True, plot_QBZ=True, plot_abs=False, 
                      plot_log=False, patch=None, chern=False, inside_QBZ=False,
                      invert=False, 
                      scale_factor=1., axlim=None, shift_q=False,
                      max_idx_dict=max_idx_dict, ax=None, show_plot=True,
                      levels=None):
    data = np.load(filename)
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    if shift_q:
        dqx = qx_vals[1] - qx_vals[0]
        qx_vals = (qx_vals + dqx/2)[:-1]
        dqy = qy_vals[1] - qy_vals[0]
        qy_vals = (qy_vals + dqy/2)[:-1]
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    curv_vals = data['curv_vals'] / scale_factor
    if curv_vals.shape[1:] != qxx.shape:
        curv_vals = curv_vals[:, :qxx.shape[0], :qxx.shape[1]]
    if inside_QBZ:
        QBZ_path = mpl.path.Path(vertices=CBS.calc_pentagon(G=1.0001, invert=invert))     # Enlarge infinitesimally so no boundary issues
        points = np.column_stack((qxx.ravel(), qyy.ravel()))
        mask = QBZ_path.contains_points(points).reshape(qxx.shape)
        curv_plot = np.where(mask[None, :, :], curv_vals, 0)

        # QBZ_path = AC.square_BZ(a=0.999*a).get_path()
        # points = np.column_stack((qxx.ravel(), qyy.ravel()))
        # mask = BZ_path.contains_points(points)
        # curv_plot = np.full_like(curv_vals, 0.)
        # curv_plot[:, mask.reshape(qxx.shape)] = curv_vals[:, mask.reshape(qxx.shape)]
        # print(curv_plot.shape)
        # curv_plot = curv_plot.reshape((curv_plot.shape[0], *qxx.shape))
    elif patch is not None:
        path = patch.get_path()
        points = np.column_stack((qxx.ravel(), qyy.ravel()))
        mask = path.contains_points(points).reshape(qxx.shape)
        curv_plot = np.where(mask[None, :, :], curv_vals, 0)
    else:
        curv_plot = curv_vals
    if ax is None:
        fig,ax = plt.subplots()
    if plot_title:
        title_str = title_params(filename, bands=bands)
        if chern:
            # print(curv_vals.shape)
            C = CC.calc_chern_number(curv_vals=curv_vals, qx_vals=qx_vals, 
                                     qy_vals=qy_vals, bands=bands, 
                                     inside_QBZ=inside_QBZ, invert=invert, shift_q=False,
                                     patch=patch)
            C_str = str(np.round(np.real(C),5))
            title_str += ', ' + r'$C = $' + C_str
        ax.set_title(title_str)
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
        levels = np.linspace(vmin,vmax,200)
        ticks = [vmin,vmax]
    else:
        if levels is None:
            vmax = np.max(np.abs(curv_sum))
            if plot_abs:
                levels = np.linspace(0,vmax,200)
                ticks = [0,vmax]
            else:
                levels = np.linspace(-vmax,vmax,200)
                ticks = [-vmax,0,vmax]
    # print(levels)
    plot = ax.contourf(qxx, qyy, curv_sum, cmap=cmap, levels=levels)
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$', rotation=0)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab, rotation=0)
    if plot_QBZ:
        QBZ_patch = mpl.patches.Polygon(xy=CBS.calc_pentagon(G=1.0, invert=invert), 
                                        edgecolor='k', facecolor=(0,0,0,0))
        ax.add_patch(QBZ_patch)
    if patch is not None:
        ax.add_patch(patch)
    if axlim is None:
        ax.set_xlim(np.min(qx_vals), np.max(qx_vals))
        ax.set_ylim(np.min(qy_vals), np.max(qy_vals))
    else:
        ax.set_xlim(*axlim)
        ax.set_ylim(*axlim)
    if show_plot:
        plt.show()
    else:
        return plot


def title_params(filename, include_bands=True, bands=None):
    data = np.load(filename)
    title_str = ''
    params = {'U0':r'$|U|$', 'V0':r'$|V|$', 'N':r'$N$'}
    for key, val in params.items():
        if key in data:
            title_str += val + r'$ = $' + str(np.round(data[key],4)) + ', '
    if include_bands:
        if bands is None:
            if 'max_idx' in data:
                idx = data['max_idx']
                bandtit = f'Bands = [0-{idx}]'
            else:
                bandtit = 'All Bands'
        elif len(bands) == 1:
            bandtit = f' Bands = {bands}'
        else:
            bandtit = f'Bands = [{np.min(bands)}-{np.max(bands)}]'
        title_str += bandtit
    else:
        title_str = title_str[:-2]
    return title_str
    


if __name__ == '__main__':
    # f = -(0.6/300)**2
    # f = 'Data/Curv_approx_a3_Fukui_U200.0_N4_V150.0.npz'
    # f = 'Data/Data_a3_U0.125_N3_V0.05.npz'

    # f = 'Data/Data_a3_U0.2_N3_V0.1.npz'
    # f = 'Approximant/Data/Data_NonAb_a4_U0.3_N3_V0.3.npz'
    # f = 'Updated Geometry/Data/TestData_o10_c3.5_U0.2_N3_V0.15_updated.npz'
    f = 'Updated Geometry/Data/RQBZData_o5_c3.5_U0.2_N3_V0.001_altbasis.npz'
    # data = np.load(f)
    # C = data['C']
    # print(C)
    # qx = data['qx_vals']
    # print(qx)
    # curv = data['curv_vals']
    # print(curv.shape)
    data = np.load(f)
    C = data['C']
    print(f'C = {C}')
    q_max = np.max(data['qx_vals'])
    print(f'q_max = {q_max}')
    q = 0.14589803
    N_bands = 126
    k = np.sqrt(N_bands/2)
    x = (np.sqrt(5)-1)/(2*k)
    print(f'RQBZ radius = {x}')
    patch = mpl.patches.Polygon(xy=CBS.calc_pentagon(G=q, x=x, invert=True), 
                                        edgecolor='k', facecolor=(0,0,0,0))

    # plot_curv_contour(f, bands=None, 
    #                   plot_abs=False, plot_log=False, patch=patch, chern=True,
    #                   inside_QBZ=False, invert=False, scale_factor=1., shift_q=True,
    #                   axlim=(-0.2,0.2))
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