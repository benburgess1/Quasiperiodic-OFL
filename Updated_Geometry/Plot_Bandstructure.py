import numpy as np
import matplotlib.pyplot as plt
import os
# import Plot_Approximant_Curvature as PAC
import Calc_Bandstructure as CBS
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


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
                 E_R=1, E_lim=None, params_in_title=True, plot_BZ=False,
                 color=None, title_params={}):
    data = np.load(filename)
    q_vals = data['q_vals']
    E_vals = data['E_vals']
    E_vals = np.sort(E_vals, axis=0)
    N = E_vals.shape[0]
    N_q = E_vals.shape[1]
    fig,ax = plt.subplots()
    if x == 'qx':
        x_vals = q_vals[:,0]
        xlab = r'$k_{x}$ / $|\mathbf{G}|$'
    elif x == 'qy':
        x_vals = q_vals[:,1]
        xlab = r'$k_{y}$'
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
        title_str = make_title_str(title_params, data,
                                   base_str='', dp=5)
        ax.set_title(title_str)
    # if plot_title:
    #     if title_str is None:
    #         title_str = title_params(filename, bands=bands)
    #     elif params_in_title:
    #         title_str += ', ' + title_params(filename, bands=bands)
    #     # if title_str is None:
    #     #     U0 = data['U0']
    #     #     V0 = data['V0']
    #     #     N_wind = data['N']
    #     #     title_str = (r'$|U| = $' + str(U0) + ', ' 
    #     #                  + r'$|V| = $' + str(V0) + ', '
    #     #                  + r'$N = $' + str(N_wind) + ', ' 
    #     #                  + bandtit)
    #     ax.set_title(title_str)
    if bands is None:
        bands = np.arange(N)
    #     bandtit = 'All Bands'
    # elif len(bands) == 1:
    #     bandtit = f' Bands = {bands}'
    # else:
    #     bandtit = f'Bands = [{np.min(bands)}-{np.max(bands)}]'
    for n in bands:
        line = ax.plot(x_vals, E_vals[n,:]/E_R, label=f'n = {n}')
        if color is not None:
            # print(len(line))
            line[0].set_color(color)
    
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
                    q_path=None, plot_selected=False, use_index=False, 
                    title_params={}):
    data = np.load(filename)
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    if plot_selected:
        E_vals = data['selected_E_vals']
    elif use_index:
        E_vals = data['E_vals']
        idx1 = data['k0_up_max_idx']
        idx2 = data['k0_down_max_idx']
        # print(idx1.shape)
        print(idx1)
        E1 = CBS.slice_array(E_vals, idx1)
        E2 = CBS.slice_array(E_vals, idx2)
        E_vals = np.array([E1, E2])
    else:
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
        # title_str = title_params(filename, bands=bands)
        title_str = make_title_str(title_params, data, base_str='Extracted Eigenvalues' if use_index else 'Bandstructure')
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
            patch = mpl.patches.Polygon(xy=CBS.calc_pentagon(G=1, x=x, invert=True), 
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


def make_title_str(title_params, data, base_str='', dp=5):
    for k, v in title_params.items():
        try:
            val = data[k]
            if len(base_str) > 0:
                base_str += ', '
            try:
                base_str += v + r'$=$' + f'{val:.{dp}g}'
            except TypeError:
                base_str += v + r'$=$' + str(val)
        except KeyError as e:
            print('Warning: ' + k + f' not contained in data: {e}')
    return base_str


def plot_convergence(filename, title_params={}, x='N_basis', y='E', fit=None, p0=None,
                     plot_residuals=False, log_x=False, log_y=False,
                     weighted=True, sigma=None, eps=1e-10):
    """
    Plot convergence data from a .npz file with an optional fit.

    Parameters
    ----------
    filename : str
        Path to the .npz file.
    title_params : dict
        Parameters for the title string.
    x : str
        The x-axis variable. Either 'N_basis' or 'O'.
    y : str
        The y-axis variable. Either 'E' or 'IPR'.
    fit : str or None
        The type of fit to plot. Either 'polynomial' (E = E0 + a * x^-b),
        'exponential' (E = E0 + a * exp(-bx)), or None for no fit.
    p0 : array-like or None
        Initial guess for the fit parameters (E0, a, b). If None, defaults to
        (E_vals[-1], 1.0, 1.0).
    plot_residuals : bool
        If True, plot (E - E0) vs x instead of E vs x, where E0 is taken from
        the fit. Requires fit to be specified.
    log_x : bool
        If True, use a log scale on the x-axis.
    log_y : bool
        If True, use a log scale on the y-axis.
    weighted : bool
        If True, fit using weights sigma = |E - E_vals.min()| + 1e-10, which
        equalises relative errors across the data range. Ignored if sigma is
        provided.
    sigma : array-like or None
        Explicit per-point uncertainties passed to curve_fit. Overrides
        weighted. If None and weighted=False, all points are equally weighted.
    """
    data = np.load(filename, allow_pickle=True)
    E_vals = data['E_vals'][:, 0]

    if x == 'N_basis':
        x_data = data['N_basis']
        x_lab = r'$N_G$'
    elif x == 'O':
        x_data = data['O_vals']
        x_lab = r'$O$'
    print(x_data)
    print(E_vals)
    # print(data['cutoff'])

    if plot_residuals and fit is None:
        raise ValueError("plot_residuals=True requires a fit to be specified.")

    popt = None
    if fit is not None:
        if fit == 'polynomial':
            def model(x, E0, a, b):
                return E0 + a * x ** (-b)
            fit_label = r'Fit: $E_0 + a \cdot x^{-b}$'
            bounds = ([-np.inf, 0, 0], [E_vals.min(), np.inf, np.inf])
            default_p0 = (E_vals[-1], 1.0, 1.0)  # b~1 is fine for power law

        elif fit == 'exponential':
            def model(x, E0, a, b):
                return E0 + a * np.exp(-b * x)
            fit_label = r'Fit: $E_0 + a \cdot e^{-bx}$'
            bounds = ([-np.inf, 0, 0], [E_vals.min(), np.inf, np.inf])
            E0_guess = E_vals[-1]
            a_guess  = E_vals[0] - E0_guess
            ratio    = (E_vals[0] - E0_guess) / max(abs(E_vals[-1] - E0_guess), eps)
            b_guess  = np.log(max(ratio, eps)) / (x_data[-1] - x_data[0])
            default_p0 = (E0_guess, a_guess, b_guess)

        else:
            raise ValueError(f"fit must be 'polynomial', 'exponential', or None, got '{fit}'")
        
        if sigma is not None:
            fit_sigma = sigma
        elif weighted:
            fit_sigma = np.abs(E_vals - E_vals.min()) + eps
        else:
            fit_sigma = None
        
        p0 = p0 if p0 is not None else default_p0

        use_absolute_sigma = fit_sigma is not None

        try:
            popt, pcov = curve_fit(model, x_data, E_vals, p0=p0, bounds=bounds, maxfev=10000,
                                   sigma=fit_sigma, absolute_sigma=use_absolute_sigma)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError as e:
            print(f"Warning: curve_fit failed to converge: {e}")

    fig, ax = plt.subplots()
    x_fit = np.linspace(x_data.min(), x_data.max(), 300)

    if plot_residuals and popt is not None:
        E0 = popt[0]
        y_data = E_vals - E0
        y_fit  = model(x_fit, *popt) - E0
        y_lab  = r'$(E - E_0)$ / $E_R$'
    else:
        y_data = E_vals
        y_fit  = model(x_fit, *popt) if popt is not None else None
        y_lab  = r'$E$ / $E_R$'

    ax.plot(x_data, y_data, marker='x', ls='-', color='b', label='Data')

    if popt is not None:
        E0, a, b = popt
        ax.plot(x_fit, y_fit, color='r', ls='--', label=fit_label)

        fit_info = '\n'.join([
            r'$E_0 = $' + f'{E0:.4g} ± {perr[0]:.4g}',
            r'$a = $'   + f'{a:.4g} ± {perr[1]:.4g}',
            r'$b = $'   + f'{b:.4g} ± {perr[2]:.4g}',
        ])
        ax.text(0.97, 0.5, fit_info, transform=ax.transAxes,
                fontsize=9, va='center', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    title_str = make_title_str(title_params, data, base_str='Eigenvalue Convergence')
    ax.set_title(title_str)

    if fit is not None:
        ax.legend()

    plt.show()


def plot_ipr_convergence(filename, title_params={}, x='N_basis', fit=None, p0=None,
                         plot_residuals=False, log_x=False, log_y=False,
                         weighted=False, sigma=None, eps=1e-10, x_min_fit=None):
    """
    Plot IPR convergence data from a .npz file with an optional fit.

    Parameters
    ----------
    filename : str
        Path to the .npz file.
    title_params : dict
        Parameters for the title string.
    x : str
        The x-axis variable. Either 'N_basis' or 'O'.
    fit : str or None
        The type of fit to plot:
          - 'polynomial'        : IPR = a * x^b
          - 'polynomial_offset' : IPR = IPR0 + a * x^b
          - 'exponential'       : IPR = a * exp(b*x)
          - 'exponential_offset': IPR = IPR0 + a * exp(b*x)
          - None                : no fit
        Note: b < 0 in all cases (IPR decreases with x).
    p0 : array-like or None
        Initial guess for fit parameters:
          - 'polynomial'        : (a, b)
          - 'polynomial_offset' : (IPR0, a, b)
          - 'exponential'       : (a, b)
          - 'exponential_offset': (IPR0, a, b)
        If None, estimated from the data.
    plot_residuals : bool
        If True, plot (IPR - IPR0) vs x. Only valid for offset fits.
    log_x : bool
        If True, use a log scale on the x-axis.
    log_y : bool
        If True, use a log scale on the y-axis.
    weighted : bool
        If True, weight points by IPR_vals + eps, equalising relative errors.
        Ignored if sigma is provided.
    sigma : array-like or None
        Explicit per-point uncertainties passed to curve_fit. Overrides weighted.
    eps : float
        Small floor added to sigma to avoid division by zero (default 1e-10).
    x_min_fit : float or None
        If specified, only data points with x > x_min_fit are used for fitting.
        All data points are still plotted. If None, all data is used.
    """
    data = np.load(filename, allow_pickle=True)
    IPR_vals = data['IPR_vals'][:, 0]

    if x == 'N_basis':
        x_data = data['N_basis']
        x_lab = r'$N_G$'
    elif x == 'O':
        x_data = data['O_vals']
        x_lab = r'$O$'
    else:
        raise ValueError(f"x must be 'N_basis' or 'O', got '{x}'")

    # --- filter data for fitting ---
    if x_min_fit is not None:
        mask = x_data > x_min_fit
        if mask.sum() < 2:
            raise ValueError(f"x_min_fit={x_min_fit} leaves fewer than 2 data points for fitting.")
        x_fit_data   = x_data[mask]
        IPR_fit_vals = IPR_vals[mask]
    else:
        x_fit_data   = x_data
        IPR_fit_vals = IPR_vals

    offset_fits = {'polynomial_offset', 'exponential_offset'}
    if plot_residuals and fit not in offset_fits:
        raise ValueError("plot_residuals=True is only valid for offset fits "
                         "('polynomial_offset' or 'exponential_offset').")

    popt = None
    if fit is not None:

        # --- model and bounds ---
        # Note: bounds on IPR0 use IPR_fit_vals.min() so the constraint is
        # relative to the fitted subset, not the full dataset
        if fit == 'polynomial':
            def model(x, a, b):
                return a * x ** b
            fit_label = r'Fit: $a \cdot x^{b}$'
            bounds = ([0, -np.inf], [np.inf, 0])

        elif fit == 'polynomial_offset':
            def model(x, IPR0, a, b):
                return IPR0 + a * x ** b
            fit_label = r'Fit: $\mathrm{IPR}_0 + a \cdot x^{b}$'
            bounds = ([-np.inf, 0, -np.inf], [IPR_fit_vals.min(), np.inf, 0])

        elif fit == 'exponential':
            def model(x, a, b):
                return a * np.exp(b * x)
            fit_label = r'Fit: $a \cdot e^{bx}$'
            bounds = ([0, -np.inf], [np.inf, 0])

        elif fit == 'exponential_offset':
            def model(x, IPR0, a, b):
                return IPR0 + a * np.exp(b * x)
            fit_label = r'Fit: $\mathrm{IPR}_0 + a \cdot e^{bx}$'
            bounds = ([-np.inf, 0, -np.inf], [IPR_fit_vals.min(), np.inf, 0])

        else:
            raise ValueError(f"fit must be 'polynomial', 'polynomial_offset', "
                             f"'exponential', 'exponential_offset', or None. Got '{fit}'")

        # --- default p0 (estimated from fitting subset) ---
        if p0 is not None:
            p0_init = p0
        elif fit == 'polynomial':
            b_guess = (np.log(IPR_fit_vals[-1] / IPR_fit_vals[0]) /
                       np.log(x_fit_data[-1] / x_fit_data[0]))
            a_guess = IPR_fit_vals[0] / (x_fit_data[0] ** b_guess)
            p0_init = (a_guess, b_guess)

        elif fit == 'polynomial_offset':
            IPR0_guess = IPR_fit_vals[-1]
            b_guess    = np.log((IPR_fit_vals[-1] - IPR0_guess + eps) /
                                (IPR_fit_vals[0]  - IPR0_guess + eps)) / \
                         np.log(x_fit_data[-1] / x_fit_data[0])
            a_guess    = (IPR_fit_vals[0] - IPR0_guess) / (x_fit_data[0] ** b_guess)
            p0_init    = (IPR0_guess, a_guess, b_guess)

        elif fit == 'exponential':
            b_guess = (np.log(IPR_fit_vals[-1] / IPR_fit_vals[0]) /
                       (x_fit_data[-1] - x_fit_data[0]))
            a_guess = IPR_fit_vals[0] / np.exp(b_guess * x_fit_data[0])
            p0_init = (a_guess, b_guess)

        elif fit == 'exponential_offset':
            IPR0_guess = IPR_fit_vals[-1]
            b_guess    = np.log((IPR_fit_vals[-1] - IPR0_guess + eps) /
                                (IPR_fit_vals[0]  - IPR0_guess + eps)) / \
                         (x_fit_data[-1] - x_fit_data[0])
            a_guess    = (IPR_fit_vals[0] - IPR0_guess) / np.exp(b_guess * x_fit_data[0])
            p0_init    = (IPR0_guess, a_guess, b_guess)

        # --- sigma / weighting ---
        if sigma is not None:
            fit_sigma = sigma[mask] if x_min_fit is not None else sigma
        elif weighted:
            fit_sigma = IPR_fit_vals + eps
        else:
            fit_sigma = None

        use_absolute_sigma = fit_sigma is not None

        try:
            popt, pcov = curve_fit(model, x_fit_data, IPR_fit_vals, p0=p0_init,
                                   bounds=bounds, sigma=fit_sigma,
                                   absolute_sigma=use_absolute_sigma, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError as e:
            print(f"Warning: curve_fit failed to converge: {e}")

    # --- plotting ---
    fig, ax = plt.subplots()
    x_plot_fit = np.linspace(x_data.min(), x_data.max(), 300)

    if plot_residuals and popt is not None:
        IPR0   = popt[0]
        y_data = IPR_vals - IPR0
        y_fit  = model(x_plot_fit, *popt) - IPR0
        y_lab  = r'$(\mathrm{IPR} - \mathrm{IPR}_0)$'
    else:
        y_data = IPR_vals
        y_fit  = model(x_plot_fit, *popt) if popt is not None else None
        y_lab  = r'IPR'

    # plot excluded points differently so it's clear what was fitted
    if x_min_fit is not None:
        ax.plot(x_data[~mask], y_data[~mask], marker='x', ls='-', color='lightblue',
                label='Data (excluded from fit)')
        ax.plot(x_data[mask], y_data[mask], marker='x', ls='-', color='b',
                label='Data (fitted)')
    else:
        ax.plot(x_data, y_data, marker='x', ls='-', color='b', label='Data')

    if popt is not None:
        ax.plot(x_plot_fit, y_fit, color='r', ls='--', label=fit_label)

        if fit in {'polynomial', 'exponential'}:
            a, b = popt
            fit_info = '\n'.join([
                r'$a = $' + f'{a:.4g} ± {perr[0]:.4g}',
                r'$b = $' + f'{b:.4g} ± {perr[1]:.4g}',
            ])
        else:
            IPR0, a, b = popt
            fit_info = '\n'.join([
                r'$\mathrm{IPR}_0 = $' + f'{IPR0:.4g} ± {perr[0]:.4g}',
                r'$a = $'              + f'{a:.4g} ± {perr[1]:.4g}',
                r'$b = $'              + f'{b:.4g} ± {perr[2]:.4g}',
            ])

        ax.text(0.97, 0.5, fit_info, transform=ax.transAxes,
                fontsize=9, va='center', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    title_str = make_title_str(title_params, data, base_str='IPR Convergence')
    ax.set_title(title_str)

    if fit is not None or x_min_fit is not None:
        ax.legend()

    plt.show()


def generate_I0_vs_U_data(U_vals, x_min_fit=None, p0=None, eps=1e-3, sigma=None, weighted=False,
                          left_str='Updated_Geometry/Data/Convergence_IPR_k00_O1-7_c3.5_U', 
                          right_str='_V0_N5_R8.npz', save_filename='Data.npz'):
    filenames = [left_str + f'{U:.4g}' + right_str for U in U_vals]
    I0_vals = np.zeros(len(U_vals))
    a_vals = np.zeros(len(U_vals))
    b_vals = np.zeros(len(U_vals))
    I0_errs = np.zeros(len(U_vals))
    a_errs = np.zeros(len(U_vals))
    b_errs = np.zeros(len(U_vals))
    def model(x, IPR0, a, b):
            return IPR0 + a * x ** b
    for i, f in enumerate(filenames):
        plot_ipr_convergence(f, title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$', 'cutoff':r'$k_{max}$'}, x='N_basis',
                         fit='polynomial_offset', plot_residuals=False, log_y=False, log_x=False, weighted=False, eps=1e-3,
                         x_min_fit=500,
                         )
        data = np.load(f)
        x_data = data['N_basis']
        IPR_vals = data['IPR_vals'][:, 0]
        if x_min_fit is not None:
            mask = x_data > x_min_fit
            if mask.sum() < 2:
                raise ValueError(f"x_min_fit={x_min_fit} leaves fewer than 2 data points for fitting.")
            x_fit_data   = x_data[mask]
            IPR_fit_vals = IPR_vals[mask]
        else:
            x_fit_data   = x_data
            IPR_fit_vals = IPR_vals
        bounds = ([-np.inf, 0, -np.inf], [IPR_fit_vals.min(), np.inf, 0])
        if p0 is not None:
            p0_init = p0
        else:
            IPR0_guess = IPR_fit_vals[-1]
            b_guess    = np.log((IPR_fit_vals[-1] - IPR0_guess + eps) /
                                (IPR_fit_vals[0]  - IPR0_guess + eps)) / \
                         np.log(x_fit_data[-1] / x_fit_data[0])
            a_guess    = (IPR_fit_vals[0] - IPR0_guess) / (x_fit_data[0] ** b_guess)
            p0_init    = (IPR0_guess, a_guess, b_guess)

        if sigma is not None:
            fit_sigma = sigma[mask] if x_min_fit is not None else sigma
        elif weighted:
            fit_sigma = IPR_fit_vals + eps
        else:
            fit_sigma = None

        use_absolute_sigma = fit_sigma is not None

        try:
            popt, pcov = curve_fit(model, x_fit_data, IPR_fit_vals, p0=p0_init,
                                   bounds=bounds, sigma=fit_sigma,
                                   absolute_sigma=use_absolute_sigma, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            I0_vals[i] = popt[0]
            a_vals[i] = popt[1]
            b_vals[i] = popt[2]
            I0_errs[i] = perr[0]
            a_errs[i] = perr[1]
            b_errs[i] = perr[2]
        except RuntimeError as e:
            print(f"Warning: curve_fit failed to converge: {e}")
    np.savez(save_filename, U_vals=U_vals, I0_vals=I0_vals, a_vals=a_vals, b_vals=b_vals,
             I0_errs=I0_errs, a_errs=a_errs, b_errs=b_errs, p0=p0, x_min_fit=x_min_fit, 
             sigma=sigma, weighted=weighted, filenames=filenames)

        
def plot_I0_vs_U(filename):
    data = np.load(filename)
    U_vals = data['U_vals']
    I0_vals = data['I0_vals']
    I0_errs = data['I0_errs']
    fig, ax = plt.subplots()
    ax.errorbar(U_vals, I0_vals, yerr=I0_errs, marker='x', ls='-', color='b')
    ax.set_xlabel(r'$U$ / $E_R$')
    ax.set_ylabel(r'$I_0$')
    ax.set_title(r'Fitted $I_0$ vs $U$')
    plt.show()


def plot_spectrum_vs_U(filenames, num_evals=100, normalise_E=True, ms=5, title_params={},
                       zero_KE_filename=None, U_zero_KE=None):
    U_vals = np.zeros(len(filenames))
    E_vals = np.zeros((len(filenames), num_evals))
    for i, f in enumerate(filenames):
        data = np.load(f, allow_pickle=True)
        U_vals[i] = data['U0']
        E_vals[i,:] = data['evals'][:num_evals]
    if normalise_E:
        E_vals /= U_vals[:, np.newaxis]
        ylab = r'$E$ / $U$'
    else:
        ylab = r'$E$ / $E_R$'
    fig, ax = plt.subplots()
    ax.plot(U_vals, E_vals, ls='-', marker='o', ms=ms, color='b')
    if zero_KE_filename is not None:
        data = np.load(zero_KE_filename, allow_pickle=True)
        evals = data['evals']
        if U_zero_KE is None:
            U_zero_KE = 1.1 * np.max(U_vals)
        ax.plot(U_zero_KE*np.ones(num_evals), evals[:num_evals], marker='o', ls='', color='r', ms=ms)
    ax.set_xlabel(r'$U$ / $E_R$')
    ax.set_ylabel(ylab)
    title_str = make_title_str(title_params, data, base_str='Energy Spectrum')
    ax.set_title(title_str)
    plt.show()


def slice_array(arr, idxs):
    """Slice arr of shape (N_evals, Nx, Ny) down to (Nx, Ny) using index array idxs."""
    Nx, Ny = idxs.shape
    i, j = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    return arr[idxs, i, j]


def plot_energy_levels(
    filenames,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    marker_size=10,
    alpha=0.6,
    plot_fig=True,
    title_params={},
    ipr_log_scale=False,
    equally_spaced_x=False,
    x_parameter="U0",
    x_label=None,
    highlight_idx=None,
):
    """
    Plot energy levels coloured by IPR for one or more .npz data files,
    overlaid on a single plot with x-position given by a saved parameter.

    Parameters
    ----------
    filenames : str or list of str
        Path(s) to .npz file(s). Each file must contain:
            E_vals          : (N_evals, Nx, Ny)
            IPR_vals        : (N_evals, Nx, Ny)
            k0_up_max_idx   : (Nx, Ny)  index into first axis of E_vals / IPR_vals
            k0_down_max_idx : (Nx, Ny)  index into first axis of E_vals / IPR_vals
            U0              : scalar    default x-axis parameter
    ax : matplotlib.axes.Axes, optional
        Axes to plot onto. A new figure is created if None.
    cmap : str, optional
        Colormap name for IPR values (default "viridis").
    vmin, vmax : float, optional
        Color scale limits. If None, determined from all loaded data.
        When ipr_log_scale=True these should be given as raw IPR values
        (not log), e.g. vmin=1e-4, vmax=1.0.
    marker_size : float, optional
        Size of scatter points (default 10).
    alpha : float, optional
        Opacity of scatter points (default 0.6).
    plot_fig : bool, optional
        Whether to call plt.show() at the end (default True).
    title_params : dict, optional
        Passed to make_title_str to build the plot title.
    ipr_log_scale : bool, optional
        If True, colour the points on a logarithmic IPR scale using
        LogNorm instead of Normalize (default False).
    equally_spaced_x : bool, optional
        If True, files are plotted at equally spaced integer x positions
        (0, 1, 2, ...) sorted by x_parameter, with tick labels showing
        the parameter values. If False (default), the actual parameter
        values are used as x positions.
    x_parameter : str, optional
        Key in the .npz file to use as the x-axis parameter (default "U0").
        Must be a scalar in each file.
    x_label : str, optional
        x-axis label. If None, defaults to r"$U$ / $E_R$" when
        x_parameter="U0", otherwise uses the parameter name as-is.

    Returns
    -------
    fig, ax : the figure and axes objects.
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    # ------------------------------------------------------------------
    # First pass: load all data, optionally infer vmin/vmax
    # ------------------------------------------------------------------
    records = []
    all_ipr = []

    for fname in filenames:
        data = np.load(fname)
        E_vals   = data["E_vals"]           # (N_evals, Nx, Ny)
        IPR_vals = data["IPR_vals"]         # (N_evals, Nx, Ny)
        idx_up   = data["k0_up_max_idx"]    # (Nx, Ny)
        idx_dn   = data["k0_down_max_idx"]  # (Nx, Ny)
        x_val    = float(data[x_parameter])

        E_up   = slice_array(E_vals,   idx_up).ravel()
        E_dn   = slice_array(E_vals,   idx_dn).ravel()
        IPR_up = slice_array(IPR_vals, idx_up).ravel()
        IPR_dn = slice_array(IPR_vals, idx_dn).ravel()

        E_all   = np.concatenate([E_up,   E_dn])
        IPR_all = np.concatenate([IPR_up, IPR_dn])

        records.append((x_val, E_all, IPR_all, E_up, E_dn))
        all_ipr.append(IPR_all)

    # Sort records by x_val so equally-spaced ticks are in ascending order
    records.sort(key=lambda r: r[0])

    if vmin is None:
        vmin = min(arr.min() for arr in all_ipr)
    if vmax is None:
        vmax = max(arr.max() for arr in all_ipr)

    if ipr_log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap_obj = mpl.colormaps.get_cmap(cmap)

    # ------------------------------------------------------------------
    # Second pass: plot
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    for plot_idx, (x_val, E_all, IPR_all, E_up, E_dn) in enumerate(records):
        x_pos = plot_idx if equally_spaced_x else x_val
        x = np.full_like(E_all, x_pos)
        ax.scatter(
            x, E_all,
            c=IPR_all,
            cmap=cmap_obj,
            norm=norm,
            s=marker_size,
            alpha=alpha,
            linewidths=0,
        )
        if highlight_idx is not None:
            E_up_sorted = np.sort(E_up)
            E_dn_sorted = np.sort(E_dn)
            ax.scatter(x_pos, E_up_sorted[highlight_idx], marker='x', c='r', s=marker_size, alpha=alpha)
            ax.scatter(x_pos, E_dn_sorted[highlight_idx], marker='x', c='r', s=marker_size, alpha=alpha)

    if equally_spaced_x:
        tick_positions = np.arange(len(records))
        tick_labels = [str(x_val) for x_val, _, _, _, _ in records]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # Colorbar
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap_obj),
        ax=ax,
        label="IPR",
    )

    if x_label is None:
        x_label = r"$U$ / $E_R$" if x_parameter == "U0" else x_parameter
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$E$ / $E_R$")
    title_str = make_title_str(title_params, data, base_str='Energy levels coloured by IPR')
    ax.set_title(title_str)

    if plot_fig:
        plt.show()

    return fig, ax


def plot_ipr_vs_energy(
    filename,
    ax=None,
    color="steelblue",
    marker_size=10,
    alpha=0.6,
    ipr_log_scale=False,
    plot_fig=True,
    title_params={},
    x_label=r"$E$ / $E_R$",
    y_label=None,
):
    """
    Plot IPR vs energy for a single .npz file.

    The two sets of eigenvalues selected by k0_up_max_idx and k0_down_max_idx
    are treated identically and all (Nx·Ny) points are flattened and plotted.

    Parameters
    ----------
    filename : str
        Path to a .npz file containing:
            E_vals          : (N_evals, Nx, Ny)
            IPR_vals        : (N_evals, Nx, Ny)
            k0_up_max_idx   : (Nx, Ny)
            k0_down_max_idx : (Nx, Ny)
    ax : matplotlib.axes.Axes, optional
        Axes to plot onto. A new figure is created if None.
    color : str, optional
        Colour of the scatter points (default "steelblue").
    marker_size : float, optional
        Size of scatter points (default 10).
    alpha : float, optional
        Opacity of scatter points (default 0.6).
    ipr_log_scale : bool, optional
        If True, the IPR (y) axis is plotted on a log scale (default False).
    plot_fig : bool, optional
        Whether to call plt.show() at the end (default True).
    title_params : dict, optional
        Passed to make_title_str to build the plot title.
    x_label : str, optional
        x-axis label (default r"$E$ / $E_R$").
    y_label : str, optional
        y-axis label. If None, defaults to "IPR" or "IPR (log scale)"
        depending on ipr_log_scale.

    Returns
    -------
    fig, ax : the figure and axes objects.
    """
    # ------------------------------------------------------------------
    # Load and slice data
    # ------------------------------------------------------------------
    data    = np.load(filename)
    E_vals   = data["E_vals"]           # (N_evals, Nx, Ny)
    IPR_vals = data["IPR_vals"]         # (N_evals, Nx, Ny)
    idx_up   = data["k0_up_max_idx"]    # (Nx, Ny)
    idx_dn   = data["k0_down_max_idx"]  # (Nx, Ny)

    E_all   = np.concatenate([slice_array(E_vals,   idx_up).ravel(),
                               slice_array(E_vals,   idx_dn).ravel()])
    IPR_all = np.concatenate([slice_array(IPR_vals, idx_up).ravel(),
                               slice_array(IPR_vals, idx_dn).ravel()])

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.scatter(
        E_all, IPR_all,
        c=color,
        s=marker_size,
        alpha=alpha,
        linewidths=0,
    )

    if ipr_log_scale:
        ax.set_yscale("log")

    if y_label is None:
        y_label = "IPR (log scale)" if ipr_log_scale else "IPR"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title_str = make_title_str(title_params, data, base_str='IPR vs Energy')
    ax.set_title(title_str)

    if plot_fig:
        plt.show()

    return fig, ax


def plot_dos_IPR(
    filename,
    ax=None,
    gaussian_width=0.01,
    n_energy_points=1000,
    cmap="viridis",
    vmin=None,
    vmax=None,
    ipr_log_scale=False,
    linewidth=2,
    plot_fig=True,
    title_params={},
    x_label=r"$E$ / $E_R$",
    y_label="DoS (arb. units)",
):
    """
    Calculate and plot the density of states (DoS) from a single .npz file,
    with the curve coloured according to the IPR of the nearest eigenstate.

    The DoS is computed by summing Gaussians centred on each selected
    eigenvalue. The colour at each point on the DoS curve reflects the
    IPR of whichever eigenstate has the closest energy to that point.

    Only the two states selected by k0_up_max_idx and k0_down_max_idx
    are used (same selection as the other plotting functions).

    Parameters
    ----------
    filename : str
        Path to a .npz file containing:
            E_vals          : (N_evals, Nx, Ny)
            IPR_vals        : (N_evals, Nx, Ny)
            k0_up_max_idx   : (Nx, Ny)
            k0_down_max_idx : (Nx, Ny)
    ax : matplotlib.axes.Axes, optional
        Axes to plot onto. A new figure is created if None.
    gaussian_width : float, optional
        Standard deviation of the Gaussians used to broaden each
        eigenstate (in the same units as E_vals, default 0.01).
    n_energy_points : int, optional
        Number of points on the energy axis at which the DoS and
        IPR colour are evaluated (default 1000).
    cmap : str, optional
        Colormap name for the IPR colouring (default "viridis").
    vmin, vmax : float, optional
        IPR colour scale limits. If None, determined from the data.
        When ipr_log_scale=True, supply raw IPR values (not log).
    ipr_log_scale : bool, optional
        If True, colour the curve on a logarithmic IPR scale
        (default False).
    linewidth : float, optional
        Width of the DoS curve (default 2).
    plot_fig : bool, optional
        Whether to call plt.show() at the end (default True).
    title_params : dict, optional
        Passed to make_title_str to build the plot title.
    x_label : str, optional
        x-axis label (default r"$E$ / $E_R$").
    y_label : str, optional
        y-axis label (default "DoS (arb. units)").

    Returns
    -------
    fig, ax : the figure and axes objects.
    """
    # ------------------------------------------------------------------
    # Load and slice data
    # ------------------------------------------------------------------
    data     = np.load(filename)
    E_vals   = data["E_vals"]
    IPR_vals = data["IPR_vals"]
    idx_up   = data["k0_up_max_idx"]
    idx_dn   = data["k0_down_max_idx"]

    E_all   = np.concatenate([slice_array(E_vals,   idx_up).ravel(),
                               slice_array(E_vals,   idx_dn).ravel()])
    IPR_all = np.concatenate([slice_array(IPR_vals, idx_up).ravel(),
                               slice_array(IPR_vals, idx_dn).ravel()])

    # ------------------------------------------------------------------
    # Build energy axis with a small margin beyond the data range
    # ------------------------------------------------------------------
    margin  = 5 * gaussian_width
    E_grid  = np.linspace(E_all.min() - margin, E_all.max() + margin,
                           n_energy_points)

    # ------------------------------------------------------------------
    # Evaluate DoS at each point on E_grid
    # weights shape: (n_energy_points, N_states)
    # ------------------------------------------------------------------
    diff    = E_grid[:, np.newaxis] - E_all[np.newaxis, :]
    weights = np.exp(-0.5 * (diff / gaussian_width) ** 2)

    dos      = weights.sum(axis=1)
    dos_norm = dos / dos.max()

    # ------------------------------------------------------------------
    # Colour by IPR of the nearest eigenstate at each energy grid point
    # np.argmin over axis=1 gives index into E_all for each E_grid point
    # ------------------------------------------------------------------
    nearest_idx = np.argmin(np.abs(diff), axis=1)   # (n_energy_points,)
    ipr_nearest = IPR_all[nearest_idx]               # (n_energy_points,)

    # ------------------------------------------------------------------
    # Colour scale
    # ------------------------------------------------------------------
    if vmin is None:
        vmin = IPR_all.min()
    if vmax is None:
        vmax = IPR_all.max()

    if ipr_log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap_obj = mpl.colormaps.get_cmap(cmap)

    # ------------------------------------------------------------------
    # Plot: draw the curve as a collection of coloured line segments
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    points   = np.array([E_grid, dos_norm]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    mid_ipr  = 0.5 * (ipr_nearest[:-1] + ipr_nearest[1:])

    lc = LineCollection(segments, cmap=cmap_obj, norm=norm, linewidth=linewidth)
    lc.set_array(mid_ipr)
    ax.add_collection(lc)

    ax.set_xlim(E_grid[0], E_grid[-1])
    ax.set_ylim(0, 1.05)

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap_obj),
        ax=ax,
        label="IPR (log scale)" if ipr_log_scale else "IPR",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title_str = make_title_str(title_params, data, base_str='Density of States')
    ax.set_title(title_str)

    if plot_fig:
        plt.show()

    return fig, ax



if __name__ == '__main__':
    f = 'Updated_Geometry/Data/BS_surface_R8_O4_c3.5_U0.05_V0_N5_R8.npz'
    plot_ipr_vs_energy(f, color='b', title_params={'U0':r'$U$', 'V0':r'$V$', 'W':r'$W$', 'orders':r'$O$', 'cutoff':r'$k_{max}$'},
                       ipr_log_scale=False)
    plot_dos_IPR(f, gaussian_width=0.015, n_energy_points=1000, ipr_log_scale=False, 
                 title_params={'U0':r'$U$', 'V0':r'$V$', 'W':r'$W$', 'orders':r'$O$', 'cutoff':r'$k_{max}$'})
    # plot_BS_surface(f, use_index=True,
    #                 title_params={'U0':r'$U$', 'V0':r'$V$',
    #                               'orders':r'$O$', 'cutoff':r'$k_{max}$'})
    left_str = 'Updated_Geometry/Data/BS_surface_R8_O4_c3.5_U'
    right_str = '_V0_N5_R8.npz'
    U_vals = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]#, 1., 2., 5., 10.]
    filenames = [left_str + f'{U:.3g}' + right_str for U in U_vals]
    # plot_energy_levels(filenames, title_params={'V0':r'$V$', 'orders':r'$O$', 'cutoff':r'$k_{max}$'},
    #                    equally_spaced_x=True, vmax=1, ipr_log_scale=True, highlight_idx=184)
    # f = 'Updated_Geometry/Data/I0_U0.05-0.22_O1-7_c3.5.npz'
    # generate_I0_vs_U_data(U_vals=[0.05, 0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22],
    #                       x_min_fit=500, save_filename=f,
    #                       left_str='Updated_Geometry/Data/Convergence_IPR_k00_O1-7_c3.5_U', 
    #                       right_str='_V0_N5_R8.npz')
    # plot_I0_vs_U(f)
    # f = 'Updated_Geometry/Data/Convergence_IPR_k00_KE0_O1-6_cNone_U1_V0_N5_R8.npz'
    # plot_ipr_convergence(f, title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$', 'cutoff':r'$k_{max}$'}, x='N_basis',
    #                      fit='polynomial', plot_residuals=False, log_y=True, log_x=True, weighted=False, eps=1e-3,
    #                      x_min_fit=500,
    #                      )
    # f = 'Updated_Geometry/Data/Convergence_k00_O1-6_cNone_U0.5_V0_N5_R8.npz'
    # plot_convergence(f, title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$', 'cutoff':r'$k_{max}$'}, x='N_basis',
    #                  fit='polynomial', plot_residuals=True, log_y=True, log_x=True, weighted=False, eps=1e-10
    #                  )
    # f = 'Updated_Geometry/Data/BS_qx_R8_O7_c3.5_U0.3_V0_N5_R8.npz'
    # f = 'Updated_Geometry/Data/WF_k00_O5_cNone_U5_V0_W0_N5_R8.npz'
    # data = np.load(f)
    # E_vals = data['evals']
    # print(E_vals.shape) 
    # print(E_vals[:10])
    # U_vals = [0.05, 0.2, 0.5, 1., 1.5, 2., 2.5, 3., 5., 10.]
    # filenames = [f'Updated_Geometry/Data/WF_k00_O5_cNone_U{U:.3g}_V0_W0_N5_R8.npz' for U in U_vals]
    # plot_spectrum_vs_U(filenames, num_evals=100, title_params={'orders':r'$O$', 'cutoff':r'$k_{max}$'},
    #                    zero_KE_filename='Updated_Geometry/Data/WF_k00_KE0_O5_cNone_U1_V0_W0_N5_R8.npz')

    # print(E_vals[:20,0])
    # plot_BS_line(f, bands=np.arange(300), x='qx', E_lim=(-0.63, 0.3), color='b',
    #              title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$', 'orders':r'$O$', 'cutoff':r'$k_{max}$'},)
    # f = 'Updated_Geometry/Data/BS_Surface_b1_R8_U20.0_N5_V0.0_.npz'
    # plot_BS_surface(f, plot_selected=True)

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
    # f = 'Updated Geometry/Data/RQBZData_o1_U0.0_N3_V0.0_finer.npz'
    # f = 'Updated Geometry/Data/RQBZData_o8_c3.5_U0.2_N3_V0.15_extended.npz'
    # data = np.load(f)
    # q_max = np.max(data['qx_vals'])
    # print(f'q_max = {q_max}')
    # N = data['qx_vals'].size
    # print(f'Size: {N} x {N}')
    # q = 0.14589803
    # N_bands = 202
    # # x = (np.sqrt(5)-1)/(2*1)
    # x = (np.sqrt(5)-1)/(2*np.sqrt(N_bands/2))
    # print(f'RQBZ radius = {x}')
    # patch = mpl.patches.Polygon(xy=CBS.calc_pentagon(G=q, x=x, invert=False), 
    #                                     edgecolor='k', facecolor=(0,0,0,0))
    # q_path = patch.get_path()
    # dE = 0.01
    # dk = 2*q_max / (N-1)
    # g = (2*np.pi / dk**2)
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
    # f7_1 = 'Updated Geometry/Data/RQBZData_o7_c3.5_U0.02_N3_V0.01.npz'
    # f8_1 = 'Updated Geometry/Data/RQBZData_o8_c3.5_U0.02_N3_V0.01.npz'
    # # f9 = 'Updated Geometry/Data/RQBZData_o9_c3.5_U0.02_N3_V0.01.npz'
    # # f2 = 'Updated Geometry/Data/RQBZData_o2_U0.0_N3_V0.0.npz'
    # # f3 = 'Updated Geometry/Data/RQBZData_o3_U0.0_N3_V0.0.npz'
    # f4 = 'Updated Geometry/Data/RQBZData_o4_U0.0_N3_V0.0.npz'
    # f5 = 'Updated Geometry/Data/RQBZData_o5_c3.5_U0.0_N3_V0.0.npz'
    # f5_old = 'Updated Geometry/Data/RQBZData_o5_c3.5_U0.0_N3_V0.0_old.npz'
    # f6 = 'Updated Geometry/Data/RQBZData_o6_c3.5_U0.0_N3_V0.0.npz'
    # f6_old = 'Updated Geometry/Data/RQBZData_o6_c3.5_U0.0_N3_V0.0_old.npz'
    # f7 = 'Updated Geometry/Data/RQBZData_o7_c3.5_U0.0_N3_V0.0.npz'
    # f8 = 'Updated Geometry/Data/RQBZData_o8_c3.5_U0.0_N3_V0.0.npz'
    # f9 = 'Updated Geometry/Data/RQBZData_o9_c3.5_U0.0_N3_V0.0.npz'

    # f2 = 'Updated Geometry/Data/TestData_o2_U0.2_N3_V0.15.npz'
    # f3 = 'Updated Geometry/Data/TestData_o3_U0.2_N3_V0.15.npz'
    # f4 = 'Updated Geometry/Data/TestData_o4_U0.2_N3_V0.15.npz'
    # f5 = 'Updated Geometry/Data/TestData_o5_c3.5_U0.2_N3_V0.15.npz'
    # f6 = 'Updated Geometry/Data/TestData_o6_c3.5_U0.2_N3_V0.15.npz'
    # f7 = 'Updated Geometry/Data/TestData_o7_c3.5_U0.2_N3_V0.15.npz'
    # f8 = 'Updated Geometry/Data/TestData_o8_c3.5_U0.2_N3_V0.15.npz'
    # f9 = 'Updated Geometry/Data/TestData_o9_c3.5_U0.2_N3_V0.15.npz'
    # f10 = 'Updated Geometry/Data/TestData_o10_c3.5_U0.2_N3_V0.15_updated.npz'

    # filenames = [f4, f5, f6, f7, f8, f9, f10]
    # orders = [4, 5, 6, 7, 8, 9, 10]
    # filenames = [f7, f8, f9, f10]
    # orders = [7, 8, 9, 10]
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
