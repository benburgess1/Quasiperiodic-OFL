import numpy as np
import matplotlib.pyplot as plt
import Calc_Bandstructure as CBS
from Plot_Bandstructure import make_title_str
from scipy.optimize import curve_fit


def plot_psi_r(psi_r, x_vals, y_vals, component='abs2', normalise=False):
    """
    Plot a 2D wave function as a colormesh.

    Parameters
    ----------
    psi_r       : 2D complex array of shape (Nx, Ny)
    x_vals      : 1D array of x coordinates
    y_vals      : 1D array of y coordinates
    component   : 'real', 'imag', or 'abs2'
    normalise   : if True, normalise so that sum(|psi_r|^2) = 1
    """
    if normalise:
        psi_r = psi_r / np.sqrt(np.sum(np.abs(psi_r) ** 2))

    if component == 'real':
        data = np.real(psi_r)
        cmap = 'bwr'
        label = r'$\mathrm{Re}(\psi)$'
    elif component == 'imag':
        data = np.imag(psi_r)
        cmap = 'bwr'
        label = r'$\mathrm{Im}(\psi)$'
    elif component == 'abs2':
        data = np.abs(psi_r) ** 2
        cmap = 'plasma'
        label = r'$|\psi|^2$'
    else:
        raise ValueError(f"component must be 'real', 'imag', or 'abs2', got '{component}'")

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(x_vals, y_vals, data.T, cmap=cmap, shading='auto')

    # Centre bwr colormaps symmetrically around zero
    if component in ('real', 'imag'):
        vmax = np.max(np.abs(data))
        mesh.set_clim(-vmax, vmax)
    else:
        vmax = np.max(np.abs(data))
        mesh.set_clim(0, vmax)

    plt.colorbar(mesh, ax=ax, label=label)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal')
    plt.show()

    return fig, ax 


def plot_density(n_r, x_vals, y_vals, psi_r_up=None, psi_r_down=None, normalise=False, cmap='plasma',
                 plot_fig=True):
    """
    Plot a 2D density as a colormesh.

    Parameters
    ----------
    n_r         : 2D array of shape (Nx, Ny)
    x_vals      : 1D array of x coordinates
    y_vals      : 1D array of y coordinates
    psi_r_up    : None or 2D complex array of shape (Nx, Ny). If not None, used to calculate n_r
    psi_r_down  : None or 2D complex array of shape (Nx, Ny). If not None, used to calculate n_r
    normalise   : if True, normalise so that sum(|psi_r|^2) = 1
    """
    if psi_r_up is not None and psi_r_down is not None:
        n_r = np.sum(np.abs(psi_r_up)**2 + np.abs(psi_r_down)**2, axis=0)
    if normalise:
        n_r = n_r / np.sum(np.abs(n_r) ** 2)
    data = n_r
    label = r'$|\psi(\mathbf{r})|^2$'

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(x_vals, y_vals, data.T, cmap=cmap, shading='auto')

    vmax = np.max(np.abs(data))
    mesh.set_clim(0, vmax)

    plt.colorbar(mesh, ax=ax, label=label)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal')
    if plot_fig:
        plt.show()

    return fig, ax


def plot_file_density(filename, x_vals, y_vals, idx=[0], title_params={},
                      title_density=False, normalise=True):
    data = np.load(filename, allow_pickle=True)
    psi_k = data['psi_k'][:, idx]
    E = data['evals'][idx]
    print(psi_k.shape)
    print(E.shape)
    k0 = data['k0']
    basis = data['basis']
    psi_r_up, psi_r_down = CBS.calc_psi_r(x_vals, y_vals, psi_k, basis, k0)
    print(psi_r_up.shape)
    n_r = np.sum(np.abs(psi_r_up)**2 + np.abs(psi_r_down)**2, axis=0)
    if normalise:
        n_r /= np.sum(np.abs(n_r) ** 2)
    n_max = np.max(n_r)
    n_min = np.min(n_r)
    j, k = np.unravel_index(np.argmax(n_r), n_r.shape)
    print(f'(x0, y0) = ({x_vals[j]:.4g}, {y_vals[k]:.4g})')
    fig, ax = plot_density(n_r, x_vals, y_vals, normalise=False, plot_fig=False)
    base_str = r'$k_0 = $' + str(np.round(k0, 2)) + r', $E = $' + str(np.round(E, 4))
    if title_density:
        base_str += r', $|\psi|^2_{max} = $' + f'{n_max:.3g}, ' + r'$|\psi|^2_{min} = $' + f'{n_min:.3g}'
    title_str = make_title_str(title_params, data, 
                               base_str=base_str, dp=4)
    ax.set_title(title_str)
    plt.show()


def plot_density_vs_potential(x_vals, y, density_filenames, potential_filename, idx=[0],
                              colors=['b', 'c', 'r'], labels=None, normalise=False,
                              V_factor=1):
    fig, ax = plt.subplots()
    for i, f in enumerate(density_filenames):
        data = np.load(f)
        psi_k = data['psi_k'][:,idx]
        k0 = data['k0']
        basis = data['basis']
        psi_r_up, psi_r_down = CBS.calc_psi_r(x_vals, np.array([y]), psi_k, basis, k0)
        n_r = np.sum(np.abs(psi_r_up)**2 + np.abs(psi_r_down)**2, axis=0)[:,0]
        if normalise:
            n_r /= np.sum(np.abs(n_r))
        if labels is None:
            label = r'$U=$' + f'{data['U0']:.3g}'
        else:
            label = labels[i]
        ax.plot(x_vals, n_r, color=colors[i], label=label, ls='-', marker=None)
    
    ax2 = ax.twinx()
    data = np.load(potential_filename)
    y_idx = np.argmin(np.abs(data['y_vals']-y))
    print(y_idx)
    x = data['x_vals']
    V = data['B_vals'][:, y_idx]
    # V *= V_factor
    # V /= np.max(V)
    # V /= 2
    ax2.plot(x, V, color=colors[-1], label=r'$|V(\mathbf{r})|$')
    ax2.set_ylim(top=np.max(V)*V_factor)

    ax.legend()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$|\psi(\mathbf{r})|^2$')
    ax2.set_ylabel(r'$|V(\mathbf{r})|$', color='r')

    ax.set_title(f'Density vs Potential Magnitude, y = {y:.3g}')

    plt.show()


def plot_density_slice(filename, x_vals, x0, y0, normalise=True, idx=[0], centre_x=True,
                       title_params={}, fit=None):
    """
    Plot density vs position data with an optional fit.

    Parameters
    ----------
    filename : str
        Path to the .npz file.
    x_vals : array-like
        x values at which to evaluate the density.
    x0 : float
        x position of the density peak (used for centring).
    y0 : float
        y position at which to take the density slice.
    normalise : bool
        If True, normalise so that n(0) = 1. Forced True if fit is specified.
    idx : list of int
        Indices of eigenstates to plot.
    centre_x : bool
        If True, plot x - x0. Forced True if fit is specified.
    title_params : dict
        Parameters for the title string.
    fit : str or None
        The type of fit to plot:
          - 'gaussian'  : n = exp(-(x/xi)^2)
          - 'lorentzian': n = 1/((x/xi)^2 + 1)
          - None        : no fit
    """
    data = np.load(filename, allow_pickle=True)
    psi_k = data['psi_k'][:, idx]
    E = data['evals'][idx]

    print(psi_k.shape)
    print(E.shape)

    k0 = data['k0']
    basis = data['basis']
    psi_r_up, psi_r_down = CBS.calc_psi_r(x_vals, np.array([y0]), psi_k, basis, k0)
    n_r = np.sum(np.abs(psi_r_up)**2 + np.abs(psi_r_down)**2, axis=0)[:, 0]

    if fit is not None and not normalise:
        print("Warning: normalise forced to True since fit is specified.")
    if fit is not None and not centre_x:
        print("Warning: centre_x forced to True since fit is specified.")

    normalise = True if fit is not None else normalise
    centre_x  = True if fit is not None else centre_x

    if normalise:
        n_r /= np.max(n_r)

    x_plot = x_vals - x0 if centre_x else x_vals

    # --- fitting ---
    popt = None
    if fit is not None:
        if fit == 'gaussian':
            def model(x, xi):
                return np.exp(-(x / xi)**2)
            fit_label = r'Fit: $e^{-(x/\xi)^2}$'

        elif fit == 'lorentzian':
            def model(x, xi):
                return 1 / ((x / xi)**2 + 1)
            fit_label = r'Fit: $1/((x/\xi)^2 + 1)$'

        else:
            raise ValueError(f"fit must be 'gaussian', 'lorentzian', or None, got '{fit}'")

        # xi > 0; initial guess: half the plot range
        bounds  = ([0], [np.inf])
        p0_init = [np.max(x_plot) / 2]

        try:
            popt, pcov = curve_fit(model, x_plot, n_r, p0=p0_init, bounds=bounds,
                                   maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError as e:
            print(f"Warning: curve_fit failed to converge: {e}")

    # --- plotting ---
    fig, ax = plt.subplots()

    ax.plot(x_plot, n_r, color='b', marker='', ls='-', label='Data')

    if popt is not None:
        xi = popt[0]
        x_fit = np.linspace(x_plot.min(), x_plot.max(), 300)
        ax.plot(x_fit, model(x_fit, xi), color='r', ls='--', label=fit_label)

        fit_info = r'$\xi = $' + f'{xi:.4g} ± {perr[0]:.4g}'
        ax.text(0.97, 0.5, fit_info, transform=ax.transAxes,
                fontsize=9, va='center', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    xlab = r'$x - x_0$' if centre_x else r'$x$'
    ylab = r'$n(x-x_0, y_0)$' if centre_x else r'$n(x, y_0)$'
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    title_str = make_title_str(title_params, data, base_str='Density at Peak')
    ax.set_title(title_str)

    if fit is not None:
        ax.legend()

    plt.show()



def generate_xi_vs_O_data(O_vals, x_vals, x0, y0,
                          left_str='Updated_Geometry/Data/WF_k00_KE0_O', 
                          right_str='_cNone_U1_V0_W0_N5_R8.npz', save_filename='Data.npz',
                          idx=[0,1]):
    filenames = [left_str + str(O) + right_str for O in O_vals]
    xi_vals = np.zeros(len(O_vals))
    xi_errs = np.zeros(len(O_vals))
    def model(x, xi):
        return np.exp(-(x / xi)**2)
    for i, f in enumerate(filenames):
        data = np.load(f, allow_pickle=True)
        psi_k = data['psi_k'][:, idx]
        E = data['evals'][idx]
        k0 = data['k0']
        basis = data['basis']
        psi_r_up, psi_r_down = CBS.calc_psi_r(x_vals, np.array([y0]), psi_k, basis, k0)
        n_r = np.sum(np.abs(psi_r_up)**2 + np.abs(psi_r_down)**2, axis=0)[:, 0]
        n_r /= np.max(n_r)
        x_plot = x_vals - x0

        bounds  = ([0], [np.inf])
        p0_init = [np.max(x_plot) / 2]

        try:
            popt, pcov = curve_fit(model, x_plot, n_r, p0=p0_init, bounds=bounds,
                                   maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            xi_vals[i] = popt[0]
            xi_errs[i] = perr[0]
        except RuntimeError as e:
            print(f"Warning: curve_fit failed to converge for O = {O_vals[i]}: {e}")
        
    np.savez(save_filename, O_vals=O_vals, xi_vals=xi_vals, xi_errs=xi_errs,
             x_vals=x_vals, x0=x0, y0=y0, filenames=filenames, idx=idx)


def plot_xi_vs_O(filename, title_params={}, logx=False, logy=False, fit=None, weighted=False):
    data = np.load(filename)
    O_vals = data['O_vals']
    xi_vals = data['xi_vals']
    xi_errs = data['xi_errs']
    fig, ax = plt.subplots()
    ax.errorbar(O_vals, xi_vals, yerr=xi_errs, color='b', marker='x', ls='-', label='Data')
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if fit == 'polynomial':
        def model(x, a, b):
            return a * x**b
        p0_init = [2*np.pi, -1]
        bounds = ([0, -np.inf], [np.inf, 0])
        sigma = data['xi_errs'] if weighted else None
        absolute_sigma = sigma is not None
        try:
            popt, pcov = curve_fit(model, O_vals, xi_vals, p0=p0_init, bounds=bounds,
                                   sigma=sigma, absolute_sigma=absolute_sigma, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            x_plot = np.linspace(min(O_vals), max(O_vals), 100)
            ax.plot(x_plot, model(x_plot, *popt), color='r', ls='--', label=r'Fit: $\xi = a \cdot O^{-b}$')
            ax.legend()
            fit_info = r'$a = $' + f'{popt[0]:.4g} ± {perr[0]:.4g}' + '\n' + r'$b = $' + f'{popt[1]:.4g} ± {perr[1]:.4g}'
            ax.text(0.97, 0.5, fit_info, transform=ax.transAxes,
                    fontsize=9, va='center', ha='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        except RuntimeError as e:
            print(f"Warning: curve_fit failed to converge for: {e}")
        
    ax.set_xlabel(r'$O$')
    ax.set_ylabel(r'$\xi$')
    title_str = make_title_str(title_params, data, base_str='Localisation Length')
    ax.set_title(title_str)
    plt.show()



if __name__ == '__main__':
    x0 = 26.67
    y0 = 11.05
    xmax = 1.5
    Nx = 301
    dx = np.linspace(-xmax, xmax, Nx)
    x_vals = dx + x0
    f = 'Updated_Geometry/Data/WF_k00_KE0_O4_cNone_U1_V0_W0_N5_R8.npz'
    # plot_density_slice(f, x_vals, x0, y0, normalise=True, idx=[6,7], centre_x=True, 
    #                    title_params={'U0':r'$U$', 'V0':r'$V$',
    #                                  'orders':r'$O$', 'cutoff':r'$k_{max}$'},
    #                    fit='gaussian')

    O_vals = [3, 4, 5, 6, 7]
    save_f = 'Updated_Geometry/Data/Xi_O3-7_cNone_KE0.npz'
    # generate_xi_vs_O_data(O_vals, x_vals=x_vals, x0=x0, y0=y0, idx=[0,1],
    #                       left_str='Updated_Geometry/Data/WF_k00_KE0_O',
    #                       right_str='_cNone_U1_V0_W0_N5_R8.npz',
    #                       save_filename=save_f)
    plot_xi_vs_O(save_f, title_params={'cutoff':r'$k_{max}$'}, fit='polynomial',
                 logx=True, logy=True, weighted=True)



    f1 = 'Updated_Geometry/Data/WF_kG1_O5_c3.5_U1_V0_W0_N5_R8.npz'
    f1 = 'Updated_Geometry/Data/WF_k00_O5_cNone_U3_V0_W0_N5_R8.npz'
    f1 = 'Updated_Geometry/Data/WF_k00_KE0_O5_cNone_U1_V0_W0_N5_R8.npz'
    f2 = 'Updated_Geometry/Data/WF_kG_O4_c2.5_U10_V0_W0_N5_R8.npz'
    f3 = f'Updated_Geometry/Data/BlochVector_R8_U1_N5_V0.npz'
    xmax = 60
    dx = 0
    Nx = 101
    x_vals = np.linspace(-xmax+dx, xmax+dx, Nx)
    y_vals = np.copy(x_vals)
    # data = np.load(f1)
    # print(data['evals'][:10])
    # psi_k = data['psi_k']
    # print(psi_k.shape)
    # I_k = np.sum(np.abs(psi_k)**4, axis=0)
    # print(f'I_k = {np.round(I_k,5)}')
    # I_loc = 1/psi_k.shape[0]
    # print(f'Localised = {I_loc:.4g}')
    # K: 28:32
    # M: 32:34
    # x_vals = np.linspace(25, 28, Nx)
    # y_vals = np.linspace(9.5, 12.5, Nx)
    # print(x_vals[1] - x_vals[0])
    # plot_file_density(f1, x_vals, y_vals, idx=[0,1], normalise=False,
    #                   title_params={'U0':r'$U$', 'V0':r'$V$',
    #                                 'orders':r'$O$', 'cutoff':r'$k_{max}$'})
    
    # plot_density_vs_potential(x_vals, y=-26.5, density_filenames=[f1, f2], potential_filename=f3, idx=[0],
    #                           V_factor=2)
    # R = 8
    # l = np.arange(R)
    # G_vects = np.column_stack((np.cos(2*np.pi*l/R),
    #                                 np.sin(2*np.pi*l/R)))
    # g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    # k0 = np.array([0.5, 0.])
    # basis = CBS.calc_basis_states(orders=1, cutoff=None, basis=None,
    #                               G_vects=G_vects, g_vects=g_vects, print_progress=False)
    # N_k = basis[0].shape[0]
    # print(basis[0])
    # psi_k = np.zeros(2*N_k)
    # psi_k[0] = 1 / np.sqrt(2)
    # psi_k[1] = 1 / np.sqrt(2)
    # xmax = 10
    # Nx = 100
    # x_vals = np.linspace(-xmax, xmax, Nx)
    # y_vals = np.copy(x_vals)
    # psi_r_up, psi_r_down = CBS.calc_psi_r(x_vals, y_vals, psi_k, basis, k0)
    # plot_psi_r(psi_r_up, x_vals, y_vals, component='abs2', normalise=True)