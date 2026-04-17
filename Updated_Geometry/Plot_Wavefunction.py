import numpy as np
import matplotlib.pyplot as plt
import Calc_Bandstructure as CBS
from Plot_Spectral import make_title_str


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
    data = np.load(filename)
    psi_k = data['psi_k'][:, idx]
    E = data['evals'][idx]
    print(psi_k.shape)
    print(E.shape)
    k0 = data['k0']
    basis = data['basis']
    psi_r_up, psi_r_down = CBS.calc_psi_r(x_vals, y_vals, psi_k, basis, k0)
    n_r = np.sum(np.abs(psi_r_up)**2 + np.abs(psi_r_down)**2, axis=0)
    if normalise:
        n_r /= np.sum(np.abs(n_r) ** 2)
    n_max = np.max(n_r)
    n_min = np.min(n_r)
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




if __name__ == '__main__':
    f1 = 'Updated_Geometry/Data/WF_kG1_O5_c3.5_U1_V0_W0_N5_R8.npz'
    f2 = 'Updated_Geometry/Data/WF_kG_O4_c2.5_U10_V0_W0_N5_R8.npz'
    f3 = f'Updated_Geometry/Data/BlochVector_R8_U1_N5_V0.npz'
    xmax = 60
    dx = 0
    Nx = 301
    x_vals = np.linspace(-xmax+dx, xmax+dx, Nx)
    y_vals = np.copy(x_vals)
    data = np.load(f1)
    print(data['evals'])
    # psi_k = data['psi_k']
    # print(psi_k.shape)
    # I_k = np.sum(np.abs(psi_k)**4, axis=0)
    # print(f'I_k = {np.round(I_k,5)}')
    # I_loc = 1/psi_k.shape[0]
    # print(f'Localised = {I_loc:.4g}')
    # K: 28:32
    # M: 32:34
    plot_file_density(f1, x_vals, y_vals, idx=[0,1], normalise=False,
                      title_params={'U0':r'$U$', 'V0':r'$V$',
                                    'orders':r'$O$', 'cutoff':r'$k_{max}$'})
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