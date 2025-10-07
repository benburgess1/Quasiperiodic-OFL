import numpy as np
import matplotlib.pyplot as plt


def plot_curv(filename, cmap=plt.colormaps['bwr'], levels=None,
                     axlim=None, plot_cbar=True, plot_title=True,
                     xticks=None, yticks=None, ticks=None, norm=None, extend=None,
                     mean_in_title=True, dp=5):
    data = np.load(filename)
    x_vals = data['x_vals']
    y_vals = data['y_vals']
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    curv = np.real(data['curv_vals'])
    fig, ax = plt.subplots()
    if levels is None:
        vmax = np.max(np.abs(curv))
        levels = np.linspace(-vmax, vmax, 200)
    if ticks is None:
        ticks = [np.min(levels), 0, np.max(levels)]
    plot = ax.contourf(xx, yy, curv, cmap=cmap, levels=levels, norm=norm, extend=extend)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(r'$B_{eff}(\mathbf{r})$', rotation=0)
    ax.set_xlabel(r'$x|\mathbf{q}|$')
    ax.set_ylabel(r'$y|\mathbf{q}|$')
    if axlim is None:
        ax.set_xlim(np.min(x_vals), np.max(x_vals))
        ax.set_ylim(np.min(y_vals), np.max(y_vals))
    else:
        ax.set_xlim(*axlim)
        ax.set_ylim(*axlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if plot_title:
        N = data['N']
        p1 = data['p1']
        p2 = data['p2']
        title_str = (r'$B_{eff}(\mathbf{r}), N=$' + str(N) + r', $p_1 = $' + str(p1) + r', $p_2 = $' + str(p2))
        if mean_in_title:
            curv_mean = np.mean(curv)
            title_str += r', $\langle B_{eff} \rangle_{\mathbf{r}} = $' + str(np.round(curv_mean, dp))
        ax.set_title(title_str)
    plt.show()


if __name__ == '__main__':
    # f = 'DarkState/Data/B_eff_p11_p20_m0.npz'
    # f = 'DarkState/Data/B_eff_p11_p20_m0_peak.npz'
    f = 'DarkState/Data/B_eff_p14_p2-3.npz'
    plot_curv(f)