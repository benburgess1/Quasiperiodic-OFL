import numpy as np
import matplotlib.pyplot as plt
import Dark_Approximant as DA
import matplotlib as mpl


def plot_curv(filename, cmap=plt.colormaps['bwr'], levels=None,
                     axlim=None, plot_cbar=True, plot_title=True,
                     xticks=None, yticks=None, ticks=None, norm=None, extend=None,
                     mean_in_title=False, dp=5, sum_in_title=True, patch=None):
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
    if patch is not None:
        ax.add_patch(patch)
    if plot_title:
        N = data['N']
        p1 = data['p1']
        p2 = data['p2']
        phi1 = data['phi1']
        phi2 = data['phi2']
        title_str = (r'$B_{eff}(\mathbf{r}), N=$' + str(N) + r', $p_1 = $' + str(p1) + r', $p_2 = $' + str(p2)
                     + r', $\phi_1 = $' + str(phi1) + r', $\phi_2 = $' + str(phi2))
        if mean_in_title:
            if patch is not None:
                path = patch.get_path()
                mask = path.contains_points(np.column_stack((xx.ravel(), yy.ravel())))
                print(curv.size)
                curv = curv.ravel()[mask]
                print(curv.size)
                curv_mean = np.mean(curv)
                # curv_mean = np.mean(curv.ravel()[mask])
            else:
                curv_mean = np.mean(curv)
            title_str += r', $\langle B_{eff} \rangle_{\mathbf{r}} = $' + str(np.round(curv_mean, dp))
        if sum_in_title:
            curv_sum = np.sum(curv)
            title_str += r', $\Phi = $' + str(np.round(curv_sum, dp))
        ax.set_title(title_str)
    plt.show()


def plot_d(filename, cmap=plt.colormaps['hot_r'], levels=None,
                     axlim=None, plot_cbar=True, plot_title=True,
                     xticks=None, yticks=None, ticks=None, norm=None, extend=None,
                     min_in_title=False, dp=5, square=False):
    data = np.load(filename)
    x_vals = data['x_vals']
    y_vals = data['y_vals']
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    d = np.real(data['d_vals'])
    if square:
        d = d**2
    fig, ax = plt.subplots()
    if levels is None:
        vmax = np.max(np.abs(d))
        levels = np.linspace(0, vmax, 200)
    if ticks is None:
        ticks = [np.min(levels), 0, np.max(levels)]
    plot = ax.contourf(xx, yy, d, cmap=cmap, levels=levels, norm=norm, extend=extend)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        ylab = r'$|d(\mathbf{r})|$'
        if square:
            ylab += r'$^2$'
        cbar.ax.set_ylabel(ylab, rotation=0)
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
        phi1 = data['phi1']
        phi2 = data['phi2']
        title_str = (ylab + r'$, N=$' + str(N) + r', $p_1 = $' + str(p1) + r', $p_2 = $' + str(p2)
                     + r', $\phi_1 = $' + str(phi1) + r', $\phi_2 = $' + str(phi2))
        if 'a' in data.files:
            title_str += r', $a = $' + str(data['a'])
        if min_in_title:
            min_d = np.min(d)
            if square:
                title_str += ', min(' + r'$|d|^2)=$' + str(np.round(min_d, dp))
            else:
                title_str += r'$, min(|d|)=$' + str(np.round(min_d, dp))
        ax.set_title(title_str)
    plt.show()


def plot_U(filename, cmap=plt.colormaps['hot_r'], levels=None,
                     axlim=None, plot_cbar=True, plot_title=True,
                     xticks=None, yticks=None, ticks=None, norm=None, extend=None,
                     max_in_title=False, dp=5, subtract1=False):
    data = np.load(filename)
    x_vals = data['x_vals']
    y_vals = data['y_vals']
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    if 'U_vals' in data.files:
        U = np.real(data['U_vals'])
    elif 'dU' in data.files:
        U = np.real(data['dU'])
    if subtract1:
        U -= 1
    fig, ax = plt.subplots()
    if levels is None:
        vmax = np.max(np.abs(U))
        levels = np.linspace(-vmax, vmax, 200)
    if ticks is None:
        ticks = [np.min(levels), 0, np.max(levels)]
    plot = ax.contourf(xx, yy, U, cmap=cmap, levels=levels, norm=norm, extend=extend)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(r'$U(\mathbf{r})$', rotation=0)
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
        title_str = (r'$U(\mathbf{r}), N=$' + str(N) + r', $p_1 = $' + str(p1) + r', $p_2 = $' + str(p2))
        if 'a' in data.files:
            title_str += r', $a = $' + str(data['a'])
        if 'M' in data.files:
            title_str += r', $M = $' + str(data['M'])
        if max_in_title:
            max_U = np.max(np.abs(U))
            title_str += r'$, max(|U|)=$' + str(np.round(max_U, dp))
        ax.set_title(title_str)
    plt.show()


def plot_U_means(filenames, x='a', xlab=r'$a$'):
    x_vals = []
    means = []
    stdevs = []
    for f in filenames:
        data = np.load(f)
        U_vals = np.real(data['U_vals'])
        if x in data.files:
            x_vals.append(data[x])
        else:
            x_vals.append(0)
            print('Warning: ' + x + ' not stored in file ' + f)
        means.append(np.mean(U_vals))
        stdevs.append(np.std(U_vals))
    fig, ax = plt.subplots()
    ax.errorbar(x_vals, means, yerr=stdevs, marker='x', ls='-', color='b')
    ax.axhline(y=1, color='r', ls='--')
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'$\langle U \rangle$')
    ax.set_title(r'Mean $U$')
    plt.show()


def plot_sv(filename, cmap=plt.colormaps['hot_r'], levels=None,
                     axlim=None, plot_cbar=True, plot_title=True,
                     xticks=None, yticks=None, ticks=None, norm=None, extend=None,
                     max_in_title=False, dp=5, real=True, factor2=True):
    data = np.load(filename)
    x_vals = data['x_vals']
    y_vals = data['y_vals']
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    sv = data['sv_vals']
    if real:
        sv = np.real(sv)
    if factor2:
        sv *= 2
    fig, ax = plt.subplots()
    if levels is None:
        vmax = np.max(np.abs(sv))
        levels = np.linspace(-vmax, vmax, 200)
    if ticks is None:
        ticks = [np.min(levels), 0, np.max(levels)]
    plot = ax.contourf(xx, yy, sv, cmap=cmap, levels=levels, norm=norm, extend=extend)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        ylab = r'$\langle s | v \rangle$'
        if real:
            ylab = r'$Re($' + ylab + r'$)$'
        if factor2:
            ylab = '2' + ylab
        cbar.ax.set_ylabel(ylab, rotation=0)
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
        title_str = (ylab + r'$, N=$' + str(N) + r', $p_1 = $' + str(p1) + r', $p_2 = $' + str(p2))
        if 'a' in data.files:
            title_str += r', $a = $' + str(data['a'])
        if max_in_title:
            max_sv = np.max(sv)
            title_str += r'$, max($' + ylab + r'$)=$' + str(np.round(max_sv, dp))
        ax.set_title(title_str)
    plt.show()


def plot_U_FT(filename, plot_quantity='mag', plot_cbar=True, 
                  cmap=None, levels=None, axlim=None, 
                  xticks=None, yticks=None, ticks=None, plot_title=True,
                  plot_log=False, kfactor=2*np.pi, reconstruct=False,
                  target_kx=None, target_ky=None):
    print('Loading data...')
    data = np.load(filename)
    if 'U_FT' in data.files:
        U_FT = data['U_FT']
    elif 'dU_FT' in data.files:
        U_FT = data['dU_FT']
    kx_vals = data['kx_vals'] * kfactor
    ky_vals = data['ky_vals'] * kfactor
    if reconstruct:
        if target_ky is None:
            target_ky = np.copy(target_kx)
        U_FT = DA.reconstruct_dU(U_FT, k_found=np.column_stack((kx_vals, ky_vals)),
                                 target_kx=target_kx, target_ky=target_ky)
        kx_vals = target_kx
        ky_vals = target_ky
    kxx, kyy = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    print('Done')
    fig, ax = plt.subplots()
    if plot_quantity == 'mag':
        z = np.abs(U_FT)
        zlab = r'$|U(k)|$'
        if plot_log:
            print('Evaluating log...')
            z = np.log10(z)
            print('Done')
            zlab = r'log$_{10}(|U(k)|)$'
        if cmap is None:
            cmap = plt.colormaps['hot']
    elif plot_quantity == 'real':
        z = np.real(U_FT)
        zlab = r'$Re(U(k))$'
        if cmap is None:
            cmap = plt.colormaps['bwr']
    elif plot_quantity == 'imag':
        z = np.imag(U_FT)
        zlab = r'$Im(U(k))$'
        if cmap is None:
            cmap = plt.colormaps['bwr']
    if levels is None:
        if not plot_log:
            vmax = np.max(np.abs(z))
            if plot_quantity == 'mag':
                levels = np.linspace(0, vmax, 200)
                if ticks is None:
                    ticks = [0, vmax]
            else:
                levels = np.linspace(-vmax, vmax, 200)
                if ticks is None:
                    ticks = [-vmax, 0, vmax]
        else:
            vmax = np.max(z)
            levels = np.linspace(-vmax, vmax, 200)
            if ticks is None:
                ticks = [-vmax, 0, vmax]
    print('Creating plot...')
    plot = ax.contourf(kxx, kyy, z, cmap=cmap, levels=levels)
    print('Done')
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab)
    ax.set_xlabel(r'$k_x$ / $|\mathbf{G}|$')
    ax.set_ylabel(r'$k_y$ / $|\mathbf{G}|$')
    if axlim is None:
        ax.set_xlim(np.min(kx_vals), np.max(kx_vals))
        ax.set_ylim(np.min(ky_vals), np.max(ky_vals))
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
        title_str = (zlab + r'$, N=$' + str(N) + r', $p_1 = $' + str(p1) + r', $p_2 = $' + str(p2))
        if 'a' in data.files:
            title_str += r', $a = $' + str(data['a'])
        ax.set_title(title_str)
    plt.show()


def plot_d_FT(filename, plot_quantity='mag', plot_cbar=True, 
                  cmap=None, levels=None, axlim=None, 
                  xticks=None, yticks=None, ticks=None, plot_title=True,
                  plot_log=False, kfactor=2*np.pi, square=True):
    print('Loading data...')
    data = np.load(filename)
    d_FT = data['d_FT']
    if square:
        d_FT = d_FT**2
    kx_vals = data['kx_vals'] * kfactor
    ky_vals = data['ky_vals'] * kfactor
    kxx, kyy = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    print('Done')
    fig, ax = plt.subplots()
    zlab = r'$|d_{eff}(k)|$'
    if square:
        zlab += r'$^2$'
        print(zlab)
    if plot_quantity == 'mag':
        z = np.abs(d_FT)
        if plot_log:
            print('Evaluating log...')
            z = np.log10(z)
            print('Done')
            zlab = r'log$_{10}($' + zlab + r'$)$'
        if cmap is None:
            cmap = plt.colormaps['hot']
    elif plot_quantity == 'real':
        z = np.real(d_FT)
        zlab = r'$Re($' + zlab + r'$)$'
        if cmap is None:
            cmap = plt.colormaps['bwr']
    elif plot_quantity == 'imag':
        z = np.imag(d_FT)
        zlab = r'$Im($' + zlab + r'$)$'
        if cmap is None:
            cmap = plt.colormaps['bwr']
    if levels is None:
        if not plot_log:
            vmax = np.max(np.abs(z))
            if plot_quantity == 'mag':
                levels = np.linspace(0, vmax, 200)
                if ticks is None:
                    ticks = [0, vmax]
            else:
                levels = np.linspace(-vmax, vmax, 200)
                if ticks is None:
                    ticks = [-vmax, 0, vmax]
        else:
            vmax = np.max(z)
            levels = np.linspace(-vmax, vmax, 200)
            if ticks is None:
                ticks = [-vmax, 0, vmax]
    print('Creating plot...')
    plot = ax.contourf(kxx, kyy, z, cmap=cmap, levels=levels)
    print('Done')
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab)
    ax.set_xlabel(r'$k_x$ / $|\mathbf{G}|$')
    ax.set_ylabel(r'$k_y$ / $|\mathbf{G}|$')
    if axlim is None:
        ax.set_xlim(np.min(kx_vals), np.max(kx_vals))
        ax.set_ylim(np.min(ky_vals), np.max(ky_vals))
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
        title_str = (zlab + r'$, N=$' + str(N) + r', $p_1 = $' + str(p1) + r', $p_2 = $' + str(p2))
        if 'a' in data.files:
            title_str += r', $a = $' + str(data['a'])
        ax.set_title(title_str)
    plt.show()


def plot_dV_FT(filename, plot_quantity='mag', plot_cbar=True, 
                  cmap=None, levels=None, axlim=None, 
                  xticks=None, yticks=None, ticks=None, plot_title=True,
                  plot_log=False, kfactor=1., reconstruct=False,
                  target_kx=None, target_ky=None, plot_variable='dV_FT_extracted'):
    print('Loading data...')
    data = np.load(filename)
    dV_FT = data[plot_variable]
    kx_vals = data['kx_vals'] * kfactor
    ky_vals = data['ky_vals'] * kfactor
    # if reconstruct:
    #     if target_ky is None:
    #         target_ky = np.copy(target_kx)
    #     U_FT = DA.reconstruct_dU(U_FT, k_found=np.column_stack((kx_vals, ky_vals)),
    #                              target_kx=target_kx, target_ky=target_ky)
    #     kx_vals = target_kx
    #     ky_vals = target_ky
    kxx, kyy = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    print('Done')
    fig, ax = plt.subplots()
    if plot_quantity == 'mag':
        z = np.abs(dV_FT)
        zlab = r'$|\delta V(k)|$'
        if plot_log:
            print('Evaluating log...')
            z = np.log10(z)
            print('Done')
            zlab = r'log$_{10}(|\delta V(k)|)$'
        if cmap is None:
            cmap = plt.colormaps['hot']
    elif plot_quantity == 'real':
        z = np.real(dV_FT)
        zlab = r'$Re(\delta V(k))$'
        if cmap is None:
            cmap = plt.colormaps['bwr']
    elif plot_quantity == 'imag':
        z = np.imag(dV_FT)
        zlab = r'$Im(\delta V(k))$'
        if cmap is None:
            cmap = plt.colormaps['bwr']
    if levels is None:
        if not plot_log:
            vmax = np.max(np.abs(z))
            if plot_quantity == 'mag':
                levels = np.linspace(0, vmax, 200)
                if ticks is None:
                    ticks = [0, vmax]
            else:
                levels = np.linspace(-vmax, vmax, 200)
                if ticks is None:
                    ticks = [-vmax, 0, vmax]
        else:
            vmax = np.max(z)
            levels = np.linspace(-vmax, vmax, 200)
            if ticks is None:
                ticks = [-vmax, 0, vmax]
    print('Creating plot...')
    plot = ax.contourf(kxx, kyy, z, cmap=cmap, levels=levels)
    print('Done')
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab)
    ax.set_xlabel(r'$k_x$ / $|\mathbf{G}|$')
    ax.set_ylabel(r'$k_y$ / $|\mathbf{G}|$')
    if axlim is None:
        ax.set_xlim(np.min(kx_vals), np.max(kx_vals))
        ax.set_ylim(np.min(ky_vals), np.max(ky_vals))
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
        title_str = (zlab + r'$, N=$' + str(N) + r', $p_1 = $' + str(p1) + r', $p_2 = $' + str(p2))
        if 'a' in data.files:
            title_str += r', $a = $' + str(data['a'])
        if 'M' in data.files:
            title_str += r', $M = $' + str(data['M'])
        ax.set_title(title_str)
    plt.show()


def plot_dV(filename, cmap=plt.colormaps['hot_r'], levels=None,
                     axlim=None, plot_cbar=True, plot_title=True,
                     xticks=None, yticks=None, ticks=None, norm=None, extend=None,
                     max_in_title=False, dp=5, plot_variable='dV'):
    data = np.load(filename)
    x_vals = data['x_vals']
    y_vals = data['y_vals']
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    if plot_variable in data.files:
        dV = np.real(data[plot_variable])
    else:
        print('Error: no such file ' + plot_variable)
        return
    fig, ax = plt.subplots()
    if levels is None:
        vmax = np.max(np.abs(dV))
        levels = np.linspace(-vmax, vmax, 200)
    if ticks is None:
        ticks = [np.min(levels), 0, np.max(levels)]
    plot = ax.contourf(xx, yy, dV, cmap=cmap, levels=levels, norm=norm, extend=extend)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(r'$\delta V(\mathbf{r})$', rotation=0)
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
        title_str = (r'$\delta V(\mathbf{r}), N=$' + str(N) + r', $p_1 = $' + str(p1) + r', $p_2 = $' + str(p2))
        if 'a' in data.files:
            title_str += r', $a = $' + str(data['a'])
        if 'M' in data.files:
            title_str += r', $M = $' + str(data['M'])
        if max_in_title:
            max_dV = np.max(np.abs(dV))
            title_str += r'$, max(|\delta V|)=$' + str(np.round(max_dV, dp))
        ax.set_title(title_str)
    plt.show()


def plot_fluxes(filename, average_uc=False, x_ticks=None, scale_dx=True):
    data = np.load(filename)
    a_vals = data['a_vals']
    flux_vals = data['flux_vals']
    if scale_dx:
        flux_vals *= data['dx']**2
    if average_uc:  
        A_vals = (2*np.pi*a_vals)**2
        flux_vals /= A_vals
        ylab = r'$\langle B \rangle_{\mathbf{r}}$'
    else:
        ylab = r'$\Phi$'
    fig, ax = plt.subplots()
    ax.plot(a_vals, flux_vals, marker='x', ls='', color='b')
    ax.set_xlabel(r'$a$')
    ax.set_ylabel(ylab, rotation=0)
    ax.set_title(ylab + r' vs $a$')
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    plt.show()


def triangle_patch(r0, a, color='r'):
    x1 = a * np.array([1, 0])
    x2 = a * np.array([-0.5, 0.5*np.sqrt(3)])
    points = np.array([r0, r0 + x1, r0 + x1 + x2])
    patch = mpl.patches.Polygon(points, closed=True, ec=color, fc=(0,0,0,0))
    return patch
    


if __name__ == '__main__':
    f_flux = 'DarkState/Data/B_eff/Phi_uc_R5_a1-20_dx0.05_offset_p10.5_p20.5_phi10_phi20.npz'
    # plot_fluxes(f_flux, average_uc=True, scale_dx=True, x_ticks=np.arange(0,21,5))

    patch = triangle_patch(r0=np.array([2*np.pi/3,2*np.pi/np.sqrt(3)]), a=8*np.pi/3)
    # f_B = 'DarkState/Data/B_eff/B_eff_uc_R5_a3_dx0.05_offset0.01_p10.5_p20.5_phi10_phi20.npz'
    f_B = 'DarkState/Data/B_eff/B_eff_uc_R6_dx0.05_offset0.001_p11_p20_phi10_phi20.npz'
    plot_curv(f_B, mean_in_title=True, sum_in_title=False, patch=patch)

    # f_d = 'DarkState/Data/B_eff/d_uc_R5_a3_dx0.05_offset0.01_p10.5_p20.5_phi10_phi20.npz'
    f_d = 'DarkState/Data/B_eff/d_uc_R6_dx0.05_offset0.001_p11_p20_phi10_phi20.npz'
    plot_d(f_d, square=True, min_in_title=True, dp=10)



    # f = 'DarkState/Data/B_eff_p11_p20_m0.npz'
    # f = 'DarkState/Data/B_eff_p11_p20_m0_peak.npz'
    # f = 'DarkState/Data/B_eff_p14_p2-3.npz'
    f_d = 'DarkState/Data/B_eff/d_aligned_extended_R5_a3_p11_p20_phi10_phi20.npz'
    # plot_d(f_d, square=True, levels=np.linspace(0,25,200), ticks=[0,25])

    f_U = 'DarkState/Data/B_eff/U_aligned_extended_R5_a3_p11_p20_phi10_phi20.npz'
    # plot_U(f_U, max_in_title=True)

    f_dU = 'DarkState/Data/B_eff/dU_extended_R5_a3_p11_p20_phi10_phi20.npz'
    # plot_U(f_dU, max_in_title=True, cmap=plt.colormaps['bwr'])
    # plot_U_FT(f_dU, plot_quantity='mag', plot_log=False, levels=np.linspace(0,0.1,200), ticks=[0,0.1],
    #           kfactor=1)
    
    f_dV = 'DarkState/Data/B_eff/dV_extended_R5_a3_M4000_p11_p20_phi10_phi20.npz'
    # vmax = 0.1
    # data = np.load(f_dV)
    # dV_k = data['dV_FT_list']
    # print(np.max(np.abs(dV_k)), np.min(np.abs(dV_k)))
    # plot_dV(f_dV, plot_variable='dV', cmap=plt.colormaps['bwr'], 
    #         levels=np.linspace(-vmax,vmax,200), ticks=[-vmax,0,vmax], axlim=(-5,5))
    # plot_dV(f_dV, plot_variable='dV_r_extracted', cmap=plt.colormaps['bwr'])
    # plot_dV(f_dV, plot_variable='dV_r_extracted')
    # plot_dV_FT(f_dV, plot_variable='dV_FT', plot_quantity='imag')
    # plot_dV_FT(f_dV, plot_variable='dV_FT_extracted')

    f_U_compensated = 'DarkState/Data/B_eff/U_compensated_extended_R5_a3_M4000_p11_p20_phi10_phi20.npz'
    # plot_U(f_U_compensated, max_in_title=False, cmap=plt.colormaps['bwr'], subtract1=True, 
    #        levels=np.linspace(-0.3,0.3,200), ticks=[-0.3, 0, 0.3])

    M = [100, 500, 1000, 1500, 2000, 3000, 4000]
    filenames = [f_U]
    for m in M:
        filenames.append('DarkState/Data/B_eff/U_compensated_extended_R5_a3_M' 
                         + str(m) + '_p11_p20_phi10_phi20.npz')
    # plot_U_means(filenames, x='M', xlab=r'$M$')
    # f = 'DarkState/Data/B_eff/U_aligned_R5_a3_p11_p20_phi10_phi20.npz'
    # plot_U_FT(f, plot_quantity='mag', plot_log=False, levels=np.linspace(0,0.1,200), ticks=[0,0.1],
    #           kfactor=1)
    # plot_U(f, max_in_title=True)
    # f = 'DarkState/Data/B_eff/U_compensated_R5_a3_M200_p11_p20_phi10_phi20.npz'
    # plot_U_FT(f, plot_quantity='mag', plot_log=False, levels=np.linspace(0,0.1,200), ticks=[0,0.1],
    #           kfactor=1)
    # plot_U(f, max_in_title=True)
    # data = np.load(f)
    # kx = data['kx_vals']
    # # f = 'DarkState/Data/B_eff/dmag_R5_a3_p11_p20_phi10_phi20.npz'
    # # plot_U(f, max_in_title=True)
    # print((kx[1] - kx[0]))
    # # print(np.min(x), np.max(x), x.size)
    # vmax = 1e3
    # f = 'DarkState/Data/B_eff/dU_R5_a3_M200_p11_p20_phi10_phi20.npz'
    # plot_U_FT(f, plot_quantity='mag', plot_log=False, levels=np.linspace(0,0.1,200), ticks=[0,0.1],
    #           reconstruct=True, target_kx=kx, kfactor=1)
    # plot_d_FT(f, plot_quantity='mag', plot_log=True, square=False, levels=np.linspace(-1,5,200), ticks=np.arange(-1,6))
    # f = 'DarkState/Data/B_eff/U_approx_R5_a5_p11_p20_phi10_phi20.npz'
    # f = 'DarkState/Data/B_eff/sv_R5_a5_p11_p20_phi10_phi20.npz'
    # plot_sv(f, max_in_title=True, real=True, factor2=True)
    # f = 'DarkState/Data/B_eff/dmag_R5_a5_p11_p20_phi10_phi20.npz'
    # plot_d(f, min_in_title=True, square=False)
    # f = 'DarkState/Data/B_eff/U_a20_p11_p20_phi10_phi20.npz'
    # data = np.load(f)
    # U_vals = data['U_vals']
    # print(np.mean(U_vals))
    # plot_d(f, min_in_title=True, square=False)
    # filenames = ['DarkState/Data/B_eff/U_a3_p11_p20_phi10_phi20.npz',
    #              'DarkState/Data/B_eff/U_a4_p11_p20_phi10_phi20.npz',
    #              'DarkState/Data/B_eff/U_a5_p11_p20_phi10_phi20.npz',
    #              'DarkState/Data/B_eff/U_a6_p11_p20_phi10_phi20.npz',
    #              'DarkState/Data/B_eff/U_a7_p11_p20_phi10_phi20.npz',
    #              'DarkState/Data/B_eff/U_a8_p11_p20_phi10_phi20.npz',
    #              'DarkState/Data/B_eff/U_a9_p11_p20_phi10_phi20.npz',
    #              'DarkState/Data/B_eff/U_a10_p11_p20_phi10_phi20.npz']
    # plot_U_means(filenames)