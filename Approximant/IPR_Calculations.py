import numpy as np
import matplotlib.pyplot as plt
import Approximant_Bandstructure as ABS
import scipy as sp


def calc_IPR_k(filename=None, evects=None, save_filename='IPR_k.npz', save=False,
               transfer_params=['U0', 'V0', 'N', 'R', 'a', 
                                'cutoff', 'qx_vals', 'qy_vals'],
               sum_spin=True):
    if filename is not None:
        data = np.load(filename)
        evects = data['evects']
    if sum_spin:
        N_q = evects.shape[0] / 2
        abs_evects_sq = np.abs(evects)**2
        psi_spinsummed = abs_evects_sq[:N_q] + abs_evects_sq[N_q:]
        IPR_k = np.sum(np.abs(psi_spinsummed[:N_q,:])**2, axis=0)
    else:
        IPR_k = np.sum(np.abs(evects)**4, axis=0)
    if save:
        save_dict = {p:data[str(p)] for p in transfer_params}
        np.savez(save_filename, IPR_k=IPR_k, **save_dict)
    else:
        return IPR_k


def plot_IPR_k(filename, ax=None, bands=None, cmap=plt.colormaps['hot'], plot_log=False,
               levels=None, plot_title=True, title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$'},
               dp=4, plot_cbar=True, axlim=None, plot_fig=True, bands_in_title=False):
    data = np.load(filename)
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    IPR_vals = data['IPR_k']
    if ax is None:
        fig,ax = plt.subplots()
    if bands is None:
        bands = np.arange(IPR_vals.shape[0])
    IPR_sum = np.sum(IPR_vals[bands,:,:], axis=0)
    zlab = r'$IPR_k$'
    if plot_log:
        IPR_sum = np.log10(IPR_sum)
        zlab = r'$log_{10}($' + zlab + r'$)$'
    if levels is None:
        vmin = np.min(IPR_sum)
        vmax = np.max(IPR_sum)
        levels = np.linspace(vmin,vmax,200)
        ticks = [vmin, vmax]
    #print(levels)
    plot = ax.contourf(qxx, qyy, IPR_sum, cmap=cmap, levels=levels)
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$', rotation=0)
    if plot_title:
        title_str = ''
        for k, v in title_params.items():
            title_str += v + r'$=$' + str(np.round(data[k],dp)) + ', '
        title_str = title_str[:-2]
        if bands_in_title:
            if len(bands) <= 2:
                band_str = str(bands)
            else:
                band_str = '[' + str(bands[0]) + '-' + str(bands[-1]) + ']'
            title_str += ', Bands = ' + band_str

        ax.set_title(title_str)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab, rotation=0)
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
    

def calc_Psi_r(Psi_k, q, basis, r_vals):
    k_vals = q - basis
    # M = np.array([[np.exp(1j*k.dot(r)) for k in k_vals] for r in r_vals])
    print(f'r_vals: {r_vals.shape}')
    print(f'k_vals: {k_vals.shape}')
    print(f'Psi_k: {Psi_k.shape}')
    phase = r_vals @ k_vals.T
    print(f'phase: {phase.shape}')
    Psi_r = np.exp(1j * phase) @ Psi_k
    print(f'Psi_r: {Psi_r.shape}')
    return Psi_r


def calc_Psi_r_file(filename, x_vals, y_vals, save_filename='Psi_r.npz', bands=None, 
                    transfer_params=['U0', 'V0', 'N', 'R', 'a', 
                              'cutoff', 'qx_vals', 'qy_vals']):
    data = np.load(filename)
    evects_arr = data['evects_arr']
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    (b_down, b_up) = data['basis']
    basis = np.vstack((b_up, b_down))
    if bands is None:
        bands = np.arange(evects_arr.shape[1])
    Psi_k = evects_arr[:,bands,:,:]
    Psi_r = np.zeros((x_vals.size, y_vals.size, *evects_arr.shape[1:]), dtype=np.complex128)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    r_vals = np.column_stack((xx.flatten(), yy.flatten()))
    for i, qx in enumerate(qx_vals):
        for j, qy in enumerate(qy_vals):
            print(f'Evaluating q value {j + qy_vals.size*i + 1} out of {qx_vals.size*qy_vals.size}...' + 10*' ', end='\r')
            q = np.array([qx, qy])
            Psi_r[:,:,:,i,j] = calc_Psi_r(Psi_k[:,:,i,j], q, basis, r_vals).reshape((x_vals.size, y_vals.size, bands.size))
    print(f'Final shape: {Psi_r.shape}')
    # Normalise to one unit cell, assuming 4 unit cells in r_vals:
    norm = np.sum(np.abs(Psi_r)**2, axis=(0,1))
    print(f'norm: {norm.shape}')
    Psi_r *= 2 / np.sqrt(norm)
    save_dict = {p:data[str(p)] for p in transfer_params}
    np.savez(save_filename, x_vals=x_vals, y_vals=y_vals, Psi_r=Psi_r, basis=basis, **save_dict)

    

def calc_IPR(filename, r_vals, save_filename='IPR.npz', save_psi=True,
             transfer_params=['U0', 'V0', 'N', 'R', 'a', 
                              'cutoff', 'qx_vals', 'qy_vals']):
    data = np.load(filename)
    evects = data['evects']
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    (b_down, b_up) = data['basis']
    basis = np.vstack((b_up, b_down))
    # print(f'Basis: {basis.shape}')
    # q = np.array([qx_vals[0], qy_vals[0]])
    # Psi_r = calc_Psi_r(evects[:,:,0,0], q, basis, r_vals)
    # print(f'Psi_k: {evects.shape}')
    # print(f'Psi_r: {Psi_r.shape}')
    Psi_r = np.zeros((r_vals.shape[0], *evects.shape[1:]), dtype=np.complex128)
    for i, qx in enumerate(qx_vals):
        for j, qy in enumerate(qy_vals):
            print(f'Evaluating q point {i*len(qy_vals) + j + 1} out of {len(qy_vals)**2}...' + 10*' ', end='\r')
            q = np.array([qx, qy])
            Psi_r[:,:,i,j] = calc_Psi_r(evects[:,:,i,j], q, basis, r_vals)
    print('\nDone')
    # Normalise
    Psi_r /= np.linalg.norm(Psi_r, axis=0)
    # print(Psi_r.shape)
    IPR = np.sum(np.abs(Psi_r)**4, axis=0)
    save_dict = {p:data[str(p)] for p in transfer_params}
    if save_psi:
        save_dict['Psi_r'] = Psi_r
    np.savez(save_filename, IPR=IPR, r_vals=r_vals, **save_dict)


def plot_Psi_r(filename, q_idx=(0,0), n=0, cmap=plt.colormaps['hot'], levels=None, axlim=None,
               plot_cbar=True, plot_fig=True, ticks=None, plot_title=False,
               title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$'}, state_in_title=True, dp=4,
               IPR_in_title=False):
    data = np.load(filename)
    Psi_r = data['Psi_r'][:, n, *q_idx]
    # print(Psi_r.shape)
    # print(np.sqrt(Psi_r.shape))
    N = int(np.sqrt(Psi_r.size))
    Psi_r = Psi_r.reshape((N,N))
    # print(Psi_r.shape)
    r_vals = data['r_vals']
    xx = r_vals[:,0].reshape(N,N)
    yy = r_vals[:,1].reshape(N,N)
    fig, ax = plt.subplots()
    if levels is None:
        vmin = 0
        vmax = np.max(np.abs(Psi_r))
        levels = np.linspace(vmin, vmax, 200)
        ticks = [0, vmax]
    plot = ax.contourf(xx, yy, np.abs(Psi_r), cmap=cmap, levels=levels)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0)
    if plot_title:
        title_str = ''
        for k, v in title_params.items():
            title_str += v + r'$=$' + str(np.round(data[k],dp)) + ', '
        title_str = title_str[:-2]
        if state_in_title:
            qx = data['qx_vals']
            qy = data['qy_vals']
            q = np.array([qx[q_idx[0]], qy[q_idx[1]]])
            title_str += r', $q=$' + str(np.round(q,2)) + r', $n=$' + str(n)
        if IPR_in_title:
            IPR = data['IPR'][n, *q_idx]
            title_str += r', $IPR=$' + str(np.round(IPR, dp))
        ax.set_title(title_str)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(r'$|\Psi(\mathbf{r})|$', rotation=0)
    if axlim is None:
        ax.set_xlim(np.min(r_vals[:,0]), np.max(r_vals[:,0]))
        ax.set_ylim(np.min(r_vals[:,1]), np.max(r_vals[:,1]))
    else:
        ax.set_xlim(*axlim)
        ax.set_ylim(*axlim)
    if plot_fig:
        plt.show()
    else:
        return plot


def plot_IPR(filename, ax=None, bands=None, cmap=plt.colormaps['hot'], plot_log=False,
               levels=None, plot_title=True, title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$'},
               dp=4, plot_cbar=True, axlim=None, plot_fig=True, bands_in_title=False):
    data = np.load(filename)
    qx_vals = data['qx_vals']
    qy_vals = data['qy_vals']
    qxx,qyy = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    IPR_vals = data['IPR']
    if ax is None:
        fig,ax = plt.subplots()
    if bands is None:
        bands = np.arange(IPR_vals.shape[0])
    IPR_sum = np.sum(IPR_vals[bands,:,:], axis=0)
    zlab = r'$IPR_r$'
    if plot_log:
        IPR_sum = np.log10(IPR_sum)
        zlab = r'$log_{10}($' + zlab + r'$)$'
    if levels is None:
        vmin = np.min(IPR_sum)
        vmax = np.max(IPR_sum)
        levels = np.linspace(vmin,vmax,200)
        ticks = [vmin, vmax]
    #print(levels)
    print(f'q = [{qxx[0,0]}, {qyy[0,0]}]: IPR = {IPR_sum[0,0]}')
    print(f'q = [{qxx[30,30]}, {qyy[30,30]}]: IPR = {IPR_sum[30,30]}')
    plot = ax.contourf(qxx, qyy, IPR_sum, cmap=cmap, levels=levels)
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$', rotation=0)
    if plot_title:
        title_str = ''
        for k, v in title_params.items():
            title_str += v + r'$=$' + str(np.round(data[k],dp)) + ', '
        title_str = title_str[:-2]
        if bands_in_title:
            if len(bands) <= 2:
                band_str = str(bands)
            else:
                band_str = '[' + str(bands[0]) + '-' + str(bands[-1]) + ']'
            title_str += ', Bands = ' + band_str

        ax.set_title(title_str)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab, rotation=0)
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
    

def calc_IPR_k_vs_U(filenames, save_filename, 
                    transfer_params=['N', 'R', 'a', 'V0',
                                     'cutoff', 'qx_vals', 'qy_vals']):
    d1 = np.load(filenames[0])
    IPR_k = np.zeros((len(filenames), *d1['evects'].shape[1:]))
    U_vals = np.zeros(len(filenames))
    # V_vals = np.zeros(len(filenames))
    for i, f in enumerate(filenames):
        data = np.load(f)
        IPR_k[i,:,:,:] = calc_IPR_k(f, save=False)
        U_vals[i] = data['U0']
        # V_vals[i] = data['V0']
    save_dict = {p:d1[str(p)] for p in transfer_params}
    np.savez(save_filename, IPR_k=IPR_k, U_vals=U_vals, **save_dict)
    

def plot_IPR_k_means(filename, bands=None, title_params={'V0':r'$V$', 'N':r'$N$'}, plot_title=True,
                     bands_in_title=True, dp=4):
    data = np.load(filename)
    IPR_k = data['IPR_k']
    U = data['U_vals']
    if bands is None:
        bands = np.arange(data['IPR_k'].shape[1])
    IPR_slice = IPR_k[:,bands,:,:]
    IPR_means = np.mean(IPR_slice, axis=(1,2,3))
    IPR_stds = np.std(IPR_slice, axis=(1,2,3))
    fig, ax = plt.subplots()
    ax.errorbar(U, IPR_means, yerr=IPR_stds, color='b', marker='x', ls='-')
    ax.set_xlabel(r'$U/E_R$')
    ax.set_ylabel(f'$IPR_k$', rotation=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if plot_title:
        title_str = ''
        for k, v in title_params.items():
            title_str += v + r'$=$' + str(np.round(data[k],dp)) + ', '
        title_str = title_str[:-2]
        if bands_in_title:
            if len(bands) <= 2:
                band_str = str(bands)
            else:
                band_str = '[' + str(bands[0]) + '-' + str(bands[-1]) + ']'
            title_str += ', Bands = ' + band_str
        ax.set_title(title_str)
    plt.show()


def plot_single_IPR_k(filename, q_idx=(15,15), ax=None, title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$'}, plot_title=True,
                     bands_in_title=True, dp=5, bands=None, q_in_title=True, x='n', band_edge=None, label=None,
                     color='b', plot_fig=True, dE_in_title=False, plot_fit=False, dE=0.1, **kwargs):
    data = np.load(filename)
    IPR_k = data['IPR_k'][:, *q_idx]
    if bands is None:
        bands = np.arange(IPR_k.shape[0])
    IPR_k = IPR_k[bands]
    if x == 'n':
        x_plot = bands
        xlab = r'$n$'
    elif x == 'E':
        x_plot = data['E_vals'][bands, *q_idx]
        xlab = r'$E$ / $E_R$'
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_plot, IPR_k, color=color, marker='x', ls='-', label=label)
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'$IPR_k$', rotation=0)
    ax.set_ylim(bottom=0)
    if plot_title:
        title_str = ''
        for k, v in title_params.items():
            title_str += v + r'$=$' + str(np.round(data[k],dp)) + ', '
        title_str = title_str[:-2]
        if bands_in_title:
            if len(bands) <= 2:
                band_str = str(bands)
            else:
                band_str = '[' + str(bands[0]) + '-' + str(bands[-1]) + ']'
            title_str += ', Bands = ' + band_str
        if q_in_title:
            qx = data['qx_vals']
            qy = data['qy_vals']
            q = np.array([qx[q_idx[0]], qy[q_idx[1]]])
            title_str += r', $q=$' + str(np.round(q,2))
        if dE_in_title:
            if band_edge is not None:
                dE = data['E_vals'][band_edge+1,*q_idx] - data['E_vals'][band_edge,*q_idx]
                title_str += r', $\Delta E=$' + str(np.round(dE,4))
        ax.set_title(title_str)
    if band_edge is not None:
        if x == 'n':
            ax.axvline(x=band_edge, color='r', ls='--', label='Gap Position')
        if x == 'E':
            E_edge = data['E_vals'][band_edge, *q_idx]
            ax.axvline(x=E_edge, color='r', ls='--', label='Gap Position')
        ax.legend()
    if plot_fit:
        popt, perr = calc_powerlaw_fit(IPR_k, x_plot, dE=dE, **kwargs)
        # popt, perr = calc_loglog_fit(IPR_k, x_plot, dE=dE, **kwargs)
        x_fit = np.linspace(popt[1], popt[1]+dE, 300)[1:]
        y_fit = popt[0] * (x_fit - popt[1]) ** (-2*popt[2])
        A = np.round(popt[0],dp)
        Ec = np.round(popt[1],dp)
        v = np.round(popt[2],dp)
        Aerr = np.round(perr[0],dp)
        Ecerr = np.round(perr[1],dp)
        verr = np.round(perr[2],dp)
        ax.plot(x_fit, y_fit, color='r', ls=':', label=r'$A|E-E_c|^{-2 \nu}$')
        ax.legend(loc='upper right')
        ax.text(x=0.65, y=0.8, s=(r'$A=$' + str(A) + r'$\pm$' + str(Aerr)), transform=ax.transAxes)
        ax.text(x=0.65, y=0.75, s=(r'$E_c=$' + str(Ec) + r'$\pm$' + str(Ecerr)), transform=ax.transAxes)
        ax.text(x=0.65, y=0.7, s=(r'$\nu=$' + str(v) + r'$\pm$' + str(verr)), transform=ax.transAxes)
        print(f'popt = {np.round(popt, 5)}')
        print(f'perr = {np.round(perr, 5)}')
    if plot_fig:
        plt.show()


def plot_several_IPR_k(filenames, q_idx=(0,0), label='U', x='E', dp=3, ylim=None,
                       colors=['royalblue','r','deepskyblue','gold','limegreen', 'orange', 'darkgreen', 'violet'],
                       q_in_title=True, title_params={'V0':r'$V$', 'N':r'$N$', 'a':r'$a$'}):
    fig, ax = plt.subplots()
    for i,f in enumerate(filenames):
        if label == 'U':
            data = np.load(f)
            label_str = str(np.round(data['U0'],dp))
        elif len(label) == len(filenames):
            label_str = label[i]
        
        plot_single_IPR_k(f, q_idx=q_idx, ax=ax, x=x, plot_title=False, label=label_str,
                          color=colors[i%len(filenames)], plot_fig=False)
    ax.legend(title=r'$U$')
    title_str = r'$IPR_k$ vs $E$'
    for k, v in title_params.items():
        title_str += ', ' + v + r'$=$' + str(np.round(data[k],dp))
    if q_in_title:
        data = np.load(filenames[0])
        qx = data['qx_vals']
        qy = data['qy_vals']
        q = np.array([qx[q_idx[0]], qy[q_idx[1]]])
        title_str += r', $q=$' + str(np.round(q,2))
    ax.set_title(title_str)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.show()


def calc_powerlaw_fit(IPR_k, E_vals, E_min='auto', dE=0.1, IPR_min=0.):
    if E_min == 'auto':
        E_min = E_vals[np.argmax(IPR_k)]
        # print(E_min)
    def f(x, A, E_c, v):
        return A * (x-E_c) ** (-2*v)
    mask = (E_vals >= E_min) & (E_vals <= E_min + dE) & (IPR_k >= IPR_min)
    y = IPR_k[mask]
    x = E_vals[mask]
    # print(x)
    # print(y)
    p0 = [1., E_min-0.02, 2.]
    popt, pcov = sp.optimize.curve_fit(f, x, y, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def calc_loglog_fit(IPR_k, E_vals, E_min='auto', dE=0.1, IPR_min=0.):
    if E_min == 'auto':
        E_min = E_vals[np.argmax(IPR_k)]
    mask = (E_vals >= E_min) & (E_vals <= E_min + dE) & (IPR_k >= IPR_min)
    y = np.log(IPR_k[mask])
    x = E_vals[mask]
    def f(x, A, E_c, v):
        return np.log(A) - 2 * v * np.log(x-E_c)
    p0 = [0.1, E_min-0.02, 0.3]
    popt, pcov = sp.optimize.curve_fit(f, x, y, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def plot_summed_Psi_r(filename, n=0, cmap=plt.colormaps['hot'], levels=None, axlim=None,
               plot_cbar=True, plot_fig=True, ticks=None, plot_title=False,
               title_params={'U0':r'$U$', 'N':r'$N$', 'V0':r'$V$'}, dp=4, band_in_title=True):
    data = np.load(filename)
    Psi_r = data['Psi_r'][:, :, n, :, :]
    z = np.sum(np.abs(Psi_r)**2, axis=(2,3))
    zlab = r'$\sum_{\mathbf{q}}|\Psi^{(n)}_{\mathbf{q}}(\mathbf{r})|^2$'
    zmax = np.max(z)
    zmin = np.min(z)
    print(f'z_max = {np.round(zmax,4)}')
    print(f'z_min = {np.round(zmin,4)}')
    print(f'z_max/z_min = {np.round(zmax/zmin,4)}')
    # if square:
    #     Psi_r = np.abs(Psi_r)**2
    #     zlab = zlab + r'$^2$'
    x_vals = data['x_vals']
    y_vals = data['y_vals']
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    fig, ax = plt.subplots()
    if levels is None:
        vmin = 0
        vmax = np.max(z)
        levels = np.linspace(vmin, vmax, 200)
        ticks = [0, vmax]
    plot = ax.contourf(xx, yy, z, cmap=cmap, levels=levels)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=0)
    if plot_title:
        title_str = ''
        for k, v in title_params.items():
            title_str += v + r'$=$' + str(np.round(data[k],dp)) + ', '
        title_str = title_str[:-2]
        if band_in_title:
            title_str += r', $n=$' + str(n)
        ax.set_title(title_str)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab, rotation=0)
    if axlim is None:
        ax.set_xlim(np.min(x_vals), np.max(x_vals))
        ax.set_ylim(np.min(y_vals), np.max(y_vals))
    else:
        ax.set_xlim(*axlim)
        ax.set_ylim(*axlim)
    if plot_fig:
        plt.show()
    else:
        return plot



if __name__ == '__main__':
    f1 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.02_N5_V0.05.npz'
    f2 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.04_N5_V0.05.npz'
    f3 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.06_N5_V0.05.npz'
    f4 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.08_N5_V0.05.npz'
    f5 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.1_N5_V0.05.npz'
    f6 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.18_N5_V0.05.npz'
    f7 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.2_N5_V0.05.npz'
    f8 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.22_N5_V0.05.npz'
    f9 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.24_N5_V0.05.npz'
    f10 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.3_N5_V0.05.npz'
    # f11 = 'Approximant/Data/8Fold/Eigenvectors/Data_R8_a3_c2.5_U0.36_N5_V0.05.npz'
    # f = 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a3_c2.5_U0.3_N5_V0.05.npz'
    f = 'Approximant/Data/8Fold/Eigenvectors/IPR_R8_a3_c2.5_U0.3_N5_V0.05.npz'
    x_vals = np.arange(-20, 21, 1)
    y_vals = np.copy(x_vals)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    r_vals = np.column_stack((xx.flatten(), yy.flatten()))
    filenames = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    f = 'Approximant/Data/8Fold/Eigenvectors/IPRs_R8_a3_c2.5_N5_V0.05.npz'
    # calc_IPR_k_vs_U(filenames, f)
    # plot_IPR_k_means(f, bands=np.arange(14))
    # print(r_vals.shape)
    # calc_IPR(f0, r_vals=r_vals, save_filename=f, save_psi=True)
    # calc_IPR_k(f0, save_filename=f)
    # # data = np.load(save_f)
    # # print(data.files)
    # for i in range(14):
    #     plot_IPR_k(f, bands=[i], bands_in_title=True)

    f = 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a3_c2.5_U0.24_N5_V0.05.npz'
    # calc_IPR_k(f9, f, save=True)
    # plot_IPR_k(f, bands=[0])

    # f = 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a7_c2.5_U0.12_N5_V0.05.npz'
    # plot_single_IPR_k(f, q_idx=(2,2), x='E', band_edge=81, dE_in_title=True)

    # filenames = ['Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a7_c2.5_U0.12_N5_V0.05.npz',
    #              'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a7_c2.5_U0.16_N5_V0.05.npz',
    #              'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a7_c2.5_U0.2_N5_V0.05.npz',
    #              'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a7_c2.5_U0.24_N5_V0.05.npz',
    #              'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a7_c2.5_U0.26_N5_V0.05.npz',
    #              'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a7_c2.5_U0.28_N5_V0.05.npz',
    #              'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a7_c2.5_U0.3_N5_V0.05.npz',]
    

    f = 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a10_c2.5_U0.28_N5_V0.05.npz'
    # plot_single_IPR_k(f, q_idx=(0,0), x='E', band_edge=163, dE_in_title=True)
    filenames = ['Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a10_c2.5_U0.2_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a10_c2.5_U0.22_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a10_c2.5_U0.24_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a10_c2.5_U0.25_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a10_c2.5_U0.26_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a10_c2.5_U0.27_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a10_c2.5_U0.28_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a10_c2.5_U0.3_N5_V0.05.npz']
    # plot_several_IPR_k(filenames, q_idx=(0,0), label='U', x='E', ylim=(0,0.6))

    f = 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a14_c2.5_U0.12_N5_V0.05.npz'
    # plot_single_IPR_k(f, q_idx=(0,0), x='E', band_edge=None, dE_in_title=False, plot_fit=True, dE=0.15, IPR_min=0.015)
    filenames = ['Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a14_c2.5_U0.2_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a14_c2.5_U0.22_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a14_c2.5_U0.24_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a14_c2.5_U0.25_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a14_c2.5_U0.26_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a14_c2.5_U0.27_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a14_c2.5_U0.28_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a14_c2.5_U0.32_N5_V0.05.npz']
    # plot_several_IPR_k(filenames, q_idx=(0,0), label='U', x='E', ylim=(0,0.6))

    f = 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a17_c2.5_U0.22_N5_V0.05.npz'
    # plot_single_IPR_k(f, q_idx=(0,0), x='E', band_edge=None, dE_in_title=False, plot_fit=True, dE=0.15, IPR_min=0.015)
    filenames = ['Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a17_c2.5_U0.2_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a17_c2.5_U0.22_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a17_c2.5_U0.24_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a17_c2.5_U0.25_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a17_c2.5_U0.26_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a17_c2.5_U0.27_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a17_c2.5_U0.28_N5_V0.05.npz',
                 'Approximant/Data/8Fold/Eigenvectors/IPR_k_R8_a17_c2.5_U0.32_N5_V0.05.npz']
    # plot_several_IPR_k(filenames, q_idx=(0,0), label='U', x='E', ylim=(0,0.6))

    f = 'Approximant/Data/8Fold/Eigenvectors/Data_Multiband_R8_a3_c2.5_U0.3_N5_V0.12.npz'
    save_filename = 'Approximant/Data/8Fold/Eigenvectors/Psi_r_R8_a3_c2.5_U0.3_N5_V0.12.npz'
    a = 3
    L = 2 * np.pi * a
    x_vals = np.linspace(-L,L,41)
    y_vals = np.copy(x_vals)
    calc_Psi_r_file(f, x_vals=x_vals, y_vals=y_vals, save_filename=save_filename)
    # data = np.load(f)
    # print(data['evects_arr'].shape)
    data = np.load(save_filename)
    Psi_r = data['Psi_r']
    sum_psi = np.sum(np.abs(Psi_r)**2, axis=(0,1))
    print(sum_psi.shape)
    print(sum_psi[0,0:10,0])
    # print(np.sum(np.abs(Psi_r)**2, axis=(0,1)))

    plot_summed_Psi_r(save_filename, n=8, plot_title=True)


    # plot_Psi_r(f, q_idx=(0,0), n=0, plot_title=True, IPR_in_title=True, dp=10)
    # plot_Psi_r(f, q_idx=(30,30), n=0, plot_title=True, IPR_in_title=True, dp=10)
    # idxs = [0,8,13]
    # for i in idxs:
    #     plot_IPR(f, bands=[i], bands_in_title=True)
    # data = np.load(f0)
    # basis = data['basis']
    # ABS.plot_basis_states(basis)
