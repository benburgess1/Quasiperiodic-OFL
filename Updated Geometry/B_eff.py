import numpy as np
import matplotlib.pyplot as plt

sz = np.array([[1, 0],
               [0, -1]])
sp = np.array([[0, 1],
               [0, 0]])
sm = np.array([[0, 0],
               [1, 0]])

def calc_n(x_vals, y_vals, U0=0.1, U=None, N=5, V0=0.05, R=8, save=True, 
               filename=None):
    l = np.arange(R)
    G_vects = np.column_stack((np.cos(2*np.pi*l/R), np.sin(2*np.pi*l/R)))
    g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    if U is None:
        U = -U0 * np.exp(-1j * 2* np.pi * N * l / R)
    Nx = x_vals.size
    Ny = y_vals.size
    Nr = Nx * Ny
    nx = np.zeros((Nx,Ny))
    ny = np.zeros((Nx,Ny))
    nz = np.zeros((Nx,Ny))
    print('Evaluating Bloch vector directions:')
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            print(f'Evaluating point {i*Ny+j+1} out of {Nr}...' + 10*' ', end='\r')
            Bp = 0
            Bz = 0
            r = np.array([x,y])
            for l in range(G_vects.shape[0]):
                Bp += U[l] * np.exp(-1j * r.dot(G_vects[l,:]))
                Bz += -2 * V0 * np.cos(r.dot(g_vects[l,:]))
            Bx = -np.real(np)
            By = -np.imag(np)
            B = np.sqrt(Bx**2 + By**2 + Bz**2)
            nx[i,j] = Bx / B
            ny[i,j] = By / B
            nz[i,j] = Bz / B
    print('\nDone')
    B_eff = np.zeros((Nx-1, Ny-1), dtype=np.complex128)
    print('Evaluating real-space Berry curvature:')
    for i in range(Nx-1):
        for j in range(Ny-1):
            print(f'Evaluating point {i*Ny+j+1} out of {(Nx-1)*(Ny-1)}...' + 10*' ', end='\r')
            n0 = np.array([nx[i,j], ny[i,j], nz[i,j]])
            n1 = np.array([nx[i+1,j], ny[i+1,j], nz[i+1,j]])
            n12 = np.array([nx[i+1,j+1], ny[i+1,j+1], nz[i+1,j+1]])
            n2 = np.array([nx[i,j+1], ny[i,j+1], nz[i,j+1]])
            U1 = np.vdot(n0, n1)
            U2 = np.vdot(n1, n12)
            U3 = np.vdot(n12, n2)
            U4 = np.vdot(n2, n0)
            curv = np.angle(U1) + np.angle(U2) + np.angle(U3) + np.angle(U4)
            if np.abs(curv) > np.pi:
                # print('\n\nWarning: curvature outside (-pi,pi] range. Manually adjusting.')
                curv = (curv + np.pi) % (2*np.pi) - np.pi
            elif curv == -np.pi:
                curv = np.pi
            B_eff[i,j] = curv
    print('\nDone')
    if save:
        if filename is None:
            filename = 'Updated Geometry/Data/BlochVector_R' + str(R) + '_U' + str(np.round(U0,4)) + '_N' + str(N) + '_V' + str(np.round(V0,4)) + '.npz'
        np.savez(filename, x_vals=x_vals, y_vals=y_vals, nx_vals=nx, ny_vals=ny, 
                 nz_vals=nz, U0=U0, V0=V0, R=R, N=N)
    else:
        return nx, ny, nz
    

def calc_B_eff(x_vals, y_vals, U0=0.1, U=None, N=5, V0=0.05, R=8, save=True, 
               filename=None):
    l = np.arange(R)
    G_vects = np.column_stack((np.cos(2*np.pi*l/R), np.sin(2*np.pi*l/R)))
    g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    if U is None:
        U = -U0 * np.exp(-1j * 2* np.pi * N * l / R)
    Nx = x_vals.size
    Ny = y_vals.size
    Nr = Nx * Ny
    evects_arr = np.zeros((2, Nx, Ny), dtype=np.complex128)
    print('Evaluating eigenvectors:')
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            print(f'Evaluating point {i*Ny+j+1} out of {Nr}...' + 10*' ', end='\r')
            r = np.array([x,y])
            H = np.zeros((2,2), dtype=np.complex128)
            for l in range(R):
                H += 2 * V0 * np.cos(r.dot(g_vects[l,:])) * sz
                H += U[l] * np.exp(-1j * r.dot(G_vects[l,:])) * sp
                H += np.conj(U[l]) * np.exp(1j * r.dot(G_vects[l,:])) * sm
            evals, evects = np.linalg.eigh(H)
            evects_arr[:,i,j] = evects[:,0]
    print('\nDone')
    B_eff = np.zeros((Nx-1, Ny-1), dtype=np.complex128)
    print('Evaluating real-space Berry curvature:')
    for i in range(Nx-1):
        for j in range(Ny-1):
            print(f'Evaluating point {i*(Ny-1)+j+1} out of {(Nx-1)*(Ny-1)}...' + 10*' ', end='\r')
            n0 = evects_arr[:,i,j]
            n1 = evects_arr[:,i+1,j]
            n12 = evects_arr[:,i+1,j+1]
            n2 = evects_arr[:,i,j+1]
            U1 = np.vdot(n0, n1)
            U2 = np.vdot(n1, n12)
            U3 = np.vdot(n12, n2)
            U4 = np.vdot(n2, n0)
            curv = np.angle(U1) + np.angle(U2) + np.angle(U3) + np.angle(U4)
            if np.abs(curv) > np.pi:
                # print('\n\nWarning: curvature outside (-pi,pi] range. Manually adjusting.')
                curv = (curv + np.pi) % (2*np.pi) - np.pi
            elif curv == -np.pi:
                curv = np.pi
            B_eff[i,j] = curv
    print('\nDone')
    if save:
        if filename is None:
            filename = 'Updated Geometry/Data/B_eff_R' + str(R) + '_U' + str(np.round(U0,4)) + '_N' + str(N) + '_V' + str(np.round(V0,4)) + '.npz'
        np.savez(filename, x_vals=x_vals, y_vals=y_vals, B_eff=B_eff, 
                 U0=U0, V0=V0, R=R, N=N)
    else:
        return B_eff


def add_centred_arrow(ax, x, y, nx, ny, color='k', lf=0.1, hl=3, hw=2, w=1):
    r = np.array([x,y])
    e = np.array([nx, ny])
    e *= lf
    r1 = r - 0.5*e
    r2 = r + 0.5*e 
    ax.annotate(xy=r2, xytext=r1, text='',
                arrowprops=dict(width=w, headwidth=hw, headlength=hl, 
                                color=color))


def plot_n(filename, cmap=plt.colormaps['bwr'], levels=None, arrowcolor='k',
               lf=0.1, hl=3, hw=2, w=1, axlim=None, plot_cbar=True, plot_title=True,
               xticks=None, yticks=None):
    data = np.load(filename)
    x_vals = data['x_vals']
    y_vals = data['y_vals']
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    nx_vals = data['nx_vals']
    ny_vals = data['ny_vals']
    nz_vals = data['nz_vals']
    fig, ax = plt.subplots()
    if levels is None:
        vmax = np.max(np.abs(nz_vals))
        levels = np.linspace(-vmax, vmax, 200)
        ticks = [-vmax, 0, vmax]
    plot = ax.contourf(xx, yy, nz_vals, cmap=cmap, levels=levels)
    for i,x in enumerate(x_vals):
        for j,y in enumerate(y_vals):
            Bx = nx_vals[i,j]
            By = ny_vals[i,j]
            add_centred_arrow(ax, x, y, Bx, By, color=arrowcolor, lf=lf, hl=hl, 
                              hw=hw, w=w)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(r'$n_z$', rotation=0)
    ax.set_xlabel(r'$x |\mathbf{G}|$')
    ax.set_ylabel(r'$y |\mathbf{G}|$')
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
        U0 = np.round(data['U0'],4)
        V0 = np.round(data['V0'],4)
        N = data['N']
        R = data['R']
        title_str = (r'$n(\mathbf{r}), R=$' + str(R) + r', $U=$' + str(U0) 
                     + r', $V=$' + str(V0) + r', $N=$' + str(N))
        ax.set_title(title_str)
    plt.show()


def plot_n_component(filename, component='x', cmap=plt.colormaps['bwr'], levels=None,
                     axlim=None, plot_cbar=True, plot_title=True,
                     xticks=None, yticks=None):
    data = np.load(filename)
    x_vals = data['x_vals']
    y_vals = data['y_vals']
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    if component == 'x':
        n = data['nx_vals']
        zlab = r'$n_x$'
    elif component == 'y':
        n = data['ny_vals']
        zlab = r'$n_y$'
    elif component == 'z':
        n = data['nz_vals']
        zlab = r'$n_z$'
    fig, ax = plt.subplots()
    if levels is None:
        vmax = np.max(np.abs(n))
        levels = np.linspace(-vmax, vmax, 200)
        ticks = [-vmax, 0, vmax]
    plot = ax.contourf(xx, yy, n, cmap=cmap, levels=levels)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab, rotation=0)
    ax.set_xlabel(r'$x |\mathbf{G}|$')
    ax.set_ylabel(r'$y |\mathbf{G}|$')
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
        U0 = np.round(data['U0'],4)
        V0 = np.round(data['V0'],4)
        N = data['N']
        R = data['R']
        title_str = (r'$n(\mathbf{r}), R=$' + str(R) + r', $U=$' + str(U0) 
                     + r', $V=$' + str(V0) + r', $N=$' + str(N))
        ax.set_title(title_str)
    plt.show()


def plot_B_eff(filename, shift_r=True, cmap=plt.colormaps['bwr'], levels=None,
                     axlim=None, plot_cbar=True, plot_title=True,
                     xticks=None, yticks=None):
    data = np.load(filename)
    x_vals = data['x_vals'][:-1]
    y_vals = data['y_vals'][:-1]
    if shift_r:
        dx = x_vals[1] - x_vals[0]
        x_vals += dx/2
        dy = y_vals[1] - y_vals[0]
        y_vals += dy/2
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    B_eff = np.real(data['B_eff'])
    fig, ax = plt.subplots()
    if levels is None:
        vmax = np.max(np.abs(B_eff))
        levels = np.linspace(-vmax, vmax, 200)
        ticks = [-vmax, 0, vmax]
    plot = ax.contourf(xx, yy, B_eff, cmap=cmap, levels=levels)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(r'$|B_{eff}|$', rotation=0)
    ax.set_xlabel(r'$x |\mathbf{G}|$')
    ax.set_ylabel(r'$y |\mathbf{G}|$')
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
        U0 = np.round(data['U0'],4)
        V0 = np.round(data['V0'],4)
        N = data['N']
        R = data['R']
        title_str = (r'$B_{eff}(\mathbf{r}), R=$' + str(R) + r', $U=$' + str(U0) 
                     + r', $V=$' + str(V0) + r', $N=$' + str(N))
        ax.set_title(title_str)
    plt.show()




if __name__ == '__main__':
    x = np.linspace(-50,50,101)
    # x = np.arange(-10,11, dtype=np.float64)
    y = np.copy(x)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dA = dx * dy
    # xx, yy = np.meshgrid(x, y, indexing='ij')
    # r_vals = np.column_stack((xx.flatten(), yy.flatten()))
    U0 = 0.2
    V0 = 0.3
    f = 'Updated Geometry/Data/B_eff_R5_U0.2_N5_V0.3.npz'
    calc_B_eff(x, y, U0=U0, V0=V0, R=5, filename=f)
    # axlim = (-50, 50)
    # xticks = np.arange(-50, 51, 10)
    # yticks = np.arange(-50, 51, 10)

    data = np.load(f)
    B_eff = data['B_eff']
    B_eff_av = np.mean(B_eff, axis=None)
    print(B_eff_av)
    print(B_eff_av/(dA*np.pi))

    plot_B_eff(f, shift_r=True, axlim=None)#, axlim=axlim, lf=1, xticks=xticks, yticks=yticks)
    # for c in ['x', 'y', 'z']:
    #     plot_B_component(f, component=c, axlim=axlim, xticks=xticks, yticks=yticks)
    # plot_B_mag(f, axlim=axlim, xticks=xticks, yticks=yticks)


