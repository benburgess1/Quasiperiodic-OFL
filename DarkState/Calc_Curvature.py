import numpy as np

N = 8
l = np.arange(N)
q8 = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
phi8 = np.zeros((N,2))

def s_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    s = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            s[i] += np.exp(1j * ((-1)**(i+1) * q.dot(r) + phi[a,i]))
    return s


def d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += (-1)**i * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]))
    return d


def del_p_d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    q_mag = np.linalg.norm(q_vals[0,:])
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += -1j * q_mag * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]
                                           + 2*np.pi*a/N))
    return d


def del_m_d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    q_mag = np.linalg.norm(q_vals[0,:])
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += -1j * q_mag * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]
                                           - 2*np.pi*a/N))
    return d


def del_x_d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += -1j * q[0] * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]))
    return d


def del_y_d_state(r, q_vals=None, phi=None, N=8, **kwargs):
    if q_vals is None:
        l = np.arange(N)
        q_vals = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    else:
        N = q_vals.shape[0]
    if phi is None:
        phi = np.zeros((N, 2))
    d = np.zeros(2, dtype=np.complex128)
    for i in range(2):
        for a, q in enumerate(q_vals):
            d[i] += -1j * q[1] * np.exp(1j * ((-1)**(i+1) * q.dot(r) 
                                           - phi[a,(i+1)%2]))
    return d


def verify_derivative(N=8, p1=0, p2=None):
    l = np.arange(N)
    q = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    q_mag = np.linalg.norm(q[0,:])
    if p2 is None:
        p2 = N - 1 - p1
    phi = np.zeros((N, 2))
    for a in range(N):
        phi[a,0] = - 2 * np.pi * a * p1 / N
        phi[a,1] = - 2 * np.pi * a * p2 / N
    # phi = np.random.rand(8,2) * 2*np.pi
    r = np.random.rand(2)
    d = d_state(r, q_vals=q, phi=phi)
    s = s_state(r, q_vals=q, phi=phi)
    del_p_d = del_p_d_state(r, q_vals=q, phi=phi)
    diff = -1j * q_mag * s - del_p_d
    print(f'Momentum transfers:\n{q}')
    print(f'Phase factors:\n{phi}')
    print(f'Position: {r}')
    print(f's = {s}')
    print(f'd = {d}')
    print(f'del_p_d = {del_p_d}')
    print(f'|-i|q|s - del_p_d| = {np.linalg.norm(diff)}')   # Should vanish


def calc_curv_point(r, **kwargs):
    d = d_state(r, **kwargs)
    dx = del_x_d_state(r, **kwargs)
    dy = del_y_d_state(r, **kwargs)
    dydx = np.vdot(dy, dx)
    ddx = np.vdot(d, dx)
    dyd = np.vdot(dy, d)
    dd = np.vdot(d, d)
    curv = np.imag(-2 * dydx / dd + 2 * ddx * dyd / dd**2)
    return curv

def verify_curvature(N=8, p1=0, p2=None, h=0.001, r=np.random.rand(2)):
    l = np.arange(N)
    q = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    if p2 is None:
        p2 = 1 - p1
    phi = np.zeros((N, 2))
    for a in range(N):
        phi[a,0] = - 2 * np.pi * a * p1 / N
        phi[a,1] = - 2 * np.pi * a * p2 / N
    # phi = np.random.rand(8,2) * 2*np.pi
    d00 = d_state(r, q_vals=q, phi=phi)
    d00 /= np.linalg.norm(d00)
    d10 = d_state(r + np.array([h,0]), q_vals=q, phi=phi)
    d10 /= np.linalg.norm(d10)
    d11 = d_state(r + np.array([h,h]), q_vals=q, phi=phi)
    d11 /= np.linalg.norm(d11)
    d01 = d_state(r + np.array([0,h]), q_vals=q, phi=phi)
    d01 /= np.linalg.norm(d01)
    # Ax0 = 1j * np.vdot(d00, d10-d00) / h
    # Ax1 = 1j * np.vdot(d01, d11-d01) / h
    # Ay0 = 1j * np.vdot(d00, d01-d00) / h
    # Ay1 = 1j * np.vdot(d10, d11-d10) / h
    # curv_num = ((Ay1 - Ay0) - (Ax1 - Ax0)) / h
    dxd = (d10 - d00) / h
    dyd = (d01 - d00) / h
    curv_num = 1j * (np.vdot(dyd, dxd) - np.vdot(dxd, dyd))

    curv_ana = calc_curv_point(r, q_vals=q, phi=phi)
    print(f'Point r = {r}')
    print(f'Numerical curvature: {curv_num}')
    print(f'Analytical curvature: {curv_ana}')


def calc_curv_surface(x_vals, y_vals, save=True, filename=None, **kwargs):
    Nx = x_vals.size
    Ny = y_vals.size
    curv_vals = np.zeros((Nx, Ny))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            print(f'Evaluating point {i*Ny + j + 1} out of {Nx * Ny}...' + 10*' ', end='\r')
            curv_vals[i,j] = calc_curv_point(r=np.array([x,y]), **kwargs)
    print('\nDone')
    if save:
        if filename is None:
            filename = 'Data.npz'
        np.savez(filename, x_vals=x_vals, y_vals=y_vals, curv_vals=curv_vals, 
                 **kwargs)
    else:
        return curv_vals


if __name__ == '__main__':
    N = 8
    l = np.arange(N)
    q = np.column_stack((np.cos(2*np.pi*l/N), np.sin(2*np.pi*l/N)))
    q_mag = np.linalg.norm(q[0,:])
    p1 = 4
    p2 = -3
    phi = np.zeros((N, 2))
    for a in range(N):
        phi[a,0] = 2 * np.pi * a * p1 / N
        phi[a,1] = 2 * np.pi * a * p2 / N

    x_vals = np.linspace(-50, 50, 250)
    y_vals = np.linspace(-50, 50, 250)
    f = 'DarkState/Data/B_eff_p14_p2-3.npz'
    calc_curv_surface(x_vals, y_vals, save=True, filename=f, q_vals=q, phi=phi, 
                      N=N, p1=p1, p2=p2)

    # phi = np.random.rand(8,2) * 2*np.pi
    # r = np.random.rand(2)
    # d = d_state(r, q_vals=q, phi=phi)
    # s = s_state(r, q_vals=q, phi=phi)
    # del_p_d = del_p_d_state(r, q_vals=q, phi=phi)
    # del_m_d = del_m_d_state(r, q_vals=q, phi=phi)
    # del_x_d = del_x_d_state(r, q_vals=q, phi=phi)
    # del_y_d = del_y_d_state(r, q_vals=q, phi=phi)
    # diff = -1j * q_mag * s - del_p_d
    # dd = np.vdot(d,d)



