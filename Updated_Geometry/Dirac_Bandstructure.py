import numpy as np
import matplotlib.pyplot as plt
from QBZ_Bandstructure import calc_curv, plot_evals_surf, plot_BS_path, plot_curv


### Useful matrices
sx = np.array([[0, 1],
               [1, 0]], dtype=np.complex128)
sy = np.array([[0, -1j],
               [1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0],
               [0, -1]], dtype=np.complex128)
sp = np.array([[0, 1],
               [0, 0]], dtype=np.complex128)
sm = np.array([[0, 0],
               [1, 0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def calc_H(q, v=1., V=0., W=0., Dx=0., Dy=0., Dz=0.):
    # Base (kinetic + U-couplings) Hamiltonian:
    Hx = np.kron(sz, I2)
    Hy = np.kron(sx, sx)
    H = Hx * q[0] * v + Hy * q[1] * v
    # Original Couplings:
    # H += V * np.kron(sz, sy)
    # H += W * np.kron(sy, I2)

    # Updated Couplings:
    # V couplings
    H += V * np.kron(sy, sx)
    # W couplings
    H += W * np.kron(sy, I2)

    # D couplings
    H += Dx * np.kron(sz, sx) 
    H += Dy * np.kron(sx, I2) 
    # H += Dx * np.kron(I2, sy)     # These forms give equivalent bandstructures, but are just different
    # H += Dy * np.kron(sy, sz)     # eigenvectors - stick to first forms for simplicity
    #H += Dz * np.kron()        Need to work this one out; but it doesn't really do anything useful
    return H


def calc_BS_path(q_vals, **kwargs):
    evals = np.zeros((q_vals.shape[0],4))
    for i in range(q_vals.shape[0]):
        print(f'Evaluating q point {i+1} out of {q_vals.shape[0]}...' + 10*' ', end='\r')
        q = q_vals[i,:]
        H = calc_H(q, **kwargs)
        evals[i,:] = np.linalg.eigvalsh(H)
    return evals


def calc_evects(qx_vals, qy_vals, **kwargs):
    print('Calculating bandstructure:')
    evals = np.zeros((4, qx_vals.size, qy_vals.size))
    evects = np.zeros((4, 4, qx_vals.size, qy_vals.size), dtype=np.complex128)
    for i, qx in enumerate(qx_vals):
        for j, qy in enumerate(qy_vals):
            print(f'Evaluating q value {i*qy_vals.size + j + 1} out of {qx_vals.size * qy_vals.size}...' + 10*' ', end='\r')
            H = calc_H(q=np.array([qx,qy]), **kwargs)
            evals[:, i, j], evects[:, :, i, j] = np.linalg.eigh(H)
    print('\nDone')
    return evals, evects


if __name__ == '__main__':
    ### Constants, parameters etc.
    V = -0.001
    W = 0.0009
    Dx = 0.01
    Dy = 0.0
    Dz = 0.0

    # Plot path bandstructure along qx and qy directions
    qmax = 0.05
    N = 1000
    qx_vals = np.column_stack((np.linspace(-qmax, qmax, N), np.zeros(N)))
    qy_vals = np.column_stack((np.zeros(N), np.linspace(-qmax, qmax, N)))
    dq = np.column_stack((np.linspace(-0.01, 0.01, 1000), np.linspace(-0.01, 0.01, 1000)))
    evals_x = calc_BS_path(qx_vals, V=V, W=W, Dx=Dx, Dy=Dy, Dz=Dz)
    evals_y = calc_BS_path(qy_vals, V=V, W=W, Dx=Dx, Dy=Dy, Dz=Dz)
    plot_BS_path(evals=evals_x, q_vals=qx_vals, x='qx', 
                 title_params={'V':V, 'W':W, 'Dx':Dx, 'Dy':Dy})
    # plot_BS_path(evals=evals_y, q_vals=qy_vals, x='qy', 
    #              title_params={'V':V, 'W':W, 'Dx':Dx, 'Dy':Dy})
    

    # Calculate eigenvalues, eigenvectors and curvature over a 2d (qx, qy) surface,
    # plot bandstructure, plot curvature
    qmax = 0.05
    N = 100
    qx_vals = np.linspace(-qmax, qmax, N)
    qy_vals = np.copy(qx_vals)

    evals, evects = calc_evects(qx_vals, qy_vals, V=V, W=W, Dx=Dx, Dy=Dy, Dz=Dz)
    # plot_evals_surf(evals, qx_vals, qy_vals, title_params={'V':V, 'W':W, 'Dx':Dx, 'Dy':Dy})
    n_bands = np.arange(2)
    curv = calc_curv(evects, n_bands=n_bands, NonAb=True)
    plot_curv(curv, qx_vals, qy_vals, n_bands=n_bands, 
              title_params={'V':V, 'W':W, 'Dx':Dx, 'Dy':Dy},
              bands_in_title=False, dp=5)

