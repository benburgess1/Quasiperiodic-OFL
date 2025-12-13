import numpy as np
import matplotlib.pyplot as plt

### Useful matrices
sx = np.array([[0, 1],
               [1, 0]])
sy = np.array([[0, -1j],
               [1j, 0]])
sz = np.array([[1, 0],
               [0, -1]])
sp = np.array([[0, 1],
               [0, 0]])
sm = np.array([[0, 0],
               [1, 0]])
I2 = np.eye(2)


### Constants, matrix elements, etc.
Umag = 0.1
l = np.arange(5)
phi0 = 0
N = 3
U = -Umag * np.exp(1j * (phi0 - 2 * np.pi * N * l / 5))
dphi = np.angle(U[1]) - np.angle(U[0])
w = np.exp(1j*dphi)
G = np.column_stack((np.cos(2*np.pi*l/5), np.sin(2*np.pi*l/5)))


def calc_H(q, U=np.zeros(5), G=G, V=0., Dx=0., Dy=0., Dz=0., W=0.):
    H = np.zeros((10, 10), dtype=np.complex128)
    # U couplings:
    H_U = np.zeros((10, 10), dtype=np.complex128)
    for i in range(5):
        H_U[(i+3)%10, i] = U[i]
        H_U[(i+8)%10, (i+5)%10] = np.conj(U[i])
        # H_U[(i+3)%10, i] = U[i]
        # H_U[(i+8)%10, (i+5)%10] = np.conj(U[i])
    H += (H_U + H_U.conj().T)
    # V couplings:
    H_V = np.zeros((10, 10), dtype=np.complex128)
    for i in range(5):
        H_V[(2*i+4)%10, 2*i] = -V
        H_V[(2*i+5)%10, 2*i+1] = V
    H += (H_V + H_V.conj().T)
    # KE:
    H_kin = np.zeros((10, 10), dtype=np.complex128)
    # b = [G[0,:],
    #      G[0,:] - G[2,:],
    #      G[0,:] - G[2,:],]
    b = {0:np.zeros(2), 
         3:G[0,:], 
         6:G[0,:] - G[4,:],
         9:G[0,:] - G[4,:] + G[3,:],
         2:G[0,:] - G[4,:] + G[3,:] - G[2,:],
         5:G[0,:] - G[4,:] + G[3,:] - G[2,:] + G[1,:],
         8:-G[4,:] + G[3,:] - G[2,:] + G[1,:],
         1:G[3,:] - G[2,:] + G[1,:],
         4:-G[2,:] + G[1,:],
         7:G[1,:]}
    for i, v in b.items():
        H_kin[i,i] = np.sum((q-v)**2)
    H += H_kin
    return H


def calc_BS_path(q_vals, **kwargs):
    evals = np.zeros((q_vals.shape[0],10))
    for i in range(q_vals.shape[0]):
        print(f'Evaluating q point {i+1} out of {q_vals.shape[0]}...' + 10*' ', end='\r')
        q = q_vals[i,:]
        H = calc_H(q, **kwargs)
        evals[i,:] = np.linalg.eigvalsh(H)
    return evals


def plot_BS_path(evals, q_vals, x='qx', alternating=False, 
                 Elim=None, title_params={}, dp=5):
    fig,ax = plt.subplots()
    if x == 'qx':
        x_vals = q_vals[:,0]
        xlab = r'$q_x$'
    elif x == 'qy':
        x_vals = q_vals[:,1]
        xlab = r'$q_y$'
    elif x == '|q|':
        x_vals = np.linalg.norm(q_vals, axis=1)
        xlab = r'$|q|$'
    else:
        x_vals = np.arange(q_vals.shape[0])
        xlab = 'N'
    for i in range(evals.shape[1]):
        if alternating:
            styles = ['-', ':']
            colours = ['k', 'r']
            ax.plot(x_vals, evals[:,i], ls=styles[i%2], c=colours[i%2])
        else:
            ax.plot(x_vals, evals[:,i], ls='-', c='b')
    # ax.set_xticks([0, 99, 198, 297])
    # ax.set_xticklabels([r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'$E/E_R$')
    # if U_mag is not None:
    #     ax.set_title(f'U = {U_mag}, V = {V_mag}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}, N = {N}, W = {W}')
    title_str = ''
    for k,v in title_params.items():
        title_str += ', ' + k + r'$=$' + str(np.round(v, dp))
    # title_str = f'U = {U_mag}, V = {V_mag}, N = {N}, Dx = {Dx}, Dy = {Dy}, Dz = {Dz}, W = {W}'
    title_str = title_str[2:]
    ax.set_title(title_str)
    if Elim is not None:
        ax.set_ylim(*Elim)
    plt.show()


def calc_q_GMKG():
    qG = np.zeros(2)
    qM = np.array([0.5, 0])
    qK = np.array([0.5, 0.5*np.tan(np.pi/5)])
    qGM = np.column_stack((np.linspace(qG[0], qM[0], 100),
                           np.linspace(qG[1], qM[1], 100)))
    qMK = np.column_stack((np.linspace(qM[0], qK[0], 100)[1:],
                           np.linspace(qM[1], qK[1], 100)[1:]))
    qKG = np.column_stack((np.linspace(qK[0], qG[0], 100)[1:],
                           np.linspace(qK[1], qG[1], 100)[1:]))
    return np.vstack((qGM, qMK, qKG))


def plot_BS_GMKG(evals, alternating=False, title_params={}, dp=5):
    fig,ax = plt.subplots()
    x = np.arange(298)
    for i in range(10):
        if alternating:
            styles = ['-', ':']
            colours = ['k', 'r']
            ax.plot(x, evals[:,i], ls=styles[i%2], c=colours[i%2])
        else:
            ax.plot(x, evals[:,i], ls='-', c='b')
    ax.set_xticks([0, 99, 198, 297])
    ax.set_xticklabels([r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])
    ax.set_ylabel(r'$E/E_R$')
    title_str = ''
    for k,v in title_params.items():
        title_str += ', ' + k + r'$=$' + str(np.round(v, dp))
    title_str = title_str[2:]
    ax.set_title(title_str)
    plt.show()


if __name__ == '__main__':
    Umag = 0.05
    Vmag = 0.02
    l = np.arange(5)
    phi0 = 0
    N = 3
    U = -Umag * np.exp(1j * (phi0 - 2 * np.pi * N * l / 5))
    G = np.column_stack((np.cos(2*np.pi*l/5), np.sin(2*np.pi*l/5)))

    q_K = np.array([0.5, 0.5*np.tan(np.pi/5)])
    H_K = calc_H(q=q_K, U=U, G=G, V=Vmag)
    evals, evects = np.linalg.eigh(H_K)
    print(np.round(evals - np.sum(q_K**2), 4))

    qmax = 0.7
    q_vals = np.column_stack((np.linspace(0, qmax*np.cos(np.pi/5), 200),
                              np.linspace(0, qmax*np.sin(np.pi/5), 200)))
    # qmax = 0.7
    # q_vals = np.column_stack((np.linspace(0, qmax, 200),
    #                           np.zeros(200)))
    # qx_vals = np.linspace(0., 0.7, 200)
    # qy_vals = np.zeros(200)
    # q_vals = np.column_stack((qx_vals, qy_vals))

    # evals = calc_BS_path(q_vals=q_vals, U=U, G=G, V=Vmag)
    # # print(evals)
    # # print(evals.shape)
    # plot_BS_path(evals, q_vals, x='|q|', alternating=False, 
    #              title_params={r'$U$':Umag, r'$V$':Vmag})
    
    q_vals = calc_q_GMKG()
    evals = calc_BS_path(q_vals=q_vals, U=U, G=G, V=Vmag)
    plot_BS_GMKG(evals, title_params={r'$U$':Umag, r'$V$':Vmag})
    

    
