import numpy as np
import matplotlib.pyplot as plt

N=3
G0 = np.array([1,0])
G1 = np.array([np.cos(2*np.pi/5),np.sin(2*np.pi/5)])
U0 = -0.01
U1 = -0.01 * np.exp(-1j*2*np.pi*N/5)
V0 = -0.01


def calc_H(q, G0=G0, G1=G1, U0=U0, U1=U1, V0=V0):
    H = np.array([[np.sum(q**2), np.conj(U0), np.conj(U1)],
                  [U0, np.sum((q-G0)**2), np.conj(V0)],
                  [U1, V0, np.sum((q-G1)**2)]])
    return H


def calc_midpoint(G1, G2):
    M = np.array([G1, G2])
    v = 0.5*np.array([np.sum(G1**2), np.sum(G2**2)])
    return np.linalg.inv(M) @ v


def plot_q_path(q_path, q0, G0=G0, G1=G1, spin='down'):
    fig,ax = plt.subplots()
    if spin == 'down':
        c1, c2 = 'b', 'r'
    elif spin == 'up':
        c1, c2 = 'r', 'b'
    ax.plot([0], [0], marker='o', color=c1)
    ax.plot(G0[0], G0[1], marker='o', color=c2)
    ax.plot(G1[0], G1[1], marker='o', color=c2)
    ax.plot(q0[0], q0[1], marker='x', color='gold')
    ax.plot(q_path[0,:], q_path[1,:], color='k', ls='-', marker=None)
    ax.annotate('', xytext=(0,0), xy=(G0[0],G0[1]), 
                    arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                    color='c'))
    ax.annotate('', xy=(0,0), xytext=(G1[0],G1[1]), 
                    arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                    color='c'))
    ax.annotate('', xytext=(G0[0],G0[1]), xy=(G1[0],G1[1]), 
                    arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                    color='limegreen'))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.show()


def calc_E(q_path, **kwargs):
    E_vals = np.zeros((3, q_path.shape[1]))
    for i in range(q_path.shape[1]):
        q = q_path[:,i]
        H = calc_H(q, **kwargs)
        evals, evects = np.linalg.eigh(H)
        E_vals[:,i] = evals
    return(E_vals)


def plot_E(E_vals, t=None, N=3, U=0.01, V=0.005, dq=0.1,
           x='t', n=1000, l=0):
    # t /= np.pi
    fig,ax = plt.subplots()
    if x == 't':
        ax.plot(t/np.pi, E_vals[0,:], color='b', label='E0')
        ax.plot(t/np.pi, E_vals[1,:], color='r', label='E1')
        ax.plot(t/np.pi, E_vals[2,:], color='cyan', label='E2')
        ax.set_xlabel(r'$\theta / \pi$')
    elif x == 'wedge':
        t = np.arange(E_vals.shape[1])
        ax.plot(t, E_vals[0,:], color='b', label='E0')
        ax.plot(t, E_vals[1,:], color='r', label='E1')
        ax.plot(t, E_vals[2,:], color='cyan', label='E2')
        ax.set_xticks([0, n-1, 2*n-2, 3*n-3])
        ax.set_xticklabels([0, 1, 2, 3])
        ax.set_xlabel('q')

    ax.set_ylabel('E')
    ax.set_title(f'U = {U}, V = {V}, N = {N}, dq = {dq}, l = {l}')
    ax.legend()
    plt.show()


def calc_A(q_path, idx=0, **kwargs):
    A_vals = np.zeros((3, q_path.shape[1]-1))
    for i in range(q_path.shape[1]-1):
        # dq = np.linalg.norm(q_path[:,i+1] - q_path[:,i])
        q1 = q_path[:,i]
        H1 = calc_H(q1, **kwargs)
        evals1, evects1 = np.linalg.eigh(H1)
        phi1 = np.angle(evects1[idx, :])
        evects1 *= np.exp(-1j*phi1)
            
    
        q2 = q_path[:,i+1]
        H2 = calc_H(q2, **kwargs)
        evals2, evects2 = np.linalg.eigh(H2)
        phi2 = np.angle(evects2[idx, :])
        evects2 *= np.exp(-1j*phi2)

        dq = np.linalg.norm(q2 - q1)
        d_evects = evects2 - evects1

        A_vals[:,i] = 1j*np.sum(np.conj(evects1) * d_evects, axis=0) / dq

    return A_vals

def plot_A(A_vals, t=None, N=3, U=0.01, V=0.005, dq=0.1, x='t', n=1000, l=0):
    fig,ax = plt.subplots()
    # t /= np.pi
    if x == 't':
        ax.plot(t[:-1]/np.pi, A_vals[0,:], color='b', label='A0')
        ax.plot(t[:-1]/np.pi, A_vals[1,:], color='r', label='A1')
        ax.plot(t[:-1]/np.pi, A_vals[2,:], color='cyan', label='A2')
        ax.set_xlabel(r'$\theta / \pi$')
    elif x == 'wedge':
        t = np.arange(A_vals.shape[1])
        ax.plot(t, A_vals[0,:], color='b', label='A0')
        ax.plot(t, A_vals[1,:], color='r', label='A1')
        ax.plot(t, A_vals[2,:], color='cyan', label='A2')
        ax.set_xticks([0, n-1, 2*n-2, 3*n-3])
        ax.set_xticklabels([0, 1, 2, 3])
        ax.set_xlabel('q')
    ax.set_ylabel('A')
    ax.set_title(f'U = {U}, V = {V}, N = {N}, dq = {dq}, l = {l}')
    ax.legend()
    plt.show()


def QBZ_path(q0, n=1000, dq=0.1, l=0, spin='down'):
    t = np.linspace(9*np.pi/10 + 2*np.pi*l/5, 15*np.pi/10 + 2*np.pi*l/5, n)
    if spin == 'up':
        t += np.pi
    q1 = dq*np.array([np.cos(t), np.sin(t)])
    q2 = np.row_stack((np.linspace(q1[0,-1], 0, n)[1:], 
                       np.linspace(q1[1,-1], 0, n)[1:]))
    q3 = np.row_stack((np.linspace(0, q1[0,0], n)[1:],
                       np.linspace(0, q1[1,0], n)[1:]))
    q_path = np.column_stack((q1,q2,q3)) + q0.reshape(2,1)
    return q_path


def calc_C_QBZ_path(A_vals, n=1000, dq=0.1):
    dt = 3*np.pi/(5*(n-1))
    # print(dt)
    C1 = np.sum(A_vals[:,:n], axis=1) * dq * dt / (2*np.pi)
    dqy = dq / (n-1)
    C2 = np.sum(A_vals[:,n-1:2*n-2], axis=1) * dqy / (2*np.pi)
    C3 = np.sum(A_vals[:,2*n-2:], axis=1) * dqy / (2*np.pi)
    C = C1 + C2 + C3
    return C


def coupling_vectors(plot=True):
    fig,ax = plt.subplots()
    l = np.arange(5)
    G_vects = np.column_stack((np.cos(2*np.pi*l/5), np.sin(2*np.pi*l/5)))
    g_vects = np.roll(G_vects, -1, axis=0) - G_vects

    for i in range(G_vects.shape[0]):
        G = G_vects[i,:]
        if i != 1:
            ax.annotate('', xytext=(0,0), xy=(G[0],G[1]), 
                        arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                        color='r'))
        else:
            ax.annotate('', xytext=(G[0],G[1]), xy=(0,0), 
                        arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                        color='r'))
        ax.plot(*G, color='k', marker='o', ms=5, zorder=5)
        G_perp = np.array([-G[1], G[0]])
        e = G_perp / np.linalg.norm(G_perp)
        m = G / 2
        L = 0.35
        r1 = m + L * e
        r2 = m - L * e
        ax.plot([r1[0], r2[0]], [r1[1], r2[1]], color='orange', ls=':')

        g = g_vects[i,:]
        g_perp = np.array([-g[1], g[0]])
        e = g_perp / np.linalg.norm(g_perp)
        m = g / 2 + G
        r1 = m + 0.2 * e
        r2 = m - 0.2 * e
        ax.plot([r1[0], r2[0]], [r1[1], r2[1]], color='deepskyblue', ls=':')

    ax.plot([0], [0], color='k', marker='o', ms=5, zorder=5)

    # AV.add_midpoints(ax, G_vects, color='orange', ms=8, mew=2)

    for i in range(G_vects.shape[0]):
        ax.annotate('', xytext=G_vects[i,:], 
                    xy=np.roll(G_vects, -1, axis=0)[i,:], 
                    arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                    color='limegreen'))
    
    g = -g_vects[1,:]
    ax.annotate('', xytext=(0,0), 
                    xy=g, 
                    arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                    color='limegreen'))
    ax.plot(*g, color='k', marker='o', ms=5, zorder=5)
    m = g/2
    g_perp = np.array([-g[1], g[0]])
    e = g_perp / np.linalg.norm(g_perp)
    L = 0.2
    r1 = m + L * e
    r2 = m - L * e
    ax.plot([r1[0], r2[0]], [r1[1], r2[1]], color='deepskyblue', ls=':')
    ax.annotate('', xy=G_vects[1,:], 
                    xytext=g, 
                    arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                    color='r'))
    v = g - G_vects[1,:]
    m = G_vects[1,:] + v/2
    v_perp = np.array([-v[1], v[0]])
    e = v_perp / np.linalg.norm(v_perp)
    L = 0.35
    r1 = m + L * e
    r2 = m - L * e
    ax.plot([r1[0], r2[0]], [r1[1], r2[1]], color='orange', ls=':')
    

    g = g_vects[4,:]
    ax.annotate('', xytext=(0,0), 
                    xy=g, 
                    arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                    color='limegreen'))
    ax.plot(*g, color='k', marker='o', ms=5, zorder=5)
    m = g/2
    g_perp = np.array([-g[1], g[0]])
    e = g_perp / np.linalg.norm(g_perp)
    L = 0.2
    r1 = m + L * e
    r2 = m - L * e
    ax.plot([r1[0], r2[0]], [r1[1], r2[1]], color='deepskyblue', ls=':')
    ax.annotate('', xytext=G_vects[0,:], 
                    xy=g, 
                    arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                    color='r'))
    v = g - G_vects[0,:]
    m = G_vects[0,:] + v/2
    v_perp = np.array([-v[1], v[0]])
    e = v_perp / np.linalg.norm(v_perp)
    L = 0.35
    r1 = m + L * e
    r2 = m - L * e
    ax.plot([r1[0], r2[0]], [r1[1], r2[1]], color='orange', ls=':')


    


    # ax.text(x=-0.3, y=0.0, s=r'$|\mathbf{q},\downarrow\rangle$', color='k', 
    #         fontsize=LARGE_FIG_FONTSIZE, verticalalignment='center')
    # ax.text(x=1.04, y=0.0, s=r'$|\mathbf{q}-\mathbf{G}_0,\uparrow\rangle$', color='k', 
    #         fontsize=LARGE_FIG_FONTSIZE, verticalalignment='center')
    # ax.text(x=0.15, y=1.05, s=r'$|\mathbf{q}-\mathbf{G}_1,\uparrow\rangle$', color='k', 
    #         fontsize=LARGE_FIG_FONTSIZE, verticalalignment='center')
    # ax.text(x=-1.2, y=0.71, s=r'$|\mathbf{q}-\mathbf{G}_2,\uparrow\rangle$', color='k', 
    #         fontsize=LARGE_FIG_FONTSIZE, verticalalignment='center')
    # ax.text(x=-1.2, y=-0.71, s=r'$|\mathbf{q}-\mathbf{G}_3,\uparrow\rangle$', color='k', 
    #         fontsize=LARGE_FIG_FONTSIZE, verticalalignment='center')
    # ax.text(x=0.15, y=-1.05, s=r'$|\mathbf{q}-\mathbf{G}_4,\uparrow\rangle$', color='k', 
    #         fontsize=LARGE_FIG_FONTSIZE, verticalalignment='center')
    # ax.text(x=0.5, y=-0.11, s=r'$U_0$', color='r', 
    #         fontsize=21.5, verticalalignment='center', ha='center')
    # ax.text(x=0.71, y=-0.1, s=r'$U_0$', color='r', 
    #         fontsize=21.5, verticalalignment='center')
    # ax.text(x=-0.05, y=0.51, s=r'$U_1^*$', color='r', 
    #         fontsize=21.5, verticalalignment='center')
    # # ax.text(x=-0.11, y=0.25, s=r'$U_1^*$', color='r', 
    # #         fontsize=21.5, verticalalignment='center')
    # ax.text(x=0.69, y=0.51, s=r'$V_0$', color='limegreen', 
    #         fontsize=21.5, verticalalignment='center')
    # # ax.text(x=0.47, y=0.84, s=r'$V_0$', color='limegreen', 
    # #         fontsize=21.5, verticalalignment='center')
    # ax.text(x=x, y=y, s=r'$\Phi_{0,\downarrow}^N$', color='navy', 
    #         fontsize=21.5, va='center', ha='center')

    ax.set_xlim(-1.3,1.3)
    ax.set_ylim(-1.3,1.3)
    ax.set_aspect('equal')

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if plot:
        plt.show()




if __name__ == '__main__':
    l = 2
    N = 3
    U = 0.001
    V = 0.0005

    # Spin-Down Case:
    G0 = np.array([np.cos(2*np.pi*l/5),np.sin(2*np.pi*l/5)])
    G1 = np.array([np.cos(2*np.pi*(l+1)/5),np.sin(2*np.pi*(l+1)/5)])
    U0 = -U * np.exp(-1j*2*np.pi*N*l/5)
    U1 = -U * np.exp(-1j*2*np.pi*N*(l+1)/5)
    V0 = V

    # Spin-Up Case:
    G0 = -np.array([np.cos(2*np.pi*l/5),np.sin(2*np.pi*l/5)])
    G1 = -np.array([np.cos(2*np.pi*(l+1)/5),np.sin(2*np.pi*(l+1)/5)])
    U0 = -U * np.exp(1j*2*np.pi*N*l/5)
    U1 = -U * np.exp(1j*2*np.pi*N*(l+1)/5)
    V0 = -V


    q0 = calc_midpoint(G0, G1)
    dq = 0.2
    # t = np.linspace(9*np.pi/10, 15*np.pi/10, 1000)
    t = np.linspace(0, 2*np.pi, 10000)
    dt = t[1] - t[0]
    # print(t)
    # print(dt)
    # print(t[-1] + dt)
    t = np.concatenate((t, np.array([t[-1]+dt])))
    # print(t)
    # q_path = dq*np.array([np.cos(t), np.sin(t)]) + q0.reshape(2,1)
    n = 10000
    q_path = QBZ_path(q0, n=n, dq=dq, l=l, spin='up')
    # print(q_path)
    # plot_q_path(q_path, q0, G0=G0, G1=G1, spin='up')
    # E_vals = calc_E(q_path, U0=U0, U1=U1, V0=V0, G0=G0, G1=G1)
    # plot_E(E_vals, t=t, N=N, U=U, V=V, dq=dq, n=n, x='wedge', l=l)
    # A_vals = calc_A(q_path, U0=U0, U1=U1, V0=V0, G0=G0, G1=G1, idx=0)
    # C = calc_C_QBZ_path(A_vals, n=n, dq=dq)
    # print(f'C = {C}')
    # # A_int = np.sum(A_vals, axis=1) * dq * dt
    # # print(A_int)
    # # print(A_int/(2*np.pi))
    # plot_A(A_vals, t=t, N=N, U=U, V=V, dq=dq, n=n, x='wedge', l=l)

    coupling_vectors(plot=True)
