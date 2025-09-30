import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
# import Approximant_Curvature as AC

l = np.arange(5)
G_vects_exact = np.column_stack((np.cos(2*np.pi*l/5),
                                 np.sin(2*np.pi*l/5)))


t = (1+np.sqrt(5))/2
# L = np.sqrt((1 + t**2 + 2*t**4 + t**6) / (4 * t**2))
# print(L)
# G_vects_analytical = (1/L) * np.array([[L, 0],
#                                        [1/(2*t), (1+t**2)/2],
#                                        [-t/2, (1+t**2)/(2*t)],
#                                        [-t/2, -(1+t**2)/(2*t)],
#                                        [1/(2*t), -(1+t**2)/2]])


G_vects_analytical = np.array([[1, 0],
                               [(-1+t)/2, np.sqrt(2+t)/2],
                               [(-1-3*t+t**2)/4, (np.sqrt(2+t)*(-1+t))/2],
                               [(-1-3*t+t**2)/4, -(np.sqrt(2+t)*(-1+t))/2],
                               [(-1+t)/2, -np.sqrt(2+t)/2]])
                            #    [(np.sqrt(2+t)*(-1+t))/2, (-1-3t+t**2)/4]])


G_vects_paper = np.array([[(1+t**2)/2, 1/(2*t)],
                          [(1+t**2)/(2*t), -t/2],
                          [-(1+t**2)/(2*t), -t/2],
                          [-(1+t**2)/2, 1/(2*t)]])


G_1_paper = -np.sum(G_vects_paper, axis=0)
# print(G_1_paper)

G_vects_paper = np.row_stack((G_1_paper, G_vects_paper))

# print(G_vects_paper)

# print(np.linalg.norm(G_vects_paper, axis=1))

t = 49/36
G_vects_approximant = np.array([[1, 0],
                               [(-1+t)/2, np.sqrt(2+t)/2],
                               [(-1-3*t+t**2)/4, (np.sqrt(2+t)*(-1+t))/2],
                               [(-1-3*t+t**2)/4, -(np.sqrt(2+t)*(-1+t))/2],
                               [(-1+t)/2, -np.sqrt(2+t)/2]])

# print(G_vects_approximant)

G_vects_approximant[:,0] *= 5184
G_vects_approximant[:,1] *= 432
# print(G_vects_approximant)


def square_approximant(a, G_vects=G_vects_analytical):
    return (1/a) * np.round(a*G_vects)


def hex_approximant(a, G_vects=G_vects_analytical):
    k1 = np.array([1/a, 0])
    k2 = (1/a) * np.array([0.5, np.sqrt(3)/2])
    # G_vects_approx = np.zeros(G_vects.shape)
    # Fix k2 contribution with y-component
    n2 = np.round(G_vects[:,1]/(np.sqrt(3)/(2*a)))
    rem = G_vects - np.outer(n2, k2)
    n1 = np.round(rem[:,0]*a)

    return np.outer(n1, k1) + np.outer(n2, k2)


def plot_vectors(vects, ax, color='k', width=1, headwidth=5, headlength=8):
    for i in range(vects.shape[0]):
        v = vects[i,:]
        ax.annotate('', xytext=(0,0), xy=(v[0],v[1]), 
                    arrowprops=dict(width=width, headwidth=headwidth, headlength=headlength, 
                                    color=color))
        

def add_square_grid(ax, a, L=4, color='r', ms=2, zorder=1):
    x = np.arange(-L,L+1) * (1/a)
    y = np.arange(-L,L+1) * (1/a)
    xx,yy = np.meshgrid(x,y)
    ax.plot(xx, yy, color=color, marker='o', ls='', ms=ms, zorder=zorder)


def add_hex_grid(ax, a, L=4, color='r'):
    k1 = np.array([1/a, 0])
    k2 = (1/a) * np.array([0.5, np.sqrt(3)/2])
    for i in range(-L,L+1):
        for j in range(-L,L+1):
            r = i*k1 + j*k2
            ax.plot(r[0], r[1], marker='o', ls='', color=color)


def add_g_vectors(ax, G_vects, color='limegreen', width=1, headwidth=5, headlength=8,
                  zorder=3):
    # g_vects = np.roll(G_vects, -1, axis=0) - G_vects
    for i in range(G_vects.shape[0]):
        ax.annotate('', xytext=G_vects[i,:], 
                    xy=np.roll(G_vects, -1, axis=0)[i,:], 
                    arrowprops=dict(width=width, headwidth=headwidth, headlength=headlength,
                                    color=color), zorder=zorder)
        

def calc_midpoint(G1, G2):
    M = np.array([G1, G2])
    v = 0.5*np.array([np.sum(G1**2), np.sum(G2**2)])
    return np.linalg.inv(M) @ v


def calc_all_midpoints(G_vects):
    midpoints = np.zeros(G_vects.shape)
    for i in range(G_vects.shape[0]):
        midpoints[i,:] = calc_midpoint(G_vects[i,:], G_vects[(i+1)%G_vects.shape[0],:])
    return midpoints
    

def add_midpoints(ax, G_vects, color='gold', arrow=False, ms=5, mew=2, connect=True):
    r_vals =  np.zeros_like(G_vects)
    for i in range(G_vects.shape[0]):
        r = calc_midpoint(G_vects[i,:], G_vects[(i+1)%G_vects.shape[0]])
        r_vals[i,:] = r
        # print(f'i = {i}, r = {r}')
        if arrow:
            ax.annotate('', xytext=np.zeros(2), xy=r, 
                        arrowprops=dict(width=1, headwidth=5, headlength=8, 
                                        color=color))
    if connect:
        ls = ':'
        r_vals = np.row_stack((r_vals, r_vals[0,:]))
    else:
        ls = ''
    ax.plot(r_vals[:,0], r_vals[:,1], marker='x', color=color, ls=ls, ms=ms, mew=mew)


def add_cutoff_radius(ax, c=2.5, color='k'):
    circ = mpl.patches.Circle((0,0), radius=c, ec=color, ls=':',
                              fill=False)
    ax.add_patch(circ)


def square_BZ(dq=None, a=3, r0=np.zeros(2), color='k', lw=1):
    if dq is None:
        dq = 1/a
    vertices = np.array([[dq/2,dq/2],
                         [-dq/2,dq/2],
                         [-dq/2,-dq/2],
                         [dq/2,-dq/2]])
    vertices += np.outer(np.ones(4), r0)

    return Polygon(vertices, edgecolor=color, facecolor=(0,0,0,0), lw=lw)


def plot_BZ(ax, a=3, color='k', lw=1):
    ax.add_patch(square_BZ(a=a, color=color, lw=lw))


def plot_errors(a_vals, G_exact=None, R=5):
    if G_exact is None:
        l = np.arange(R)
        G_exact = np.column_stack((np.cos(2*np.pi*l/R), np.sin(2*np.pi*l/R)))
    err_vals = np.zeros(np.size(a_vals))
    for i,a in enumerate(a_vals):
        G = square_approximant(a=a, G_vects=G_exact)
        dG = G_exact - G
        err_vals[i] = np.sqrt(np.sum(dG**2, axis=None)/R)
    
    fig,ax = plt.subplots()
    ax.plot(a_vals, err_vals, color='b', marker='x', ls='-')
    ax.set_xlabel(r'$a$')
    ax.set_ylabel(r'$\sqrt{\langle\epsilon^{2}\rangle}$')
    ax.set_title('Root Mean Square Error vs Approximant Order')
    plt.show()


def plot_density(a=np.arange(1,10), N_bands=np.array([2,4,16,28,48,72,94,116,148,188]),
                 plot_midpoints=False, a_midpoints=None, R=5):
    n = N_bands / ((2*np.pi)**2 * a**2)
    N = N_bands / a**2

    # print(N)
    # print(n)
    # if R == 5:
    #     A = 5 * np.sin(2*np.pi/5) / (8 * np.cos(np.pi/5)**2)
    # elif R == 8:
    #     A = np.sqrt(2)
    A = R * np.sin(2*np.pi/R) / (8 * np.cos(np.pi/R)**2)

    fig,ax = plt.subplots()
    if plot_midpoints:
        if a_midpoints is None:
            a_midpoints = a
        midpoint_areas = np.zeros(a_midpoints.shape)
        for i, a_val in enumerate(a_midpoints):
            l = np.arange(R)
            G_vects_exact = np.column_stack((np.cos(2*np.pi*l/R),
                                            np.sin(2*np.pi*l/R)))
            midpoints = calc_all_midpoints(G_vects=square_approximant(a=a_val, G_vects=G_vects_exact))
            midpoints = np.vstack((midpoints, midpoints[0,:]))
            x = midpoints[:, 0]
            y = midpoints[:, 1]
            midpoint_areas[i] = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        ax.plot(a_midpoints, 2*midpoint_areas, color='limegreen', marker='o', ls='-', label='Approximant')
        print(midpoint_areas)
        print(2*midpoint_areas)
        print(2*midpoint_areas * a_midpoints**2)
    ax.plot(a, N, color='b', marker='x', ls='-', label='Numerical')
    ax.set_xlabel(r'$a$')
    ax.set_ylabel(r'$\frac{N_{bands}}{a^2}$', rotation=0, fontsize=15, labelpad=15)
    ax.set_title('Density vs ' + r'$a$')

    # ax.set_ylim(0,0.06)
    ax.axhline(y=2*A, color='r', ls='--', label='Quasicrystal')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # l = np.arange(8)
    # G_vects_exact = np.column_stack((np.cos(2*np.pi*l/8),
    #                                 np.sin(2*np.pi*l/8)))
    # N_bands = np.array([2,4,14,28,46,56,82,112,126,164])
    # plot_density(plot_midpoints=True, a_midpoints=np.arange(1,21), N_bands=N_bands, R=8, a=np.arange(1,11))
    # G1 = np.array([3,-1])
    # G2 = np.array([1,-3])
    # m = calc_midpoint(G1, G2)
    # print(m)
    # plot_errors(a_vals=np.arange(3,21), R=8)
    fig,ax = plt.subplots()
    a = 19
    l = np.arange(8)
    G_vects_exact = np.column_stack((np.cos(2*np.pi*l/8),
                                    np.sin(2*np.pi*l/8)))
    G_vects_approx = square_approximant(a=a, G_vects=G_vects_exact)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    add_square_grid(ax, a=a, L=10)
    plot_BZ(ax, a=a)
    plot_vectors(G_vects_exact, ax, color='k')
    plot_vectors(G_vects_approx, ax, color='r')
    plot_vectors(G_vects_approx, ax, color='c')
    lim = 1.2
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_title(f'a = {a}')
    # # # # # # add_hex_grid(ax, a=a, L=10)
    
    add_g_vectors(ax, G_vects=G_vects_approx)
    add_midpoints(ax, G_vects=G_vects_approx)
    # # # # # add_cutoff_radius(ax, c=3)
    plt.show()

    # print('G-vectors (units 1/a):')
    # G_vects_approx = a*square_approximant(a=a)
    # print(G_vects_approx)
    # g_vects = np.roll(G_vects_approx,-1,axis=0) - G_vects_approx
    # print('g-vectors (units 1/a):')
    # print(g_vects)