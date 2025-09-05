import numpy as np
import matplotlib.pyplot as plt


def plot_N_bands(O=np.arange(2,11),
                 N_gap=np.array([12, 12, 32, 62, 72, 102, 126, 162, 202]),
                 N_QBZ=np.array([12, 32, 32, 62, 82, 102, 122, 142, 202])):
    
    N_pred = 2*O**2
    fig,ax = plt.subplots()

    ax.plot(O, N_pred, color='b', marker='o', label='Predicted')
    ax.plot(O, N_gap, color='r', marker='x', label='Bands Below Gap')
    ax.plot(O, N_QBZ, color='limegreen', marker='^', label='Basis States in QBZ')

    ax.set_xlabel(r'$O$')
    ax.set_ylabel(r'$N_{bands}$')
    ax.set_title(r'$N_{bands}$ vs Basis Set Order')
    ax.legend()

    # ax.set_xlim(0,10)
    ax.set_ylim(bottom=0)

    plt.show()


def plot_C(O=np.arange(2,11),
           C_pred=np.array([-0.98688, -0.60965, -0.83323, -0.99598, -0.61835, -0.80174, np.nan, -1.02902, np.nan]),
           C_bands=np.array([-0.44695, -0.88709, -0.83323, -0.77108, -0.61835, -0.78622, np.nan, -1.02902, np.nan])):
    
    fig,ax = plt.subplots()
    ax.plot(O, C_pred, marker='o', color='b', label='Quadratic-Predicted QBZ')
    ax.plot(O, C_bands, marker='x', color='r', label='Band Number-Calculated QBZ')
    ax.axhline(y=-1, color='k', ls='--', label=r'$C=-1$')

    ax.set_xlabel(r'$O$')
    ax.set_ylabel(r'$C$', rotation=0)
    ax.set_title('Chern Number vs Basis Set Order')

    ax.legend()
    ax.set_ylim(top=0)

    plt.show()


if __name__ == '__main__':
    plot_N_bands()
    # plot_C()