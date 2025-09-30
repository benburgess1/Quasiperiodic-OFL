import numpy as np
import matplotlib.pyplot as plt
import Approximant_Bandstructure as AB


def plot_trends(filename, y='V'):
    data = np.load(filename)
    a_vals = data['a_vals']
    a_inv = 1/a_vals
    min_vals = data['min_vals']
    min_errs = data['min_errs']
    max_vals = data['max_vals']
    max_errs = data['max_errs']
    U = data['U']
    N = data['N']
    R = data['R']
    if y == 'V' and data['y'] == 'V/U':
        s = U
    elif y =='V/U' and data['y'] == 'V':
        s = 1/U
    fig,ax = plt.subplots()
    ax.errorbar(a_inv, s*min_vals, yerr=s*min_errs, color='b', marker='x', ls='-')
    ax.errorbar(a_inv, s*max_vals, yerr=s*max_errs, color='r', marker='x', ls='-')

    ax.set_xlabel(r'$a^{-1}$')
    ax.set_ylabel(r'$(\frac{V}{U})_{crit}$')
    ax.set_title(r'$(\frac{V}{U})_{crit}$' + ' vs ' + r'$a^{-1}$' + ', ' + 
                 '$N = $' + str(N) + ', ' + '$|U| = $' + str(U) + ', ' + '$R = $' + str(R))
    
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add a second x-axis manually and map a_vals to a_inv for ticks
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())  # match x limits

    # Set tick positions at a_inv, but label them with a_vals
    ax_top.set_xticks(a_inv)
    ax_top.set_xticklabels([str(int(a)) for a in a_vals])
    ax_top.set_xlabel(r'$a$')

    plt.tight_layout()
    
    plt.show()


def generate_E_gap_data(filename, a_vals, R=5):
    if R == 5:
        N_bands = np.array([2,4,16,28,48,72,94,116,148,188])
    elif R == 8:
        N_bands = np.array([2,4,14,28,46,56,82,112,126,164])
    E_gap_vals = []
    for a in a_vals:
        if R == 5:
            if a == 3:
                f = 'Approximant/Data/Data_a3_U0.2_N3_V0.15.npz'
            else:
                f = 'Approximant/Data/Data_NonAb_a' + str(a) + '_U0.2_N3_V0.15.npz'
        elif R == 8:
            f = 'Approximant/Data/8Fold/Data_R8_a' + str(a) + '_c2.5_U0.15_N5_V0.15.npz'
        data = np.load(f)
        E_vals = data['E_vals']
        idx = N_bands[a-1] - 1
        print(f'a = {a}, idx = {idx}')
        E_gap = np.min(E_vals[idx+1,:,:]) - np.max(E_vals[idx,:,:])
        E_gap_vals.append(E_gap)

    np.savez(filename, a_vals=a_vals, E_gap_vals=E_gap_vals, R=R)


def plot_E_gap(f, min_a=0):
    fig, ax = plt.subplots()

    data = np.load(f)
    a_vals = data['a_vals']
    mask = a_vals >= min_a
    a_inv = 1/a_vals
    E_gap = data['E_gap_vals']
    a_vals = a_vals[mask]
    E_gap = E_gap[mask]
    a_inv = a_inv[mask]

    # ax.plot(a_vals, E_gap, color='b', marker='x')
    ax.plot(a_inv, E_gap, color='b', marker='x')

    # ax.set_xlabel(r'$a$', fontsize=LARGE_FIG_FONTSIZE)
    ax.set_xlabel(r'$1/a$')
    ax.set_ylabel(r'$\Delta E/E_R$')
    # ax.tick_params(labelsize=LARGE_FIG_TICK_FONTSIZE)

    ax.set_xlim(left=0)
    ax.set_ylim(0.,0.075)
    ax.set_yticks(np.arange(0, 0.1, 0.025))

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())  # match x limits

    # Set tick positions at a_inv, but label them with a_vals
    ax_top.set_xticks(a_inv)
    ax_top.set_xticklabels([str(int(a)) for a in a_vals])
    ax_top.set_xlabel(r'$a$')
    # ax_top.tick_params(labelsize=LARGE_FIG_TICK_FONTSIZE)
    plt.show()




if __name__ == '__main__':
    f = 'Approximant/Data/8Fold/V_crit.npz'
    a_vals = np.array([3, 4, 5, 6, 7, 8, 9, 10])
    min_vals = np.array([0.03, 0.015, 0.0005, 0.03, 0.0005, 0.015, 0.03, 0.0001])
    min_errs = np.array([0.01, 0.005, 0.0005, 0.01, 0.0005, 0.005, 0.01, 0.0005])
    max_vals = np.array([0.21, 0.23, 0.35, 0.21, 0.23, 0.23, 0.21, 0.21])
    max_errs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    np.savez(f, a_vals=a_vals, min_vals=min_vals, min_errs=min_errs,
             max_vals=max_vals, max_errs=max_errs, U=0.15, y='V', N=5, R=8)

    
    plot_trends(f, y='V/U')

    # f = 'Approximant/Data/8Fold/E_gap_U0.15_N5_V0.15.npz'
    # generate_E_gap_data(f, a_vals=np.arange(1, 11, dtype=np.int32), R=8)
    # plot_E_gap(f, min_a=3)
