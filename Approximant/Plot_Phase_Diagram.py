import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import Approximant_Curvature as AC
import matplotlib.colors as mcolors
import glob
from scipy.spatial import cKDTree
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from matplotlib.colors import ListedColormap, BoundaryNorm
import re 


bwr = plt.cm.get_cmap('bwr')
upper_half_bwr = mcolors.LinearSegmentedColormap.from_list(
    'upper_half_bwr', bwr(np.linspace(0.5, 1, 256)))


def build_file_path(U, V, N, a=3, NonAb=False):
    (U, V) = np.round((U, V), 4)
    N = int(N)
    a = int(a)
    file_str = 'Data/Data_'
    if NonAb:
        file_str += 'NonAb_'
    file_str += 'a' +  str(a) + '_U' + str(U) + '_N' + str(N) + '_V' + str(V) + '.npz'
    return file_str


def locate_files(U_vals=None, V_vals=None, N=3, a=4, NonAb=True, R=None):
    if U_vals is not None and V_vals is not None:
        files = []
        for i, U in enumerate(U_vals):
            for j, V in enumerate(V_vals):
                file_path = build_file_path(U, V, N, a=a, NonAb=NonAb)
                if os.path.exists(file_path):
                    files.append(file_path)
    else:
        file_path = 'Approximant/Data/8Fold/Data_'
        if NonAb == True:
            file_path += 'NonAb_'
        if R is not None:
            file_path += 'R' + str(R)
        file_path += '*a' + str(a) + '*N' + str(N) + '*.npz'
        files = glob.glob(file_path)
    return files


def generate_legend(indetermined=True, C2=True, gap_closed=True, E_gap=0.01):
    legend_entries = [
        mlines.Line2D([], [], color='grey', marker='x', linestyle='None', label=r'$C = 0$'),
        mlines.Line2D([], [], color='r', marker='x', linestyle='None', label=r'$C = +1$'),
        mlines.Line2D([], [], color='b', marker='x', linestyle='None', label=r'$C = -1$')]
    if C2:
        legend_entries += [
            mlines.Line2D([], [], color='darkred', marker='x', linestyle='None', label=r'$C = +2$'),
            mlines.Line2D([], [], color='navy', marker='x', linestyle='None', label=r'$C = -2$')]
    if indetermined:
        legend_entries += [
            mlines.Line2D([], [], color='r', marker='D', linestyle='None', label=r'$C > 0$'),
            mlines.Line2D([], [], color='b', marker='D', linestyle='None', label=r'$C < 0$')]
    if gap_closed:
        legend_entries += [
            mlines.Line2D([], [], color='grey', marker='s', linestyle='None', label=r'$\Delta E < $' + str(E_gap))]
    return legend_entries


def plot_C(U_vals=None, V_vals=None, N=3, a=3, max_idx=15, x_axis='V', E_gap=0.01,
           calc_C=False, NonAb=False):
    fig,ax = plt.subplots()
    indetermined = False
    files = locate_files(U_vals=U_vals, V_vals=V_vals, N=N, a=a, NonAb=NonAb)
    for file_path in files:
        data = np.load(file_path)
        U = data['U0']
        V = data['V0']
        if calc_C:
            C = AC.calc_chern_number(curv_vals=data['curv_vals'], 
                                        inside_BZ=False, bands=np.arange(max_idx+1))
        else:
            C = data['C']
        if x_axis == 'V/U':
            if U != 0:
                V /= U
            else:
                continue
        if check_gap_open(file_path, idx=max_idx, E_gap=E_gap):
        # if True:
            if np.round(C, 1) == 0.0:
                ax.plot(V, U, marker='x', color='grey')
            elif np.round(C, 1) == 1.0:
                ax.plot(V, U, marker='x', color='r')
            elif np.round(C, 1) == -1.0:
                ax.plot(V, U, marker='x', color='b')
            elif C > 0:
                indetermined = True
                ax.plot(V, U, marker='D', color='r')
            elif C < 0:
                indetermined = True
                ax.plot(V, U, marker='D', color='b')
        else:
            ax.plot(V, U, marker='s', color='grey')
    if x_axis == 'V/U':
        ax.set_xlabel(r'$V$ / $U$')
    else:
        ax.set_xlabel(r'$V$ / $E_R$')
    ax.set_ylabel(r'$U$ / $E_R$')

    legend_entries = generate_legend(indetermined=indetermined, E_gap=E_gap)
    ax.legend(handles=legend_entries)

    ax.set_title(f'Chern Number, a = {a}, Bands = [0-{max_idx}]')

    plt.show()


def plot_C_fromfile(filename, a=4, max_idx=27, y_axis='V', N=3, E_gap=0.001, scalefactor=1, legend_loc='upper left'):
    fig,ax = plt.subplots()
    indetermined = False
    C2 = False
    data = np.load(filename)
    U_vals = data['U_vals']
    V_vals = data['V_vals']
    C_vals = data['C_vals'] * scalefactor
    if 'a' in data:
        a = data['a']
    if 'max_idx' in data:
        max_idx = data['max_idx']
    if 'E_gap' in data:
        E_gap = data['E_gap']
    if 'N' in data:
        N = data['N']
    
    for i in range(U_vals.size):
        U = U_vals[i]
        V = V_vals[i]
        C = C_vals[i]
        if y_axis == 'V/U':
            if U != 0:
                V /= U
            else:
                continue
        if np.isnan(C):
            ax.plot(U, V, marker='s', color='grey')
        elif np.round(C, 1) == 0.0:
                ax.plot(U, V, marker='x', color='grey')
        elif np.round(C, 1) == 1.0:
            ax.plot(U, V, marker='x', color='r')
        elif np.round(C, 1) == -1.0:
            ax.plot(U, V, marker='x', color='b')
        elif np.round(C, 1) == 2.0:
            C2 = True
            ax.plot(U, V, marker='x', color='darkred')
        elif np.round(C, 1) == -2.0:
            C2 = True
            ax.plot(U, V, marker='x', color='navy')
        elif C > 0:
            indetermined = True
            ax.plot(U, V, marker='D', color='r')
        elif C < 0:
            indetermined = True
            ax.plot(U, V, marker='D', color='b')
    
    if y_axis == 'V/U':
        ax.set_ylabel(r'$V$ / $U$')
    else:
        ax.set_ylabel(r'$V$ / $E_R$')
    ax.set_xlabel(r'$U$ / $E_R$')

    legend_entries = generate_legend(indetermined=indetermined, C2=C2, E_gap=E_gap)
    ax.legend(handles=legend_entries, loc=legend_loc)

    title_str = 'Chern Number, ' + r'$a = $' + str(a) + ', ' + r'$N = $' + str(N) + ', ' + f'Bands = [0-{max_idx}]'
    ax.set_title(title_str)

    plt.show()


def plot_C_grid_fromfile(filename, Cfactor=-1, Vfactor=0.5, print_outliers=True):
    data = np.load(filename)
    U = data['U_vals']
    V = data['V_vals'] * Vfactor
    C = data['C_vals'] * Cfactor
    C_rounded = np.round(C, 1)
    U_grid, V_grid = np.meshgrid(U, V, indexing='ij')
    colors = ['grey', 'navy', 'blue', 'white', 'red', 'darkred', 'yellow']
    cmap = ListedColormap(colors)

    # Map values to indices
    C_index = np.full(C.shape, 6)  # default to yellow (outlier)
    mapping = {-2: 1, -1: 2, 0: 3, 1: 4, 2: 5}

    for k, v in mapping.items():
        C_index[C_rounded == k] = v
    C_index[np.isnan(C_rounded)] = 0  # NaN -> grey

    if print_outliers:
        idxs = np.argwhere(C_index==6)
        # print(idxs.shape)
        # print(idxs)
        for idx in idxs:
            print(f'U = {U[idx[0]]}, V = {V[idx[1]]*Vfactor}: C = {C[idx[0]][idx[1]]*Cfactor}')

    # Norm ensures correct color boundaries
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]  # One more than number of colors
    norm = BoundaryNorm(bounds, cmap.N)

    # Plot
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(U_grid, V_grid, C_index, cmap=cmap, norm=norm, shading='nearest')
    # ax.set_aspect('equal')

    ax.set_xlabel(r'$U/E_R$')
    ax.set_ylabel(r'$V/E_R$')
    ax.set_xlim(np.min(U), np.max(U))
    ax.set_ylim(np.min(V), np.max(V))
    # Colorbar
    cbar = plt.colorbar(mesh, ticks=[0, 1, 2, 3, 4, 5, 6], ax=ax)
    cbar.ax.set_yticklabels(['Nan','-2', '-1', '0', '1', '2','Outlier'])
    cbar.set_label('C value')

    plt.show()


def check_gap_open(filename, idx=15, E_gap=0.01):
    data = np.load(filename)
    E_vals = data['E_vals']
    E_max = np.max(E_vals[idx,:,:])
    E_min = np.min(E_vals[idx+1,:,:])
    # print(E_max)
    # print(E_max + E_gap)

    # E_1 = E_vals[idx,:,:]
    # E_2 = E_vals[idx+1,:,:]

    # print(np.sort(E_1, axis=None))
    # print(np.sort(E_2, axis=None))

    # print(not np.any(E_vals[idx+1,:,:]<(E_max+E_gap)))
    
    return E_min > E_max + E_gap
    # return not np.any(E_vals[idx+1,:,:]<(E_max+E_gap))
    # dos_vals = data['dos_vals']
    # E_bins = data['E_bins']
    # dE = data['dE']
    # E_max = np.max(E_vals[idx,:,:])
    # # print(E_max)
    # # print(E_bins[np.logical_and(E_bins > E_max, E_bins < (E_max + dE))])
    # mask = (E_bins + dE/2) > E_max
    # # E_0 = np.min(E_bins[mask])
    # # print(mask)
    # # print(E_0)
    # return np.isclose(dos_vals[np.argmax(mask)+1], 0., atol=1e-3)


def generate_E_gap_data(filename, U_vals, V_vals, N=3, a=3, max_idx=15, NonAb=False,
                        R=None):
    files = locate_files(U_vals=U_vals, V_vals=V_vals, a=a, N=N, NonAb=NonAb, R=R)
    U_found = []
    V_found = []
    E_gap_vals = []
    for i,file_path in enumerate(files):
        print(f'Analysing file {i+1} out of {len(files)}...' + 10*' ', end='\r')
        data = np.load(file_path)
        U_found.append(data['U0'])
        V_found.append(data['V0'])
        E_vals = data['E_vals']
        E_gap = np.min(E_vals[max_idx+1,:,:]) - np.max(E_vals[max_idx,:,:])
        # print(f'U = {U}, V = {V}, E_gap = {E_gap}')
        if E_gap < 0:
            E_gap = 0
        E_gap_vals.append(E_gap)
    np.savez(filename, U_vals=np.array(U_found), V_vals=np.array(V_found), 
             E_gap_vals=np.array(E_gap_vals), 
             N=N, a=a, max_idx=max_idx)
    

def generate_C_data(filename, U_vals=None, V_vals=None, N=3, a=4, max_idx=27, NonAb=False, 
                    E_gap=0.001, R=None):
    print('Locating files...')
    files = locate_files(U_vals=U_vals, V_vals=V_vals, a=a, N=N, NonAb=NonAb, R=R)
    print(f'Done: {len(files)} files found')
    U_found = []
    V_found = []
    C_vals = []
    for i, file_path in enumerate(files):
        print(f'Analysing file {i+1} out of {len(files)}...' + 10*' ', end='\r')
        data = np.load(file_path)
        if 'max_idx' not in data.files:
            # print('max_idx not in file')
            continue
        elif data['max_idx'] != max_idx:
            # print(f'U = {U_found[-1]}, V = {V_found[-1]}: incorrect max_idx')
            continue
        U_found.append(data['U0'])
        V_found.append(data['V0'])
        C = data['C']
        if check_gap_open(file_path, idx=max_idx, E_gap=E_gap):
            C_vals.append(C)
        else:
            C_vals.append(np.nan)
    print('\nDone')
    np.savez(filename, U_vals=np.array(U_found), V_vals=np.array(V_found), 
             C_vals=np.array(C_vals), 
             N=N, a=a, max_idx=max_idx, E_gap=E_gap)
    

def locate_files_strict(U_vals=None, V_vals=None, N=5, a=3, R=8,
                        file_stem='', c=2.5):
    files = []
    for i, U in enumerate(U_vals):
        for j, V in enumerate(V_vals):
            U = np.round(U,4)
            V = np.round(V,4)
            file_path = file_stem
            if R is not None:
                file_path += '_R' + str(R)
            if a is not None:
                file_path += '_a' + str(a)
            if c is not None:
                file_path += '_c' + str(c)
            file_path += '_U' + str(U) + '_N' + str(N) + '_V' + str(V) + '.npz'
            if os.path.exists(file_path):
                files.append(file_path)
    return files
    

def generate_C_data_clean(filename, U_vals=None, V_vals=None, N=3, a=4, max_idx=27, c=2.5, 
                    E_gap=0.001, R=None, file_stem='Approximant/Data/8Fold/Old_Data/Data',
                    exceptions={}):
    C_vals = np.zeros((U_vals.size, V_vals.size))
    for i, U in enumerate(U_vals):
        for j, V in enumerate(V_vals):
            print(f'Searching for data point {i*U_vals.size + j + 1} out of {C_vals.size}...' + 10*' ', end='\r')
            U = np.round(U,4)
            V = np.round(V,4)
            if (U,V) in exceptions.keys():
                C_vals[i,j] = exceptions[(U,V)]
                continue
            file_path = file_stem
            if R is not None:
                file_path += '_R' + str(R)
            if a is not None:
                file_path += '_a' + str(a)
            if c is not None:
                file_path += '_c' + str(c)
            file_path += '_U' + str(U) + '_N' + str(N) + '_V' + str(V) + '.npz'
            if os.path.exists(file_path):
                data = np.load(file_path)
                if 'max_idx' not in data.files:
                    continue
                elif data['max_idx'] != max_idx:
                    continue
                C = np.real(data['C'])
                if check_gap_open(file_path, idx=max_idx, E_gap=E_gap):
                    C_vals[i,j] = C
                else:
                    C_vals[i,j] = np.nan
    print('\nDone')
    np.savez(filename, U_vals=U_vals, V_vals=V_vals, R=R, c=c,
             C_vals=np.array(C_vals), 
             N=N, a=a, max_idx=max_idx, E_gap=E_gap)


def generate_C_data_clean_5fold(filename, U_vals=None, V_vals=None, N=3, max_idx=27,  
                    E_gap=0.001, file_stem='Approximant/Data/8Fold/Old_Data/Data',
                    exceptions={}):
    U_vals, V_vals = np.round(U_vals,4), np.round(V_vals,4)
    C_vals = np.zeros((U_vals.size, V_vals.size))

    targets = np.array([(u, v) for u in U_vals for v in V_vals])

    # Prepare dictionary to hold candidates
    candidates = {tuple(t): [] for t in targets}

    # File parsing
    pattern = re.compile(r"U(-?\d+(?:\.\d+)?).*V(-?\d+(?:\.\d+)?)")
    print('Locating files...')
    files = glob.glob(file_stem + '*N' + str(N) + '*.npz')
    print(f'Done: {len(files)} files found')
    # files = glob.glob("Data*U*V*.npz")

    def closest_target(Uf, Vf, targets):
        # Compute squared distance to all targets
        diffs = targets - np.array([Uf, Vf])
        dist2 = np.sum(diffs**2, axis=1)
        return tuple(targets[np.argmin(dist2)]), np.sqrt(np.min(dist2))

    for i,f in enumerate(files):
        print(f'Assigning files to (U,V) values: file {i+1}/{len(files)}' + 10 *' ', end='\r')
        m = pattern.search(os.path.basename(f))
        if not m:
            print(f"Warning: Could not parse U,V from {f}")
            continue
        Uf, Vf = map(float, m.groups())
        if Uf > np.max(U_vals) or Vf > np.max(V_vals):
            continue
        else:
            t, d = closest_target(Uf, Vf, targets)
            candidates[t].append((d, f))
    print('\nDone')

    # Pick closest file for each target
    closest_files = {}
    for t, lst in candidates.items():
        print(f'Selecting best files: point (U,V) = {t}' + 10 *' ', end='\r')
        if lst:
            d, f = min(lst, key=lambda x: x[0])
            closest_files[t] = f
            if d > 0.01:
                print(f"\nWarning: closest file for {t} is {f} (distance={d:.4f})")
        else:
            print(f"\nWarning: No file assigned to {t}")
    print('\nDone')

    # print(closest_files)

    for t, f in closest_files.items():
        # print(t)
        print(f'Analysing best files: point (U,V) = {t}' + 10*' ', end='\r')
        # print(t)
        # print(type(t))
        U, V = t
        U, V = np.round(U,4), np.round(V,4)
        if (U,V) in exceptions.keys():
            C_vals[i,j] = exceptions[(U,V)]
            continue
        i = np.argwhere(U_vals == U)[0][0]
        j = np.argwhere(V_vals == V)[0][0]
        # print(i, j)
        data = np.load(f)
        if 'max_idx' not in data.files:
            print(f'\nWarning: no max_idx in {t} data files. Proceeding regardless.')
        elif data['max_idx'] != max_idx:
            print(f'\nWarning: stored max_idx for {t} is incorrect. Skipping data point.')
            continue
        C = np.real(data['C'])
        if check_gap_open(f, idx=max_idx, E_gap=E_gap):
            C_vals[i,j] = C
        else:
            C_vals[i,j] = np.nan
    print('\nDone')

    np.savez(filename, U_vals=U_vals, V_vals=V_vals, 
             C_vals=np.array(C_vals), 
             N=N, max_idx=max_idx, E_gap=E_gap)
    

def plot_E_gap(filename, cmap=plt.cm.get_cmap('hot'), plot_log=False, y_axis='V',
               plot_title=True, plot_cbar=True, vline=None, hline=None):
    data = np.load(filename)
    E_gap_vals = data['E_gap_vals']
    U_vals = data['U_vals']
    V_vals = data['V_vals']

    fig,ax = plt.subplots()

    zlab = r'$\Delta E$ / $E_R$'

    if plot_log:
        mask = E_gap_vals > 0
        E_gap_vals = np.log10(E_gap_vals[mask])
        U_vals = U_vals[mask]
        V_vals = V_vals[mask]
        zlab = r'$log_{10}($' + zlab + r'$)$'
        vmin = np.min(E_gap_vals)
        vmax = np.max(E_gap_vals)
        levels = np.linspace(vmin,vmax,200)
        ticks = [vmin, vmax]
        ticklabels = [np.round(vmin,3), np.round(vmax,3)]
    else:
        vmax = np.max(E_gap_vals)
        levels = np.linspace(0,vmax,200)
        ticks = [0, vmax]
        ticklabels = [r'$\leq 0.0$', str(np.round(vmax,3))]
    #print(levels)
    if y_axis == 'V/U':
        mask = U_vals > 0
        U_vals = U_vals[mask]
        V_vals = V_vals[mask]
        V_vals /= U_vals
        E_gap_vals = E_gap_vals[mask]
        ylab = r'$V$ / $U$'
    else:
        ylab = r'$V$ / $E_R$'
        if y_axis != 'V':
            print('Warning: unkown x-axis specification, defaulting to V')

    plot = ax.tricontourf(U_vals, V_vals, E_gap_vals, cmap=cmap, levels=levels)
    ax.set_ylabel(ylab)
    ax.set_xlabel(r'$U$ / $E_R$')
    if plot_title:
        a = data['a']
        N = data['N']
        title_str = 'Energy Gap, ' + r'$a = $' + str(a) + ', ' + r'$N = $' + str(N)
        ax.set_title(title_str)
    if plot_cbar:
        cbar = fig.colorbar(plot, ticks=ticks)
        cbar.ax.set_ylabel(zlab, rotation=0)
        cbar.ax.set_yticklabels(ticklabels)

    if vline is not None:
        ax.axvline(x=vline, color='limegreen', ls='--')
    if hline is not None:
        ax.axhline(y=hline, color='limegreen', ls='--')

    plt.show()


def find_unstructured_boundary(U, V, C, neighbor_radius=1e-3, k=None):
    points = np.column_stack((U, V))
    tree = cKDTree(points)

    boundary_inside = []
    boundary_outside = []

    for idx, (u, v, c_val) in enumerate(zip(U, V, C)):
        if c_val != -1:
            continue  # only consider -1 region points

        # Find neighbors either by fixed radius or fixed count
        if k is not None:
            dists, neighbors = tree.query([u, v], k=k+1)
            neighbors = neighbors[1:]  # exclude self
        else:
            neighbors = tree.query_ball_point([u, v], r=neighbor_radius)
            neighbors = [n for n in neighbors if n != idx]  # exclude self

        for n in neighbors:
            neighbor_c = C[n]
            if neighbor_c == 0 or np.isnan(neighbor_c):
                boundary_inside.append((u, v))
                boundary_outside.append((U[n], V[n]))
                break  # one neighbor is enough to count this point as on the boundary

    return np.array(boundary_inside), np.array(boundary_outside)


def find_boundary(filename, val=-1, tol=0.1, save=True, 
                  save_filename='Boundary.npz', ratio=True):
    data = np.load(filename)
    U_vals = np.round(data['U_vals'],5)
    V_vals = np.round(data['V_vals'],5)
    C_vals = data['C_vals']
    if ratio:
        mask = U_vals > 0
        U_vals, V_vals, C_vals = U_vals[mask], V_vals[mask], C_vals[mask]
        V_vals /= U_vals
        V_vals = np.round(V_vals,5)
    
    idxs = np.lexsort((V_vals,U_vals))
    U_vals, V_vals, C_vals = U_vals[idxs], V_vals[idxs], C_vals[idxs]
    # U_B_l_i = []
    # V_B_l_i = []
    # U_B_l_o = []
    # V_B_l_o = []
    # U_B_r_i = []
    # V_B_r_i = []
    # U_B_r_o = []
    # V_B_r_o = []
    # U_B_lo_i = []
    # V_B_lo_i = []
    # U_B_lo_o = []
    # V_B_lo_o = []
    # U_B_u_i = []
    # V_B_u_i = []
    # U_B_u_o = []
    # V_B_u_o = []

    B_i = np.array([[0, 0]])
    B_o = np.array([[0, 0]])

    # Go up left boundary first
    print('Left')
    U_max = 0
    for U in np.unique(U_vals):
        print(f'U = {U}')
        V_slice = V_vals[U_vals==U]
        C_slice = C_vals[U_vals==U]
        print(V_slice)
        print(C_slice)
        idxs = np.argwhere(np.isclose(C_slice, val, atol=tol))
        print(idxs)
        if idxs.size > 0:
            # Find left boundary
            idx = idxs[0][0]
            if idx > 0:
                B_i = np.row_stack((B_i, np.array([V_slice[idx], U])))
                B_o = np.row_stack((B_o, np.array([V_slice[idx-1], U])))
                # U_B_l_i.append(U)
                # V_B_l_i.append(V_slice[idx])
                print(f'Inside: {V_slice[idx]}')
                # U_B_l_o.append(U)
                print(f'Outside: {V_slice[idx-1]}')
                # V_B_l_o.append(V_slice[idx-1])
                U_max = U
            # else:
            #     V_B_l_o.append(V_slice[idx])
                # print('failed')

    # Go across top
    print('\nTop')
    V_slice = V_vals[U_vals==U_max]
    C_slice = C_vals[U_vals==U_max]
    print(f'U = {U_max}')
    print(V_slice)
    print(C_slice)
    idxs = np.argwhere(np.isclose(C_slice, val, atol=tol))
    print(idxs)
    if idxs.size >= 3:
        for i in range(1, idxs.size-1):
            B_i = np.row_stack((B_i, np.array([V_slice[idxs[i][0]], U_max])))
    for i in range(idxs.size):
        V_idx = idxs[i][0]
        print(V_idx)
        print(V_slice[V_idx])
        U_slice = U_vals[V_vals==V_slice[V_idx]]
        print(U_slice)
        U_idx = np.argwhere(U_slice==U_max)[0][0] + 1
        print(U_idx)
        print(U_slice[U_idx])
        B_o = np.row_stack((B_o, np.array([V_slice[V_idx], U_slice[U_idx]])))

    # Go down the right
    print('\nRight')
    U_min = 0
    for U in np.unique(U_vals)[::-1]:
        print(f'U = {U}')
        V_slice = V_vals[U_vals==U]
        C_slice = C_vals[U_vals==U]
        print(V_slice)
        print(C_slice)
        idxs = np.argwhere(np.isclose(C_slice, val, atol=tol))
        print(idxs)
        if idxs.size > 0:
            idx = idxs[-1][0]
            # print('Right:')
            if idx < len(V_slice) - 1:
                B_i = np.row_stack((B_i, np.array([V_slice[idx], U])))
                B_o = np.row_stack((B_o, np.array([V_slice[idx+1], U])))
                # U_B_r_i.append(U)
                # V_B_r_i.append(V_slice[idx])
                print(f'Inside: {V_slice[idx]}')
                # U_B_r_o.append(U)
                # V_B_r_o.append(V_slice[idx+1])
                print(f'Outside: {V_slice[idx+1]}')
                U_min = U
            # else:
            #     V_B_r_o.append(V_slice[idx])
    
    # Go across bottom
    print('\nBottom')
    print(f'U = {U_min}')
    V_slice = V_vals[U_vals==U_min]
    C_slice = C_vals[U_vals==U_min]
    print(V_slice)
    print(C_slice)
    idxs = np.argwhere(np.isclose(C_slice, val, atol=tol))
    print(idxs)
    if idxs.size >= 3:
        for i in range(1, idxs.size-1):
            B_i = np.row_stack((B_i, np.array([V_slice[idxs[i][0]], U_min])))
    for i in range(idxs.size)[::-1]:
        V_idx = idxs[i][0]
        print(V_idx)
        print(V_slice[V_idx])
        U_slice = U_vals[V_vals==V_slice[V_idx]]
        print(U_slice)
        U_idx = np.argwhere(U_slice==U_min)[0][0] - 1
        print(U_idx)
        print(U_slice[U_idx])
        B_o = np.row_stack((B_o, np.array([V_slice[V_idx], U_slice[U_idx]])))

    B_0 = np.array([[0.,0.]])

    # Find E=0 Boundary
    
    for V in np.unique(V_vals):
        print(f'V = {V}')
        U_slice = U_vals[V_vals==V]
        C_slice = C_vals[V_vals==V]
        print(U_slice)
        print(C_slice)
        idxs = np.argwhere(np.isnan(C_slice))
        print(idxs)
        if idxs.size > 0:
            idxs = idxs.reshape(idxs.size)
            idxs = idxs[idxs < U_slice.size/3]
            print(idxs)
            if idxs.size > 0:
                idx = idxs[-1]
                print(idx)
                print(U_slice[idx])
                B_0 = np.row_stack((B_0, np.array([V, U_slice[idx]])))
    
    V_max = np.max(V_vals)
    B_0 = np.row_stack((B_0, np.array([V_max, 0.]), np.array([0., 0.,]),
                        np.array([0., np.max(B_0[:,1])])))
    
    



    # for V in np.unique(V_vals):
    #     print(f'V = {V}')
    #     U_slice = U_vals[V_vals==V]
    #     C_slice = C_vals[V_vals==V]
    #     print(U_slice)
    #     print(C_slice)
    #     idxs = np.argwhere(np.isclose(C_slice, val, atol=tol))
    #     print(idxs)
    #     if idxs.size > 0:
    #         # Find lower boundary
    #         idx = idxs[0][0]
    #         if idx > 0:
    #             U_B_lo_i.append(U_slice[idx])
    #             print(f'Inside: {U_slice[idx]}')
    #             V_B_lo_i.append(V)
    #             U_B_lo_o.append(U_slice[idx-1])
    #             print(f'Outside: {U_slice[idx-1]}')
    #         # else:
    #         #     U_B_lo_o.append(U_slice[idx])
    #             V_B_lo_o.append(V)
    #         # Find upper boundary
    #         idx = idxs[-1][0]
    #         if idx < len(U_slice) - 1:
    #             U_B_u_i.append(U_slice[idx])
    #             print(f'Inside: {U_slice[idx]}')
    #             V_B_u_i.append(V)
    #             U_B_u_o.append(U_slice[idx+1])
    #             print(f'Inside: {U_slice[idx+1]}')
    #         # else:
    #         #     U_B_u_o.append(U_slice[idx])
    #             V_B_u_o.append(V)


    # print('Left Boundary:')
    # print(np.column_stack((np.array(V_B_l_i),np.array(U_B_l_i))))
    # print('Right Boundary:')
    # print(np.column_stack((np.array(V_B_r_i),np.array(U_B_r_i))))
    # print('Upper Boundary:')
    # print(np.column_stack((np.array(V_B_u_i),np.array(U_B_u_i))))
    # print('Lower Boundary:')
    # print(np.column_stack((np.array(V_B_lo_i),np.array(U_B_lo_i))))

    

    # U_B_i = np.concatenate((np.array(U_B_lo_i), np.array(U_B_r_i),
    #                         np.array(U_B_u_i), np.array(U_B_l_i)))
    # U_B_o = np.concatenate((np.array(U_B_lo_o), np.array(U_B_r_o),
    #                         np.array(U_B_u_o), np.array(U_B_l_o)))
    # V_B_i = np.concatenate((np.array(V_B_lo_i), np.array(V_B_r_i),
    #                         np.array(V_B_u_i), np.array(V_B_l_i)))
    # V_B_o = np.concatenate((np.array(V_B_lo_o), np.array(V_B_r_o),
    #                         np.array(V_B_u_o), np.array(V_B_l_o)))
    
    # B_i = np.unique(np.column_stack((V_B_i,U_B_i)), axis=0)
    # B_o = np.unique(np.column_stack((V_B_o,U_B_o)), axis=0)

    B_i = B_i[1:,:]
    B_o = B_o[1:,:]
    B_0 = B_0[1:,:]
    print('Unique Inside:')
    print(B_i)
    print('Unique Outside:')
    print(B_o)
    print('E=0 Boundary:')
    print(B_0)

    # hull_i = ConvexHull(B_i)
    # B_i = B_i[hull_i.vertices]
    # hull_o = ConvexHull(B_o)
    # B_o = B_o[hull_o.vertices]

    if save:
        np.savez(save_filename, B_i=B_i, B_o=B_o, N=data['N'], a=data['a'], 
                 max_idx=data['max_idx'], ratio=ratio, B_0=B_0) #U_B_i=U_B_i, U_B_o=U_B_o, V_B_i=V_B_i, V_B_o=V_B_o)
    # return U_B_i, U_B_o, V_B_i, V_B_o


def plot_boundary(filename, x_axis='V', xlim=None, ylim=None, plot_title=True,
                  plot_E0=False):
    data = np.load(filename)
    # U_B_i = data['U_B_i']
    # V_B_i = data['V_B_i']
    # U_B_o = data['U_B_o']
    # V_B_o = data['V_B_o']
    # print(np.column_stack((V_B_i, U_B_i)))
    B_i = data['B_i']
    B_o = data['B_o']
    B_0 = data['B_0']
    ratio = data['ratio']
    if x_axis == 'V/U' and not ratio:
        B_i[:,0] /= B_i[:,1]
        B_o[:,0] /= B_o[:,1]
        B_0[:,0] /= B_0[:,1]
    elif x_axis == 'V' and ratio:
        B_i[:,0] *= B_i[:,1]
        B_o[:,0] *= B_o[:,1]
        B_0[:,0] *= B_0[:,1]
    fig,ax = plt.subplots()
    B_i = Polygon(B_i, facecolor='b')
    B_o = Polygon(B_o, facecolor='lightskyblue', edgecolor='b')
    B_0 = Polygon(B_0, facecolor='grey')
    ax.add_patch(B_o)
    ax.add_patch(B_i)
    if plot_E0:
        ax.add_patch(B_0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if plot_title:
        N = data['N']
        a = data['a']
        max_idx = data['max_idx']
        title_str = 'Phase Diagram, ' + r'$a = $' + str(a) + ', ' + r'$N = $' + str(N) + f', Bands = [0-{max_idx}]'
        ax.set_title(title_str)
    ax.set_ylabel(r'$U$ / $E_R$')
    if x_axis == 'V':
        ax.set_xlabel(r'$V$ / $E_R$')
    elif x_axis == 'V/U':
        ax.set_xlabel(r'$V$ / $U$')
    #To Do: legend, separate regions for ungapped vs gapped and trivial
    plt.show()




if __name__ == '__main__':
    # f = 'Data/Data_a1_U0.2_N3_V0.2.npz'
    # print(check_gap_open(f, idx=))
    # U_vals = np.arange(0, 0.6, 0.1)
    # V_vals = np.copy(U_vals)

    # U_vals = np.concatenate((U_vals, np.linspace(0.2,0.4,11)[1:-1]))
    # V_vals = np.concatenate((V_vals, np.linspace(0,0.3,11)[1:-1]))

    ### a = 3 values ###
    # U_vals = np.concatenate((np.arange(0,0.6,0.1), 
    #                          np.array([0.05,0.125,0.15,0.175]),
    #                          np.array([0.25,0.35,0.45])))
    # V_vals = np.concatenate((np.arange(0,0.6,0.1),
    #                          np.arange(0.05,0.11,0.005),
    #                          np.array([0.12,0.15]),
    #                          np.array([0.25, 0.35])))

    # U_vals = np.concatenate((np.array([0.2]),np.arange(0,0.51,0.05)))
    # V_vals = np.concatenate((np.arange(0.0, 0.35, 0.05),0.7*np.arange(0,0.51,0.05)))
    # U_vals = [0.0, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2]
    # # # print(np.arange(0.05, 0.11, 0.005))
    # V_vals = np.concatenate((np.array([0.]), np.arange(0.05, 0.11, 0.005), 
    #                          np.array([0.12, 0.15, 0.2])))
    # print(U_vals)
    # print(V_vals)
    # N_vals = [3]

    # U_vals = np.arange(0.0, 0.61, 0.05)
    # # U_vals = np.array([0.2])
    # r_vals = np.arange(0.4,1.81,0.1)
    # V_vals = np.array([])
    # for r in r_vals:
    #     V_vals=np.concatenate((V_vals, r*U_vals))

    # print(U_vals)
    # print(V_vals)

    # generate_E_gap_data('Approximant/Data/8Fold/E_gap_R8_a3_N5.npz', U_vals=None, V_vals=None, N=5, a=3, max_idx=13,
    #                     NonAb=False, R=8)

    # f = 'Approximant/Data/8Fold/E_gap_R8_a3_N5.npz'
    # plot_E_gap(f, plot_log=False, cmap=plt.cm.get_cmap('hot'), plot_title=True,
    #            y_axis='V', vline=None, hline=None)
    U_vals = np.arange(0.,0.41,0.01)
    V_vals = np.arange(0.,0.41,0.01)

    # f = 'Approximant/Data/5Fold/Data_NonAb_a4_U0.2_N3_V0.14.npz'
    # data = np.load(f)
    # print(data['C'])
    # f = 'Approximant/Data/8Fold/Old_Data/Chern_R8_a3_N5_strict_dE0.npz'
    f = 'Approximant/Data/5Fold/Chern_R5_a4_N3_strict_dE0.npz'
    # generate_C_data_clean_5fold(f, U_vals=U_vals, V_vals=V_vals, N=3, max_idx=27,
    #                             E_gap=0, file_stem = 'Approximant/Data/5Fold/Data_NonAb_a4_')
    # # generate_C_data_clean(f, 
    # #                       N=5, a=3, max_idx=13, U_vals=U_vals, V_vals=V_vals, R=8, 
    # #                       E_gap=0., c=2.5)
    # f = 'Approximant/Data/5Fold/Chern_NonAb_a4_N3_updated.npz'
    # plot_C_grid_fromfile(f, Cfactor=-1, Vfactor=1.)
    f = 'Approximant/Data/5Fold/Data_NonAb_a4_U0.16_N3_V0.33.npz'
    data = np.load(f)
    U = data['U0']
    V = data['V0']
    C = data['C']
    E = data['E_vals']
    E_gap = np.min(E[28,:,:]) - np.max(E[27,:,:])
    print(f'U = {U}, V = {V}, C = {C} (no scaling), E_gap = {E_gap}')

    # data = np.load(f)
    # print(data.files)
    # U = data['U_vals']
    # V = data['V_vals']
    # C = data['C_vals']
    # print(U.shape, V.shape, C.shape)
    # U_rounded = np.round(U,2)
    # V_rounded = np.round(V,2)
    # P = np.column_stack((U_rounded, V_rounded))
    # P = np.unique(P, axis=0)
    # U_grid, V_grid = np.meshgrid(U_vals, V_vals, indexing='ij')
    # P1 = np.column_stack((U_grid.flatten(), V_grid.flatten()))
    # P1 = P1[np.any(P1 > 0.3, axis=1)]

    # mask = np.any(np.all(P1[:, None, :] == P[None, :, :], axis=2), axis=1)

    # # Count matches
    # count_matches = np.sum(mask)

    # # Rows in P1 not in arr
    # P_req = P1[~mask]

    # print(f'Total number of required data points: {P1.shape[0]}')
    # print(f'Already calculated: {count_matches}')
    # print(f'Extra to calculate: {P1.shape[0] - count_matches}')
    # # print(P_req)

    # np.savez('Approximant/Req_5Fold.npz', P_req=P_req)
    
    # f = 'Approximant/Data/5Fold/Chern_NonAb_a4_N3_updated.npz'
    # plot_C_fromfile(f, 
    #                 y_axis='V', scalefactor=-1, legend_loc='upper left')


    # find_boundary('Data/Chern_NonAb_a4_N3.npz', save_filename='Data/Boundary_a4_N3.npz')
    # generate_C_data('Approximant/Data/Chern_NonAb_a9_N3.npz', N=3, a=9, max_idx=147, NonAb=True, E_gap=0.000001)
    # plot_C_fromfile('Approximant/Data/Chern_NonAb_a4_N3_updated.npz', x_axis='V')
    # data = np.load('Data/Chern_NonAb_a4_N3_updated.npz')
    # print(data['U_vals'].size)

    # plot_boundary('Approximant/Data/Boundary_a4_N3.npz', x_axis='V', xlim=(0,0.6), ylim=(0,0.9),
    #               plot_E0=True)
# 
    # plot_C(U_vals=None, V_vals=None, N=3, a=4, max_idx=27, 
    #        x_axis='V/U', E_gap=0.001, calc_C=False, NonAb=True)
    # f = 'Data/Data_a3_U0.1_N3_V0.15.npz'
    # print(check_gap_open(f))