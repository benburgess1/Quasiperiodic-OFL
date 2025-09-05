import numpy as np

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
l = np.arange(8)
phi0 = 0
N = 5
U = -Umag * np.exp(1j * (phi0 - 2 * np.pi * N * l / 8))
dphi = np.angle(U[1]) - np.angle(U[0])
w = np.exp(1j*dphi)
Vmag = 0.01
print(f'Parameters: U={Umag}, V={Vmag}, N={N}\n')


### Build K-point Hamiltonian
H_U = np.zeros((16, 16), dtype=np.complex128)
for i in range(8):
    orb = np.zeros((8,8), dtype=np.complex128)
    orb[(i+3)%8, i] = U[i]
    orb[i, (i+3)%8] = U[(i+4)%8]
    dH = np.kron(orb, sp)
    H_U += (dH + dH.conj().T)

orb = np.zeros((8,8), dtype=np.complex128)
for i in range(8):
    orb[(i+2)%8, i] += 2*Vmag
    orb[i, (i+2)%8] += 2*Vmag
# print(np.round(np.real(orb)))
H_V = np.kron(orb, sz)
# H_V += H_V.conj().T
# print(np.round(np.real(H_V)))

H = H_U + H_V
## Checks:
# print('\nChecking H:')
# res = np.linalg.norm(H_U - H_U.conj().T)
# print(f'Hermiticity: H_U - H_U.conj().T = {res}')
# res = np.linalg.norm(H_V - H_V.conj().T)
# print(f'Hermiticity: H_V - H_V.conj().T = {res}')
# # print(H)        # Looks OK
# evals,evects = np.linalg.eigh(H)
# print(f'Eigenvalues: {evals}')    # Matches expected degeneracies: 4, 4, 4, 4 for V=0; 2, 2, 2, 2, 2, 2, 2, 2 for V != 0


### Build Unitary representation of R(pi/4) x exp(-1j * dphi * sz)
c8_orb = np.zeros((8, 8), dtype=np.complex128)
for i in range(8):
    c8_orb[(i+1)%8, i] = 1
c8_spin = np.array([[np.exp(1j*dphi/2), 0],
                    [0, np.exp(-1j*dphi/2)]])
U_c8 = np.kron(c8_orb, c8_spin)
## Check unitarity
# print('\nTesting U_c8:')
# norm = np.linalg.norm(U_c8.conj().T @ U_c8 - np.eye(16))
# print(f'U_c8 U_c8^dag: {norm}')
# ## Check commutativity with H
# norm = np.linalg.norm(H_U @ U_c8 - U_c8 @ H_U)
# print(f'[H_U, U_c8] = {norm}')     # Vanishes as required
# norm = np.linalg.norm(H_V @ U_c8 - U_c8 @ H_V)
# print(f'[H_V, U_c8] = {norm}')     # Vanishes as required


## Build Unitary representation of inversion P x sz
# P_orb = np.zeros((8, 8), dtype=np.complex128)
# for i in range(8):
#     P_orb[(i+4)%8, i] = 1
# U_Pz = np.kron(P_orb, sz)
U_Pz = np.linalg.matrix_power(U_c8, 4)
## Check commutativity with H
# norm = np.linalg.norm(H @ U_Pz - U_Pz @ H)
# print(norm)     # Vanishes as required
## Check commutativity of U_C8 and U_Pz
# norm = np.linalg.norm(U_c8 @ U_Pz - U_Pz @ U_c8)
# print(norm)     # Vanishes as expected


### Build Unitary representation of mirror sigma_v x I2
U_sv = np.zeros((16, 16), dtype=np.complex128)
for i in range(8):
    sv_orb = np.zeros((8, 8), dtype=np.complex128)
    sv_orb[i, (8-i)%8] = 1
    # t = i+2
    sv_spin = np.array([[np.exp(1j*dphi*(i-0.5)), 0],
                        [0, np.exp(-1j*dphi*(i-0.5))]])
    # sv_spin = I2
    U_sv += np.kron(sv_orb, sv_spin)
sv_orb = np.zeros((8, 8), dtype=np.complex128)
for i in range(8):
    sv_orb[(8-i)%8, i] = 1
# print(np.round(np.real(sv_orb)))
# print(np.linalg.norm(sv_orb - sv_orb.T))
# print(np.round(np.real(sv_orb.T @ sv_orb)))
# Sv = np.kron(sv_orb, I2)
# print('')
# print(f'Original block: \n{H[6:8,:2]}')
# H2 = Sv @ H @ Sv.conj().T
# print(f'Block after performing spatial reflection only: \n{H2[6:8,:2]}')
# sv_spin = np.array([[np.exp(-1j*dphi*(0.5)), 0],
#                         [0, np.exp(1j*dphi*(0.5))]])
# B = sv_spin @ H2[6:8,:2] @ sv_spin.conj().T
# print(f'Block after additionally performing spin rotation: \n{B}')
# Combine the operations

    # t = i+2
    # sv_spin = np.array([[np.exp(1j*dphi*(i-0.5)), 0],
    #                     [0, np.exp(-1j*dphi*(i-0.5))]])
    # sv_spin = I2
    # U_sv += np.kron(sv_orb, sv_spin)
# t = 5*dphi/2
# sv_spin = np.array([[0, np.exp(1j*t)],
#                     [np.exp(1j*t), 0]])
# U_sv = np.kron(sv_orb, sv_spin)
# ## Check unitarity
# print('\nTesting U_sv:')
# # print(np.round(U_sv[:,7],3))
# # print(np.round(H_V, 1))
# # print(np.round(U_sv @ H_V @ U_sv.conj().T, 1))
# norm = np.linalg.norm(U_sv.conj().T @ U_sv - np.eye(16))
# print(f'U_sv U_sv^dag: {norm}')     # Vanishes as required
# # ## Check commutativity with H
# norm = np.linalg.norm(H_U @ U_sv - U_sv @ H_U)
# print(f'[H_U, U_sv] = {norm}')     # Vanishes as required
# norm = np.linalg.norm(H_V @ U_sv - U_sv @ H_V)
# print(f'[H_V, U_sv] = {norm}')     # Vanishes as required
# # ## Check commutativity of U_C8 and U_sv
# norm = np.linalg.norm(U_c8 @ U_sv - U_sv @ U_c8)
# print(f'[U_c8, U_sv] = {norm}')     # Non-zero as expected
# norm = np.linalg.norm(U_Pz @ U_sv - U_sv @ U_Pz)
# print(f'[U_Pz, U_sv] = {norm}')     # Non-zero as expected


### Build unitary representation of sigma_d x sy
sd_orb = np.zeros((8,8), dtype=np.complex128)
for i in range(8):
    sd_orb[7-i, i] = 1
# print(np.round(np.real(sd_orb)))
U_sd = np.kron(sd_orb, sy)
# print('\nTesting U_sd:')
# norm = np.linalg.norm(U_sd.conj().T @ U_sd - np.eye(16))
# print(f'U_sd U_sd^dag: {norm}')     # Vanishes as required
# # ## Check commutativity with H
# norm = np.linalg.norm(H_U @ U_sd - U_sd @ H_U)
# print(f'[H_U, U_sd] = {norm}')     # Vanishes as expected
# norm = np.linalg.norm(H_V @ U_sd - U_sd @ H_V)
# print(f'[H_V, U_sd] = {norm}')     # Non-zero as expected
# norm = np.linalg.norm(H_V @ U_sd + U_sd @ H_V)
# print(f'anticomm(H_V, U_sd) = {norm}')     # Vanishes as expected
# # ## Check commutativity of U_C8 and U_sv
# norm = np.linalg.norm(U_c8 @ U_sd - U_sd @ U_c8)
# print(f'[U_c8, U_sd] = {norm}')     # Non-zero as expected



### Build group elements: C8^k and C8^k * sigma_v
rotations = []
for k in range(8):
    rotations.append(np.linalg.matrix_power(U_c8, k))

mirrors = []
for k in range(8):
    mirrors.append(np.linalg.matrix_power(U_c8, k) @ U_sv)

group_elems = rotations + mirrors  # total 16 elements
group_names = [f"C8^{k}" for k in range(8)] + [f"C8^{k}*sigma" for k in range(8)]


### Build (anti-)unitary representation of TRS K x sx
U_orb = np.zeros((8, 8), dtype=np.complex128)
for i in range(8):
    U_orb[(i+4)%8, i] = 1
U_Tx = np.kron(U_orb, sx)

def left_T(H):
    return U_Tx @ H.conj()

def right_T(H):
    return H @ U_Tx.conj().T
# print('\nTesting T_x = K x sx:')
# norm = np.linalg.norm(right_T(H_U) - left_T(H_U))
# print(f'[H_U, T_x] = {norm}')
# norm = np.linalg.norm(right_T(H_V) - left_T(H_V))
# print(f'[H_V, T_x] = {norm}')
# norm = np.linalg.norm(right_T(U_c8) - left_T(U_c8))
# print(f'[U_c8, T_x] = {norm}')
# norm = np.linalg.norm(right_T(U_sv) - left_T(U_sv))
# print(f'[U_sv, T_x] = {norm}')
# norm = np.linalg.norm(right_T(H_V @ U_sd) - U_sd @ left_T(H_V))
# print(f'[H_V, U_sd x T_x] = {norm}')

# norm = np.linalg.norm(U_Tx @ H.conj() - H @ U_Tx)
# print(norm)


# norm = np.linalg.norm(U_Tx @ (U_c8).conj() - (U_c8) @ U_Tx)
# print(norm)
# norm = np.linalg.norm(U_Tx @ (-1j*U_Pz).conj() - (-1j*U_Pz) @ U_Tx)
# print(norm)

# norm = np.linalg.norm(U_Tx @ H - H @ U_Tx)
# print(norm)


### Build anti-unitary representation of I8 K x sy
U_Tp = np.kron(np.eye(8), sy)
# print(np.round(np.real(U_Tp)))
def left_Tp(H):
    return U_Tp @ H.conj()

def right_Tp(H):
    return H @ U_Tp.conj().T
## Check commutativity with H
# print('\nTesting S = P_z T_x:')
# norm = np.linalg.norm(right_Tp(H_U) - left_Tp(H_U))
# print(f'[H_U, S] = {norm}')
# norm = np.linalg.norm(right_Tp(H_V) - left_Tp(H_V))
# print(f'[H_V, S] = {norm}')
# norm = np.linalg.norm(right_Tp(U_c8) - left_Tp(U_c8))
# print(f'[U_c8, S] = {norm}')
# norm = np.linalg.norm(right_Tp(U_sv) - left_Tp(U_sv))
# print(f'[U_sv, S] = {norm}')


### Testing All Operator Commutations ###
print('Operators: C8, sig_d, Tx, Pz.Tx, sig_d.Tx\n')
print('Testing commutations with H_U:')
norm = np.linalg.norm(H_U @ U_c8 - U_c8 @ H_U)
print(f'[H_U, C8] = {np.round(norm,4)}')
norm = np.linalg.norm(H_U @ U_sd - U_sd @ H_U)
print(f'[H_U, sig_d] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(H_U) - left_T(H_U))
print(f'[H_U, Tx] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(H_U @ U_Pz) - U_Pz @ left_T(H_U))
print(f'[H_U, Pz.Tx] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(H_U @ U_sd) - U_sd @ left_T(H_U))
print(f'[H_U, sig_d.Tx] = {np.round(norm,4)}')
# print('All operators individually commute, so products Pz.Tx and sig_d.Tx also commute.\n')

print('\nTesting commutations with H_V:')
norm = np.linalg.norm(H_V @ U_c8 - U_c8 @ H_V)
print(f'[H_V, C8] = {np.round(norm,4)}')
norm = np.linalg.norm(H_V @ U_sd - U_sd @ H_V)
print(f'[H_V, sig_d] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(H_V) - left_T(H_V))
print(f'[H_V, Tx] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(H_V @ U_Pz) - U_Pz @ left_T(H_V))
print(f'[H_V, Pz.Tx] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(H_V @ U_sd) - U_sd @ left_T(H_V))
print(f'[H_V, sig_d.Tx] = {np.round(norm,4)}')

print('\nTesting mutual commutations of operators:')
norm = np.linalg.norm(U_c8 @ U_sd - U_sd @ U_c8)
print(f'[C8, sig_d] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(U_c8) - left_T(U_c8))
print(f'[C8, Tx] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(U_c8 @ U_Pz) - U_Pz @ left_T(U_c8))
print(f'[C8, Pz.Tx] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(U_c8 @ U_sd) - U_sd @ left_T(U_c8))
print(f'[C8, sig_d.Tx] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(U_sd) - left_T(U_sd))
print(f'[sig_d, Tx] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(U_sd @ U_Pz) - U_Pz @ left_T(U_sd))
print(f'[sig_d, Pz.Tx] = {np.round(norm,4)}')
norm = np.linalg.norm(right_T(U_sd @ U_sd) - U_sd @ left_T(U_sd))
print(f'[sig_d, sig_d.Tx] = {np.round(norm,4)}')
