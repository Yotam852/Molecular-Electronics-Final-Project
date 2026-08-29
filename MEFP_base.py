import numpy as np
import matplotlib.pyplot as plt

#universal const
e = 1.602176634e-19 #elect charge C
kB = 8.617333262e-5 #eV
hbar = 1.054571817e-34 #J*Sec
_2e_h =  e/(np.pi*hbar) # 2e/h  Amp /J
# constants
beta = -0.5 #eV
M = 3 #num of molecular orbitals
alpha = np.array([0.1,0.1,0.1]) #eV in degenerate case
alpha_array=np.array([-0.1,0,0.1]) #eV non degenerate case

mu = -0.1 #eV
mu_L = 2.0
mu_R = -2.0
#mu_L=mu_R=mu # no potatial bais
mu_min = min(mu_L, mu_R)
mu_max = max(mu_L, mu_R)

T = 10 #K
T_L = 300
T_R= 1
T_R = T_L= T #no thermal bais
T_min = min(T_L, T_R)
T_max = max(T_L, T_R)
#coupling
Gamma_L = 0.001 #eV
Gamma_R = 0.001 #eV

#main functions:
# self energy calc
def self_energy(Gamma):
    return -1j * (Gamma/2.0)
# hamiltonian
def hamiltonian(M, alpha, beta):
    # create hamiltonian
    H_eff = np.zeros((M, M), dtype=complex)

    # huckel tight-binding matrix
    for i in range(M):
        H_eff[i, i] = alpha[i]
        if i < M - 1:
            H_eff[i, i + 1] = beta
            H_eff[i + 1, i] = beta
    return H_eff
#eigenstates for bounds
def eigen(H_eff):
    eigenvalues, eigenvectors = np.linalg.eigh(H_eff)
    return eigenvalues

#Green function matrix element
def G_1M_element(E_array,M, H_eff_input, Sigma_L, Sigma_R):
    #make copy
    H_eff = H_eff_input.copy()
    # adding the self energy
    H_eff[0, 0] += Sigma_L
    H_eff[M - 1, M - 1] += Sigma_R
    # stack of energy matricies
    E_mat_stack = E_array[:, np.newaxis, np.newaxis] * np.eye(M)

    # green operator for all energy values (inverse)
    G_mat_stack = np.linalg.inv(E_mat_stack - H_eff)
    #find element
    return G_mat_stack[:,0,M-1]

# unified current function
def calculate_current(Gamma_L, Gamma_R, E_grid, dE, M, H_eff_input, fd_diff):
    Sigma_L = self_energy(Gamma_L)
    Sigma_R = self_energy(Gamma_R)

    # calculate trans Trace
    G1M_array = G_1M_element(E_grid, M, H_eff_input, Sigma_L, Sigma_R)
    t_E = Gamma_L * Gamma_R * (np.abs(G1M_array) ** 2)
    # integrate current (rectangle method)
    integrand = t_E * fd_diff
    I = _2e_h * np.sum(integrand) * dE * e  # multiply by 'e' to fix units to Amp

    return I * 1e6  # microamperes

# set up
H_eff = hamiltonian(M, alpha_array, beta)
eigenvalues = eigen(H_eff)
#boundries
es_min = np.min(eigenvalues) #lowest self energy
es_max = np.max(eigenvalues) #highest self energy
buffer = 2.0 #eV
low_bound = min(T_min*kB,mu_min,es_min)-buffer
high_bound = max(T_max*kB,mu_max,es_max)+buffer
print(low_bound,high_bound)
desired_dE = 1e-4
num_steps = int((high_bound - low_bound) / desired_dE)
print(num_steps)
E_grid = np.linspace(low_bound, high_bound, num_steps)
dE = E_grid[1]-E_grid[0]

#fermi dirac
fd_L = 1.0 / (1.0 + np.exp(np.clip((E_grid - mu_L) / (kB * T_L), -100, 100)))
fd_R = 1.0 / (1.0 + np.exp(np.clip((E_grid - mu_R) / (kB * T_R), -100, 100)))
fd_diff =fd_L - fd_R

#case 1 - asymmetrical coupling
Gamma_L_values = np.linspace(0.001, 1.0, 100)
Gamma_R_fixed_values = [0.001,0.01,0.1, 0.5,1] # fixed coupling strengths

plt.figure(figsize=(8, 6))

for Gamma_R in Gamma_R_fixed_values:
    currents_asym = np.zeros_like(Gamma_L_values)
    Sigma_R = self_energy(Gamma_R)

    for i, Gamma_L in enumerate(Gamma_L_values):
        Sigma_L = self_energy(Gamma_L)

        # current calc
        currents_asym[i] = calculate_current(Gamma_L, Gamma_R, E_grid, dE, M, H_eff, fd_diff)

    plt.plot(Gamma_L_values, currents_asym, lw=2, label=rf"$\Gamma_R = {Gamma_R}$ eV")
plt.title(r"Asymmetric: Steady-State Current vs. Left Coupling ($\Gamma_L$)")
plt.axvline(x=2*np.abs(beta), color='red', linestyle=':', lw=2, label=r"$2|\beta|$")
plt.xlabel(r"Left Coupling Strength $\Gamma_L$ (eV)")
plt.ylabel(r"Current ($\mu\mathrm{A}$)")
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#case 2 - symmetrical
Gamma_values = np.linspace(0.001, 6.0, 200)
currents_sym = np.zeros_like(Gamma_values)
#trace
for i, Gamma in enumerate(Gamma_values):
    currents_sym[i] = calculate_current(Gamma, Gamma, E_grid, dE, M, H_eff, fd_diff)

plt.figure(figsize=(7, 5))
plt.plot(Gamma_values, currents_sym, color='purple', lw=2, label="Calculated Current")
plt.axvline(x=2*np.abs(beta), color='red', linestyle=':', lw=2, label=r"$2|\beta|$")
plt.title(r"Symmetric: Current vs. Coupling ($\Gamma_L = \Gamma_R = \Gamma$)")
plt.xlabel(r"Coupling Strength $\Gamma$ (eV)")
plt.ylabel(r"Current ($\mu\mathrm{A}$)")
plt.legend(loc='upper right', fontsize=10, shadow=True, borderpad=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#2D map
#grid
Gamma_L_range = np.linspace(0.001, 3.0, 50)
Gamma_R_range = np.linspace(0.001, 3.0, 50)

GL_grid, GR_grid = np.meshgrid(Gamma_L_range, Gamma_R_range)
current_2D = np.zeros_like(GL_grid)

for i in range(GL_grid.shape[0]):
    for j in range(GL_grid.shape[1]):
        GL = GL_grid[i, j]
        GR = GR_grid[i, j]
        current_2D[i, j] = calculate_current(GL, GR, E_grid, dE, M, H_eff, fd_diff)
plt.figure(figsize=(9, 7))
contour = plt.contourf(GL_grid, GR_grid, current_2D, levels=50, cmap='viridis')

#colorbar to show what current value each color represents
plt.colorbar(contour, label=r"Current ($\mu\mathrm{A}$)")

# dashed line to represent the Symmetric Case (Gamma_L = Gamma_R)
plt.plot([0.001, 3.0], [0.001, 3.0], color='white', linestyle='--', alpha=0.7, label=r"symmetric case ($\Gamma_L = \Gamma_R$)")
plt.title(r"current heat map vs coupling strength")
plt.xlabel(r"Left Coupling $\Gamma_L$ (eV)")
plt.ylabel(r"Right Coupling $\Gamma_R$ (eV)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# ==========================================
# 3D Surface Plot with Symmetry Line (No Color Gradient)
# ==========================================
Gamma_L_range = np.linspace(0.001, 3.0, 50)
Gamma_R_range = np.linspace(0.001, 3.0, 50)

X, Y = np.meshgrid(Gamma_L_range, Gamma_R_range)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        GL = X[i, j]
        GR = Y[i, j]
        Z[i, j] = calculate_current(GL, GR, E_grid, dE, M, H_eff, fd_diff)

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')

# Solid surface without colormap gradient
surf = ax.plot_surface(X, Y, Z, color='royalblue', edgecolor='none', alpha=0.7)

# --- Symmetric Path (Gamma_L = Gamma_R) ---
gamma_sym = np.linspace(0.001, 3.0, 50)
Z_sym = np.array([
    calculate_current(g, g, E_grid, dE, M, H_eff, fd_diff) for g in gamma_sym
])
ax.plot(gamma_sym, gamma_sym, Z_sym, color='red', lw=3, label=r'Symmetry Line ($\Gamma_L = \Gamma_R$)')

ax.set_xlabel(r'$\Gamma_L$ [eV]', labelpad=10)
ax.set_ylabel(r'$\Gamma_R$ [eV]', labelpad=10)
ax.set_zlabel(r'Current [$\mu$A]', labelpad=10)
plt.title(f'3D Surface Plot of Steady-State Current\n(Bias: $\mu_L={mu_L}$, $\mu_R={mu_R}$ eV)')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()