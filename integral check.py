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

mu = 0.1 #eV
mu_L = 2.0
mu_R = -2.0
mu_L=mu_R=mu # no potatial bais
mu_min = min(mu_L, mu_R)
mu_max = max(mu_L, mu_R)

T = 10 #K
T_L = 300
T_R= 1
#T_R = T_L= T #no thermal bais
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
    I = _2e_h * np.sum(integrand) * dE * e  #multiply by 'e' to fix units to Amp

    return I #amp
# set up
H_eff = hamiltonian(M, alpha_array, beta)
eigenvalues = eigen(H_eff)
#boundries
es_min = np.min(eigenvalues) #lowest self energy
es_max = np.max(eigenvalues) #highest self energy
buffer = np.linspace(0.0, 100.0, 500)#eV
current_b = []
current_b = []

for b in buffer:
    low_bound = min(T_min * kB, mu_min, es_min) - b
    high_bound = max(T_max * kB, mu_max, es_max) + b
    print(low_bound, high_bound)

    num_steps = 1000000
    E_grid = np.linspace(low_bound, high_bound, num_steps)
    dE = E_grid[1] - E_grid[0]

    # fermi dirac
    fd_L = 1.0 / (1.0 + np.exp(np.clip((E_grid - mu_L) / (kB * T_L), -100, 100)))
    fd_R = 1.0 / (1.0 + np.exp(np.clip((E_grid - mu_R) / (kB * T_R), -100, 100)))
    fd_diff = fd_L - fd_R

    Gamma = 0.01
    currents_sym = calculate_current(Gamma, Gamma, E_grid, dE, M, H_eff, fd_diff)

    # 2. Use .append() to add the newly calculated float to your list
    current_b.append(currents_sym)
diff = []
print(current_b)

rtol = 1e-5

for i in range(len(current_b)-1):
    # Calculate the absolute difference between steps
    step_diff = abs(current_b[i+1] - current_b[i])
    diff.append(step_diff)

    # CONDITION: Is the change smaller than our relative tolerance?
    if step_diff / abs(current_b[i+1]) < rtol:
        print(f"SUCCESS: Converged at index {i+1} (buffer = {buffer[i+1]} eV)")
print(diff)
fig = plt.figure(figsize=(10, 7))
plt.figure(figsize=(7, 5))
plt.plot(buffer, current_b, marker='o', linestyle='-', color='purple', lw=2)
# Graph labels and title
plt.title("Current vs. Grid Boundary Buffer")
plt.xlabel("Buffer Size (eV)")
plt.ylabel(r"Current ($\mu\mathrm{A}$)")

# Visual formatting
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Display the plot
plt.show()
fig = plt.figure(figsize=(10, 7))
plt.figure(figsize=(7, 5))
plt.plot(buffer[1:], diff, marker='o', linestyle='-', color='purple', lw=2)
# Graph labels and title
plt.title("diff vs. Grid Boundary Buffer")
plt.xlabel("Buffer Size (eV)")
plt.ylabel(r"diff")

# Visual formatting
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Display the plot
plt.show()
relative_diff = np.zeros(len(current_b))
for i in range(1, len(current_b)):
    relative_diff[i] = abs(current_b[i] - current_b[i-1]) / abs(current_b[i])

# Create a two-panel plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Top Panel: Absolute Current vs Buffer Size
ax1.plot(buffer, current_b, marker='o', color='purple', lw=2)
ax1.set_title("Integral Convergence vs. Energy Boundary Buffer")
ax1.set_ylabel(r"Calculated Current (Hz)")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.ticklabel_format(useOffset=False)
# Bottom Panel: Relative Change vs Buffer Size
ax2.plot(buffer, relative_diff, marker='s', color='red', lw=2)
ax2.set_yscale('log') # Log scale highlights the exponential drop to zero
ax2.set_xlabel("Buffer Size (amp)")
ax2.set_ylabel("Relative Change")
ax2.axhline(1e-5, color='black', linestyle=':', label="Convergence Limit (1e-5)")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()