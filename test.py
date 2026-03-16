import sympy as sp
from dwave_simulator import DwaveSimulator

# ---------------------------
# Initialisation
# ---------------------------
x1, x2 = sp.symbols("x1 x2")
sim = DwaveSimulator()

# ---------------------------
# TEST TODO 1 : annealing schedule
# ---------------------------
print("=== TODO 1 : annealing schedule ===")
print("3 premiers points :")
print(sim.annealing_schedule[:3])

print("\n3 derniers points :")
print(sim.annealing_schedule[-3:])

# ---------------------------
# TEST TODO 2 : Hfinal
# ---------------------------
print("\n=== TODO 2 : Hfinal ===")
ising_problem = 3 * x1 - 2 * x2 + 5 * x1 * x2
Hfinal = sim.build_Hfinal(ising_problem)

print("Problème d'Ising :", ising_problem)
print("Hfinal =")
print(Hfinal)
print("Dimension de Hfinal :", Hfinal.shape)

# ---------------------------
# TEST TODO 3 : Hinit
# ---------------------------
print("\n=== TODO 3 : Hinit ===")
Hinit = sim.build_Hinit(2)

print("Hinit =")
print(Hinit)
print("Dimension de Hinit :", Hinit.shape)

# ---------------------------
# TEST TODO 4 : simulate_evolution
# ---------------------------
print("\n=== TODO 4 : simulate_evolution ===")
eigenvalues_history, eigenvectors_history = sim.simulate_evolution(
    ising_problem,
    nb_eigenvalues=3
)

print("Nombre d'étapes :", len(eigenvalues_history))

print("\n3 plus petites valeurs propres au début :")
print(eigenvalues_history[0])

print("\n3 plus petites valeurs propres au milieu :")
print(eigenvalues_history[len(eigenvalues_history) // 2])

print("\n3 plus petites valeurs propres à la fin :")
print(eigenvalues_history[-1])

print("\nDimension du tableau des vecteurs propres à la fin :")
print(eigenvectors_history[-1].shape)

print("\nVecteur propre associé à la plus petite valeur propre finale :")
print(eigenvectors_history[-1][:, 0])

# ---------------------------
# TEST TODO 5 : plots + spectral gap
# ---------------------------
print("\n=== TODO 5 : plots + spectral gap ===")

sim.plot_eigenvalues(
    eigenvalues_history,
    title="Evolution of the lowest eigenvalues"
)

gap = sim.plot_spectral_gap(
    eigenvalues_history,
    title="Spectral gap during annealing"
)

print("Spectral gap au début :", gap[0])
print("Spectral gap au milieu :", gap[len(gap) // 2])
print("Spectral gap à la fin :", gap[-1])
print("Gap minimal :", gap.min())
print("Étape du gap minimal :", gap.argmin())

# ---------------------------
# TEST TODO 6 : noisy simulation
# ---------------------------
print("\n=== TODO 6 : noisy simulation ===")

random_problem = sim.generate_random_ising_problem(
    n=3,
    random_seed=42
)

print("Problème aléatoire :", random_problem)

noisy_eigenvalues_history, noisy_eigenvectors_history = sim.simulate_noisy_evolution(
    random_problem,
    nb_eigenvalues=5,
    noise_std=0.1,
    random_seed=42
)

print("Nombre d'étapes bruitées :", len(noisy_eigenvalues_history))

print("\n5 plus petites valeurs propres au début :")
print(noisy_eigenvalues_history[0])

print("\n5 plus petites valeurs propres à la fin :")
print(noisy_eigenvalues_history[-1])

sim.plot_eigenvalues(
    noisy_eigenvalues_history,
    title="Noisy evolution of the lowest eigenvalues"
)

noisy_gap = sim.plot_spectral_gap(
    noisy_eigenvalues_history,
    title="Noisy spectral gap"
)

print("\nSpectral gap bruité au début :", noisy_gap[0])
print("Spectral gap bruité à la fin :", noisy_gap[-1])
print("Gap bruité minimal :", noisy_gap.min())
print("Étape du gap bruité minimal :", noisy_gap.argmin())

# ---------------------------
# TEST BONUS TODO 6 : rescaling
# ---------------------------
print("\n=== BONUS TODO 6 : rescaling ===")

scaled_problem = sim.rescale_ising_problem(random_problem, scale=0.5)
print("Problème rescalé :", scaled_problem)

scaled_eigenvalues_history, _ = sim.simulate_noisy_evolution(
    scaled_problem,
    nb_eigenvalues=5,
    noise_std=0.1,
    random_seed=42
)

scaled_gap = sim.plot_spectral_gap(
    scaled_eigenvalues_history,
    title="Spectral gap with rescaled Ising problem"
)

print("Gap minimal avec rescaling :", scaled_gap.min())
print("Étape du gap minimal avec rescaling :", scaled_gap.argmin())