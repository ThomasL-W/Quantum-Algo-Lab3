import sympy as sp
from dwave_simulator import DwaveSimulator

# On crée deux variables d'Ising
x1, x2 = sp.symbols("x1 x2")

# On crée le simulateur
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
ising_problem = 3*x1 - 2*x2 + 5*x1*x2
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