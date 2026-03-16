import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class DwaveSimulator:
    def __init__(self, nb_points: int = 101):
        # TODO 1 : créer une annealing schedule linéaire
        if nb_points < 2:
            raise ValueError("nb_points doit être >= 2")

        s_values = np.linspace(0.0, 1.0, nb_points)
        self.annealing_schedule = [
            {"s": float(s), "A": float(1.0 - s), "B": float(s)}
            for s in s_values
        ]

    def get_schedule_arrays(self):
        s = np.array([point["s"] for point in self.annealing_schedule], dtype=float)
        A = np.array([point["A"] for point in self.annealing_schedule], dtype=float)
        B = np.array([point["B"] for point in self.annealing_schedule], dtype=float)
        return s, A, B

    def _tensor_product(self, matrices):
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)
        return result

    def _single_z_operator(self, n, target):
        # opérateur avec Z sur une seule position
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]], dtype=float)

        matrices = []
        for i in range(n):
            if i == target:
                matrices.append(Z)
            else:
                matrices.append(I)

        return self._tensor_product(matrices)

    def _double_z_operator(self, n, target1, target2):
        # opérateur avec Z sur deux positions
        I = np.eye(2)
        Z = np.array([[1, 0], [0, -1]], dtype=float)

        matrices = []
        for i in range(n):
            if i == target1 or i == target2:
                matrices.append(Z)
            else:
                matrices.append(I)

        return self._tensor_product(matrices)

    def build_Hfinal(self, ising_problem):
        # TODO 2 : construire Hfinal à partir du problème d'Ising
        expr = sp.expand(ising_problem)

        variables = sorted(expr.free_symbols, key=lambda s: s.name)
        n = len(variables)

        if n == 0:
            raise ValueError("Le problème d'Ising doit contenir au moins une variable.")

        dim = 2 ** n
        Hfinal = np.zeros((dim, dim), dtype=float)

        for term in expr.as_ordered_terms():
            coeff, factors = term.as_coeff_mul()
            spin_vars = [f for f in factors if f in variables]

            # terme linéaire
            if len(spin_vars) == 1:
                i = variables.index(spin_vars[0])
                Hfinal += float(coeff) * self._single_z_operator(n, i)

            # terme quadratique
            elif len(spin_vars) == 2:
                i = variables.index(spin_vars[0])
                j = variables.index(spin_vars[1])
                Hfinal += float(coeff) * self._double_z_operator(n, i, j)

            # terme constant
            elif len(spin_vars) == 0:
                Hfinal += float(coeff) * np.eye(dim)

            else:
                raise ValueError(
                    "Le problème d'Ising ne doit contenir que des termes linéaires ou quadratiques."
                )

        return Hfinal

    def build_Hinit(self, n):
        # TODO 3 : construire Hinit
        if n < 1:
            raise ValueError("n doit être >= 1")

        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]], dtype=float)

        dim = 2 ** n
        Hinit = np.zeros((dim, dim), dtype=float)

        # on met X sur chaque qubit, un par un
        for target in range(n):
            matrices = []
            for i in range(n):
                if i == target:
                    matrices.append(X)
                else:
                    matrices.append(I)

            Hinit += self._tensor_product(matrices)

        return -Hinit

    def simulate_evolution(self, ising_problem, nb_eigenvalues):
        # TODO 4 : simuler l'évolution sans bruit
        expr = sp.expand(ising_problem)
        variables = sorted(expr.free_symbols, key=lambda s: s.name)
        n = len(variables)

        if n == 0:
            raise ValueError("Le problème d'Ising doit contenir au moins une variable.")

        if nb_eigenvalues < 1:
            raise ValueError("nb_eigenvalues doit être >= 1")

        dim = 2 ** n
        if nb_eigenvalues > dim:
            raise ValueError(f"nb_eigenvalues doit être <= {dim}")

        Hfinal = self.build_Hfinal(ising_problem)
        Hinit = self.build_Hinit(n)

        eigenvalues_history = []
        eigenvectors_history = []

        for point in self.annealing_schedule:
            A = point["A"]
            B = point["B"]

            # Hamiltonien à l'instant s
            Hs = A * Hinit + B * Hfinal

            # diagonalisation
            eigenvalues, eigenvectors = np.linalg.eigh(Hs)

            # on garde les plus petites valeurs propres
            eigenvalues_history.append(eigenvalues[:nb_eigenvalues])
            eigenvectors_history.append(eigenvectors[:, :nb_eigenvalues])

        return eigenvalues_history, eigenvectors_history

    def plot_eigenvalues(self, eigenvalues_history, title="Evolution of the lowest eigenvalues"):
        # TODO 5 : tracer les valeurs propres
        eigenvalues_array = np.array(eigenvalues_history, dtype=float)

        for i in range(eigenvalues_array.shape[1]):
            plt.plot(eigenvalues_array[:, i], label=f"Eigenvalue {i}")

        plt.xlabel("Annealing step")
        plt.ylabel("Energy")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_spectral_gap(self, eigenvalues_history, title="Spectral gap during annealing"):
        # TODO 5 : tracer le spectral gap
        eigenvalues_array = np.array(eigenvalues_history, dtype=float)

        if eigenvalues_array.shape[1] < 2:
            raise ValueError("Il faut au moins 2 valeurs propres pour calculer le spectral gap.")

        gap = eigenvalues_array[:, 1] - eigenvalues_array[:, 0]

        plt.plot(gap)
        plt.xlabel("Annealing step")
        plt.ylabel("Spectral gap")
        plt.title(title)
        plt.grid(True)
        plt.show()

        return gap

    def simulate_noisy_evolution(
        self,
        ising_problem,
        nb_eigenvalues=5,
        noise_std=0.1,
        random_seed=None
    ):
        # TODO 6 : simuler l'évolution avec bruit sur Hfinal
        expr = sp.expand(ising_problem)
        variables = sorted(expr.free_symbols, key=lambda s: s.name)
        n = len(variables)

        if n == 0:
            raise ValueError("Le problème d'Ising doit contenir au moins une variable.")

        dim = 2 ** n
        if nb_eigenvalues < 1 or nb_eigenvalues > dim:
            raise ValueError(f"nb_eigenvalues doit être entre 1 et {dim}")

        rng = np.random.default_rng(random_seed)

        Hinit = self.build_Hinit(n)
        Hfinal = self.build_Hfinal(ising_problem)

        eigenvalues_history = []
        eigenvectors_history = []

        for point in self.annealing_schedule:
            A = point["A"]
            B = point["B"]

            # bruit gaussien symétrique qui change à chaque étape
            noise = rng.normal(loc=0.0, scale=noise_std, size=(dim, dim))
            noise = (noise + noise.T) / 2.0

            # seul Hfinal est bruité
            Hs_noisy = A * Hinit + B * (Hfinal + noise)

            eigenvalues, eigenvectors = np.linalg.eigh(Hs_noisy)

            eigenvalues_history.append(eigenvalues[:nb_eigenvalues])
            eigenvectors_history.append(eigenvectors[:, :nb_eigenvalues])

        return eigenvalues_history, eigenvectors_history

    def generate_random_ising_problem(self, n, weight_min=-1.0, weight_max=1.0, random_seed=None):
        # TODO 6 : générer un problème d'Ising aléatoire
        if n < 1:
            raise ValueError("n doit être >= 1")

        rng = np.random.default_rng(random_seed)
        variables = sp.symbols(f"x1:{n+1}")

        expr = 0

        # termes linéaires
        for x in variables:
            h = rng.uniform(weight_min, weight_max)
            expr += h * x

        # termes quadratiques
        for i in range(n):
            for j in range(i + 1, n):
                J = rng.uniform(weight_min, weight_max)
                expr += J * variables[i] * variables[j]

        return sp.expand(expr)

    def rescale_ising_problem(self, ising_problem, scale):
        # TODO 6 : rescaling des poids
        return sp.expand(scale * ising_problem)
    
    def simulate_noisy_evolution_both(
        self,
        ising_problem,
        nb_eigenvalues=5,
        noise_std_final=0.1,
        noise_std_init=0.1,
        random_seed=None
    ):
        """
        Bruit sur Hfinal ET sur Hinit.
        Le bruit change à chaque étape.
        """
        expr = sp.expand(ising_problem)
        variables = sorted(expr.free_symbols, key=lambda s: s.name)
        n = len(variables)

        if n == 0:
            raise ValueError("Le problème d'Ising doit contenir au moins une variable.")

        dim = 2 ** n
        if nb_eigenvalues < 1 or nb_eigenvalues > dim:
            raise ValueError(f"nb_eigenvalues doit être entre 1 et {dim}")

        rng = np.random.default_rng(random_seed)

        Hinit = self.build_Hinit(n)
        Hfinal = self.build_Hfinal(ising_problem)

        eigenvalues_history = []
        eigenvectors_history = []

        for point in self.annealing_schedule:
            A = point["A"]
            B = point["B"]

            noise_init = rng.normal(loc=0.0, scale=noise_std_init, size=(dim, dim))
            noise_init = (noise_init + noise_init.T) / 2.0

            noise_final = rng.normal(loc=0.0, scale=noise_std_final, size=(dim, dim))
            noise_final = (noise_final + noise_final.T) / 2.0

            Hs_noisy = A * (Hinit + noise_init) + B * (Hfinal + noise_final)

            eigenvalues, eigenvectors = np.linalg.eigh(Hs_noisy)

            eigenvalues_history.append(eigenvalues[:nb_eigenvalues])
            eigenvectors_history.append(eigenvectors[:, :nb_eigenvalues])

        return eigenvalues_history, eigenvectors_history


    def create_duplicated_node_instance(self, h1=0.5, h2=-0.3, j12=0.8, strong_coupling=-4.0):
        """
        Crée une petite instance avec un nœud dupliqué.
        x1 et x3 représentent le même nœud et sont fortement couplés.
        """
        x1, x2, x3 = sp.symbols("x1 x2 x3")

        expr = (
            h1 * x1
            + h2 * x2
            + h1 * x3
            + j12 * x1 * x2
            + j12 * x3 * x2
            + strong_coupling * x1 * x3
        )

        return sp.expand(expr)