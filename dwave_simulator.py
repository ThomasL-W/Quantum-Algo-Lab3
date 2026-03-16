import numpy as np
import sympy as sp


class DwaveSimulator:
    def __init__(self, nb_points: int = 101):
        # On vérifie qu'on a au moins un début et une fin
        if nb_points < 2:
            raise ValueError("nb_points doit être >= 2")

        # On crée les valeurs de s entre 0 et 1
        s_values = np.linspace(0.0, 1.0, nb_points)

        # TODO 1 : schedule linéaire
        # A(s) diminue de 1 à 0
        # B(s) augmente de 0 à 1
        self.annealing_schedule = [
            {"s": float(s), "A": float(1.0 - s), "B": float(s)}
            for s in s_values
        ]

    def get_schedule_arrays(self):
        # Méthode pratique pour récupérer s, A et B séparément
        s = np.array([point["s"] for point in self.annealing_schedule], dtype=float)
        A = np.array([point["A"] for point in self.annealing_schedule], dtype=float)
        B = np.array([point["B"] for point in self.annealing_schedule], dtype=float)
        return s, A, B

    def _tensor_product(self, matrices):
        # Fait le produit tensoriel de plusieurs matrices
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)
        return result

    def _single_z_operator(self, n, target):
        # Construit un opérateur avec Z sur une seule variable
        # Exemple si n=3 et target=1 : I ⊗ Z ⊗ I
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
        # Construit un opérateur avec Z sur deux variables
        # Exemple si n=3 et cibles 0 et 2 : Z ⊗ I ⊗ Z
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
        # TODO 2 :
        # transformer l'expression d'Ising en matrice Hfinal
        expr = sp.expand(ising_problem)

        # On récupère les variables du problème
        variables = sorted(expr.free_symbols, key=lambda s: s.name)
        n = len(variables)

        # Taille de la matrice finale : 2^n x 2^n
        dim = 2 ** n
        Hfinal = np.zeros((dim, dim), dtype=float)

        # On parcourt chaque terme de l'expression
        for term in expr.as_ordered_terms():
            coeff, factors = term.as_coeff_mul()

            # On garde seulement les variables d'Ising
            spin_vars = [f for f in factors if f in variables]

            # Si 1 variable : terme linéaire
            if len(spin_vars) == 1:
                i = variables.index(spin_vars[0])
                Hfinal += float(coeff) * self._single_z_operator(n, i)

            # Si 2 variables : terme quadratique
            elif len(spin_vars) == 2:
                i = variables.index(spin_vars[0])
                j = variables.index(spin_vars[1])
                Hfinal += float(coeff) * self._double_z_operator(n, i, j)

            # Si aucune variable : constante
            elif len(spin_vars) == 0:
                Hfinal += float(coeff) * np.eye(dim)

            else:
                raise ValueError(
                    "Le problème d'Ising ne doit contenir que des termes linéaires ou quadratiques."
                )
        return Hfinal

    def build_Hinit(self, n):
        # TODO 3 : construire l'Hamiltonien initial
        if n < 1:
            raise ValueError("n doit être >= 1")

        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]], dtype=float)

        dim = 2 ** n
        Hinit = np.zeros((dim, dim), dtype=float)

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
        # On récupère les variables du problème pour connaître n
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

        # On construit les deux Hamiltoniens une seule fois
        Hfinal = self.build_Hfinal(ising_problem)
        Hinit = self.build_Hinit(n)

        eigenvalues_history = []
        eigenvectors_history = []

        # On parcourt toute la schedule
        for point in self.annealing_schedule:
            A = point["A"]
            B = point["B"]

            # Hamiltonien instantané
            Hs = A * Hinit + B * Hfinal

            # Diagonalisation
            # eigh est adaptée aux matrices symétriques/hermitiennes
            eigenvalues, eigenvectors = np.linalg.eigh(Hs)

            # Les valeurs propres sont triées dans l'ordre croissant
            eigenvalues_history.append(eigenvalues[:nb_eigenvalues])
            eigenvectors_history.append(eigenvectors[:, :nb_eigenvalues])

        return eigenvalues_history, eigenvectors_history