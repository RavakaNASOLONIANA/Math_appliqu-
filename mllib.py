"""
Mini-librairie de Machine Learning "from scratch" (NumPy uniquement).
Contient :
 - standardize / StandardScalerScratch
 - PCAScratch
 - LogisticRegressionRidgeScratch (descente de gradient, régularisation L2)
"""
import numpy as np


class StandardScalerScratch:
    """Centrage-réduction : X_standard = (X - mu) / sigma"""

    def fit(self, X):
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0)
        # évite une division par zéro si une colonne est constante
        self.sigma_[self.sigma_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu_) / self.sigma_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PCAScratch:
    """Analyse en Composantes Principales from scratch.

    Sigma = (1/m) X^T X   (X déjà centrée-réduite)
    On extrait valeurs propres / vecteurs propres de Sigma, on les trie
    par valeur propre décroissante, puis on projette sur les k premiers
    vecteurs propres.
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        m = X.shape[0]
        # X supposée déjà standardisée (moyenne 0, écart-type 1)
        cov = (1.0 / m) * X.T @ X

        eigvals, eigvecs = np.linalg.eigh(cov)  # cov symétrique -> eigh (stable, trié croissant)
        order = np.argsort(eigvals)[::-1]       # tri décroissant
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.explained_variance_ = eigvals
        self.explained_variance_ratio_ = eigvals / eigvals.sum()
        self.components_ = eigvecs[:, : self.n_components]  # (n_features, k)
        return self

    def transform(self, X):
        return X @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def sigmoid(z):
    """σ(z) = 1 / (1 + e^-z), version stable numériquement."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionRidgeScratch:
    """Régression logistique + régularisation Ridge (L2), entraînée par
    descente de gradient. Tout est vectorisé (aucune boucle for sur les
    échantillons ou les features).

    Convention : theta[0] = biais (non régularisé), theta[1:] = poids.
    """

    def __init__(self, lr=0.1, n_iter=2000, lambda_=1.0, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.verbose = verbose

    @staticmethod
    def _add_bias(X):
        m = X.shape[0]
        return np.hstack([np.ones((m, 1)), X])

    def _cost(self, X_b, y, theta):
        m = X_b.shape[0]
        z = X_b @ theta
        y_hat = sigmoid(z)
        eps = 1e-12  # évite log(0)
        data_loss = -(1.0 / m) * np.sum(
            y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
        )
        reg = (self.lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)  # pas de régul. sur le biais
        return data_loss + reg

    def _gradient(self, X_b, y, theta):
        m = X_b.shape[0]
        y_hat = sigmoid(X_b @ theta)
        grad = (1.0 / m) * (X_b.T @ (y_hat - y))
        reg_grad = (self.lambda_ / m) * theta
        reg_grad[0] = 0.0  # le biais n'est jamais régularisé
        return grad + reg_grad

    def fit(self, X, y):
        X_b = self._add_bias(X)
        m, n = X_b.shape
        theta = np.zeros(n)
        self.cost_history_ = np.zeros(self.n_iter)

        for i in range(self.n_iter):
            grad = self._gradient(X_b, y, theta)
            theta = theta - self.lr * grad
            self.cost_history_[i] = self._cost(X_b, y, theta)
            if self.verbose and i % (self.n_iter // 10 or 1) == 0:
                print(f"iter {i:5d} | J(theta) = {self.cost_history_[i]:.5f}")

        self.theta_ = theta
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict_proba(self, X):
        X_b = self._add_bias(X)
        return sigmoid(X_b @ self.theta_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
