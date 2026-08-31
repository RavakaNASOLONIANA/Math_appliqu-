"""
Script de test : compare les poids theta obtenus par notre implémentation
"from scratch" (mllib.py) à ceux obtenus par Scikit-Learn, pour prouver
l'exactitude mathématique de l'implémentation.

Usage : python3 test_comparison.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mllib import StandardScalerScratch, PCAScratch, LogisticRegressionRidgeScratch

RANDOM_STATE = 42
LAMBDA_ = 1.0        # force de régularisation Ridge (notre convention)
LR = 0.5
N_ITER = 3000
TOLERANCE_CORR = 0.999    # seuil de corrélation minimal accepté
TOLERANCE_ACC_DIFF = 0.02  # écart d'accuracy toléré


def main():
    df = pd.read_csv("data.csv")
    df = df.drop(columns=[c for c in df.columns if c == "id" or c.startswith("Unnamed")])
    y = (df["diagnosis"] == "M").astype(int).values
    X = df.drop(columns=["diagnosis"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Standardisation from scratch --------------------------------
    scaler = StandardScalerScratch()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # --- ACP from scratch (sanity check) ------------------------------
    pca = PCAScratch(n_components=2)
    pca.fit(X_train_std)
    var_2d = pca.explained_variance_ratio_[:2].sum()
    print(f"[ACP]     Variance expliquée par les 2 premières composantes : {var_2d*100:.1f}%")
    assert 0 < var_2d <= 1, "La variance expliquée doit être comprise entre 0 et 1."

    # --- Régression logistique from scratch ---------------------------
    model = LogisticRegressionRidgeScratch(lr=LR, n_iter=N_ITER, lambda_=LAMBDA_)
    model.fit(X_train_std, y_train)
    acc_scratch = accuracy_score(y_test, model.predict(X_test_std))

    # --- Régression logistique Scikit-Learn (même régularisation) -----
    # Convention : J_scratch = (1/m)*data_loss + (lambda/2m)*||theta||^2
    #              J_sklearn = C*data_loss + 0.5*||w||^2
    #           => C = 1 / lambda  pour un objectif équivalent (à un
    #              facteur m près, qui ne change pas le minimiseur).
    sk_model = LogisticRegression(C=1.0 / LAMBDA_, max_iter=5000)
    sk_model.fit(X_train_std, y_train)
    acc_sklearn = accuracy_score(y_test, sk_model.predict(X_test_std))

    diff = np.abs(model.coef_ - sk_model.coef_[0])
    corr = np.corrcoef(model.coef_, sk_model.coef_[0])[0, 1]
    intercept_diff = abs(model.intercept_ - sk_model.intercept_[0])

    print("\n[Régression logistique] Comparaison des poids theta")
    print(f"  Accuracy (scratch)              : {acc_scratch:.4f}")
    print(f"  Accuracy (sklearn)               : {acc_sklearn:.4f}")
    print(f"  Écart absolu moyen |Δtheta|       : {diff.mean():.5f}")
    print(f"  Écart absolu max   |Δtheta|       : {diff.max():.5f}")
    print(f"  Écart sur le biais (intercept)    : {intercept_diff:.5f}")
    print(f"  Corrélation theta_scratch/sklearn : {corr:.6f}")

    # --- Assertions de validation --------------------------------------
    assert corr >= TOLERANCE_CORR, (
        f"Corrélation trop faible ({corr:.4f} < {TOLERANCE_CORR}) : "
        "l'implémentation diverge de Scikit-Learn."
    )
    assert abs(acc_scratch - acc_sklearn) <= TOLERANCE_ACC_DIFF, (
        "Écart d'accuracy trop important entre les deux implémentations."
    )

    print("\n TEST RÉUSSI : l'implémentation from scratch est mathématiquement "
          "cohérente avec Scikit-Learn (corrélation et accuracy quasi identiques).")


if __name__ == "__main__":
    main()
