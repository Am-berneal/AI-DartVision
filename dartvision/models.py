"""
The five ML classifiers plus the deterministic geometry baseline.

Every ML model is wrapped in a Pipeline with the same StandardScaler so that
all models see identical, consistently-preprocessed inputs.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .board import LABELS, score_region

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM"]


def get_models(seed: int = 42) -> dict[str, Pipeline]:
    clfs = {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=10.0),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "SVM": SVC(kernel="rbf", C=50.0, gamma="scale", random_state=seed),
    }
    return {name: Pipeline([("scale", StandardScaler()), ("clf", clf)]) for name, clf in clfs.items()}


class GeometryBaseline(BaseEstimator, ClassifierMixin):
    """Applies the board rules directly. 'fit' does nothing - there is nothing to learn.

    Expects a DataFrame (or array) whose first two columns are x and y.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array(LABELS)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return score_region(X[:, 0], X[:, 1])