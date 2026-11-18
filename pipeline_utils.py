"""Transformadores propios del pipeline (deben poder importarse al cargar el .joblib)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Variables donde un 0 no es fisiológicamente válido -> se tratan como faltantes
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Reemplaza ceros inválidos por NaN. Se aplica dentro del pipeline, por lo que
    la imputación posterior se ajusta solo con datos de entrenamiento (sin fuga)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cols = self.columns if self.columns is not None else ZERO_INVALID_COLS
        for c in cols:
            if c in X.columns:
                X[c] = X[c].replace(0, np.nan)
        return X
