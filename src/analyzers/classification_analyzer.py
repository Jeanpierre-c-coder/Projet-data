from typing import Dict, Any
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.core.base_analyzer import BaseAnalyzer
from src.core.analyzer_factory import AnalyzerFactory


class ClassificationAnalyzer(BaseAnalyzer):
    """
    Analyseur de classification supervisée.
    - utilise la matrice TF-IDF
    - entraîne un modèle SVM linéaire
    - calcule les métriques
    - sauvegarde le modèle
    """

    def __init__(self, label_column: str = "Document Type", model_path: str = "models/classifier.joblib"):
        self.label_column = label_column
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:

        df = data["df"]
        X = data["tfidf_matrix"]
        y = df[self.label_column]

        # 1. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 2. Modèle supervisé
        model = LinearSVC()
        model.fit(X_train, y_train)

        # 3. Prédictions
        y_pred = model.predict(X_test)

        # 4. Métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        # 5. Sauvegarde du modèle
        joblib.dump(model, self.model_path)

        return {
            "model": model,
            "metrics": metrics,
            "model_path": str(self.model_path)
        }


# Enregistrement dans la factory
AnalyzerFactory.register("classification", ClassificationAnalyzer)
