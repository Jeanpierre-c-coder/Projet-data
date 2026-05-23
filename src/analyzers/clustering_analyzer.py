from typing import Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.core.base_analyzer import BaseAnalyzer
from src.core.analyzer_factory import AnalyzerFactory


class ClusteringAnalyzer(BaseAnalyzer):
 
    def __init__(self, k_min: int = 3, k_max: int = 10):
        self.k_min = k_min
        self.k_max = k_max

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:

        tfidf_matrix = data["tfidf_matrix"]
        feature_names = data["feature_names"]

        silhouette_scores = {}
        best_k = None
        best_score = -1
        best_labels = None

        # 1. Tester plusieurs valeurs de k
        for k in range(self.k_min, self.k_max + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(tfidf_matrix)

            score = silhouette_score(tfidf_matrix, labels)
            silhouette_scores[k] = score

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        # 2. Extraire les mots-clés dominants par cluster
        cluster_keywords = self._extract_cluster_keywords(
            tfidf_matrix, best_labels, feature_names
        )

        return {
            "best_k": best_k,
            "silhouette_scores": silhouette_scores,
            "labels": best_labels,
            "cluster_keywords": cluster_keywords
        }

    def _extract_cluster_keywords(self, tfidf_matrix, labels, feature_names):

        clusters = {}
        num_clusters = len(set(labels))

        for k in range(num_clusters):
            idx = np.where(labels == k)[0]
            submatrix = tfidf_matrix[idx].mean(axis=0).A1

            top_idx = submatrix.argsort()[-10:][::-1]
            keywords = [(feature_names[i], submatrix[i]) for i in top_idx]

            clusters[k] = keywords

        return clusters


# Enregistrement dans la factory
AnalyzerFactory.register("clustering", ClusteringAnalyzer)
