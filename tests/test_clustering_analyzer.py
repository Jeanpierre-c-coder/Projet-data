import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analyzers.clustering_analyzer import ClusteringAnalyzer


def test_clustering_analyzer_basic():
    texts = [
        "neural network tire wear prediction",
        "deep learning tire defect detection",
        "spectral data generation",
        "chemical spectral analysis",
        "rubber product manufacturing control"
    ]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    data = {
        "tfidf_matrix": tfidf,
        "feature_names": vectorizer.get_feature_names_out()
    }

    analyzer = ClusteringAnalyzer(k_min=2, k_max=3)
    results = analyzer.analyze(data)

    assert "best_k" in results
    assert "labels" in results
    assert len(results["labels"]) == len(texts)
