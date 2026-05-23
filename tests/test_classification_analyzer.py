import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analyzers.classification_analyzer import ClassificationAnalyzer


def test_classification_analyzer_basic(tmp_path):
    df = pd.DataFrame({
        "text_clean": [
            "neural network tire wear prediction",
            "deep learning tire defect detection",
            "spectral data generation",
            "chemical spectral analysis"
        ],
        "Document Type": ["Patent", "Patent", "Application", "Application"]
    })

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df["text_clean"])

    data = {
        "tfidf_matrix": tfidf,
        "df": df
    }

    model_path = tmp_path / "model.joblib"
    analyzer = ClassificationAnalyzer(model_path=str(model_path))

    results = analyzer.analyze(data)

    assert "metrics" in results
    assert model_path.exists()
