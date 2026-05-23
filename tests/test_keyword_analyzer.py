import pandas as pd
from src.analyzers.keyword_analyzer import KeywordAnalyzer

def test_keyword_analyzer_basic():
    df = pd.DataFrame({
        "text_clean": [
            "neural network model for tire wear prediction",
            "tire wear forecasting using neural networks",
            "spectral data generation using deep learning"
        ]
    })

    analyzer = KeywordAnalyzer()
    results = analyzer.analyze(df)

    assert "keywords_global" in results
    assert len(results["keywords_global"]) > 0
    assert results["tfidf_matrix"].shape[0] == 3
