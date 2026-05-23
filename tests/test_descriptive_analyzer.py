import pandas as pd
from src.analyzers.descriptive_analyzer import DescriptiveAnalyzer

def test_descriptive_analyzer_basic():
    df = pd.DataFrame({
        "Jurisdiction": ["US", "EP", "US"],
        "Publication Year": [2020, 2021, 2020],
        "Document Type": ["Patent", "Patent", "Application"],
        "text_clean": ["hello world", "test text", "another test"]
    })

    analyzer = DescriptiveAnalyzer()
    results = analyzer.analyze(df)

    assert results["total_documents"] == 3
    assert results["by_country"]["US"] == 2
    assert results["by_year"][2020] == 2
    assert "text_stats" in results
