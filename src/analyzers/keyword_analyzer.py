from typing import Dict, Any, List, Tuple
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.base_analyzer import BaseAnalyzer
from src.core.analyzer_factory import AnalyzerFactory

"df": data

class KeywordAnalyzer(BaseAnalyzer):
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.85
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        if "text_clean" not in data.columns:
            raise ValueError("La colonne 'text_clean' est manquante.")

        texts = data["text_clean"].tolist()

        # 1. Vectorisation TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # 2. Extraction des mots-clés globaux
        global_scores = tfidf_matrix.sum(axis=0).A1
        keywords_global = sorted(
            zip(feature_names, global_scores),
            key=lambda x: x[1],
            reverse=True
        )[:30]

        # 3. Extraction des mots-clés par document (optionnel)
        keywords_per_doc = []
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray().flatten()
            top_idx = row.argsort()[-5:][::-1]
            keywords = [(feature_names[j], row[j]) for j in top_idx]
            keywords_per_doc.append(keywords)

        return {
            "tfidf_matrix": tfidf_matrix,
            "feature_names": feature_names,
            "keywords_global": keywords_global,
            "keywords_per_doc": keywords_per_doc
        }


# Enregistrement dans la factory
AnalyzerFactory.register("keywords", KeywordAnalyzer)
