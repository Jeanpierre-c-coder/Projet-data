from typing import Dict, Any
import pandas as pd

from src.pipelines.data_loader import DataLoader
from src.core.analyzer_factory import AnalyzerFactory


class DataMiningPipeline:

    def __init__(
        self,
        raw_path: str,
        processed_path: str,
        label_column: str = "Document Type"
    ):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.label_column = label_column

        self.data: pd.DataFrame = None
        self.tfidf_results: Dict[str, Any] = None
        self.cluster_results: Dict[str, Any] = None
        self.classification_results: Dict[str, Any] = None

    # 1. Prétraitement
    def run_preprocessing(self) -> pd.DataFrame:
        loader = DataLoader(self.raw_path, self.processed_path)
        self.data = loader.run()
        return self.data
    # 2. TF-IDF

    def run_keywords(self) -> Dict[str, Any]:
        analyzer = AnalyzerFactory.create("keywords")
        self.tfidf_results = analyzer.analyze(self.data)
        self.tfidf_results["df"] = self.data
        return self.tfidf_results

    # 3. Clustering

    def run_clustering(self) -> Dict[str, Any]:
        analyzer = AnalyzerFactory.create("clustering")
        self.cluster_results = analyzer.analyze(self.tfidf_results)
        return self.cluster_results

   
    # 4. Classification

    def run_classification(self) -> Dict[str, Any]:
        analyzer = AnalyzerFactory.create("classification")
        self.classification_results = analyzer.analyze(self.tfidf_results)
        return self.classification_results


    # 5. Pipeline complet
   
    def run_all(self) -> Dict[str, Any]:
        self.run_preprocessing()
        self.run_keywords()
        self.run_clustering()
        self.run_classification()

        return {
            "data": self.data,
            "tfidf": self.tfidf_results,
            "clustering": self.cluster_results,
            "classification": self.classification_results
        }
