from typing import Dict, Any
import pandas as pd

from src.core.base_analyzer import BaseAnalyzer
from src.core.analyzer_factory import AnalyzerFactory


class DescriptiveAnalyzer(BaseAnalyzer):

    def __init__(self):
        pass

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        results = {}

        # 1. Nombre total de documents
        results["total_documents"] = len(data)

        # 2. Répartition par pays
        if "Jurisdiction" in data.columns:
            results["by_country"] = (
                data["Jurisdiction"].value_counts().to_dict()
            )

        # 3. Répartition par année
        if "Publication Year" in data.columns:
            results["by_year"] = (
                data["Publication Year"].value_counts().sort_index().to_dict()
            )

        # 4. Répartition par type
        if "Document Type" in data.columns:
            results["by_type"] = (
                data["Document Type"].value_counts().to_dict()
            )

        # 5. Statistiques textuelles
        if "text_clean" in data.columns:
            data["text_len"] = data["text_clean"].apply(lambda x: len(x.split()))
            results["text_stats"] = {
                "min": int(data["text_len"].min()),
                "max": int(data["text_len"].max()),
                "mean": float(data["text_len"].mean()),
            }
        return results

# Enregistrement dans la factory
AnalyzerFactory.register("descriptive", DescriptiveAnalyzer)
