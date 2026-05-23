from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAnalyzer(ABC):
    """
    Classe abstraite pour tous les analyseurs.
    Chaque analyseur doit implémenter la méthode analyze().
    """

    @abstractmethod
    def analyze(self, data: Any) -> Dict:
        """Analyse les données et retourne un dictionnaire de résultats."""
        pass
