from typing import Type, Dict
from .base_analyzer import BaseAnalyzer

class AnalyzerFactory:
    """
    Fabrique permettant d'instancier dynamiquement un analyseur
    en fonction d'un nom fourni.
    """

    registry: Dict[str, Type[BaseAnalyzer]] = {}

    @classmethod
    def register(cls, name: str, analyzer_cls: Type[BaseAnalyzer]):
        cls.registry[name] = analyzer_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseAnalyzer:
        if name not in cls.registry:
            raise ValueError(f"Analyseur inconnu : {name}")
        return cls.registry[name](**kwargs)
