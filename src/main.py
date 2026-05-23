from src.pipelines.data_loader import DataLoader
from src.core.analyzer_factory import AnalyzerFactory

def main():
    loader = DataLoader(
        input_path="data/raw/lens-export(1).csv",
        output_path="data/processed/clean.csv"
    )
    df = loader.run()

    analyzer = AnalyzerFactory.create("descriptive")
    results = analyzer.analyze(df)

    print("Résultats descriptifs")
    for key, value in results.items():
        print(f"{key} : {value}")

if __name__ == "__main__":
    main()

# azs
from src.pipelines.data_loader import DataLoader
from src.core.analyzer_factory import AnalyzerFactory

def main():
    loader = DataLoader(
        input_path="data/raw/lens-export(1).csv",
        output_path="data/processed/clean.csv"
    )
    df = loader.run()

    # Étape 3 : TF-IDF
    keyword_analyzer = AnalyzerFactory.create("keywords")
    tfidf_results = keyword_analyzer.analyze(df)

    # Étape 4 : Clustering
    clustering_analyzer = AnalyzerFactory.create("clustering")
    cluster_results = clustering_analyzer.analyze(tfidf_results)

    print("Meilleur k ", cluster_results["best_k"])
    print("Mots-clés par cluster ")
    for k, words in cluster_results["cluster_keywords"].items():
        print(f"Cluster {k} :", [w for w, _ in words])

if __name__ == "__main__":
    main()

# bz
from src.pipelines.data_mining_pipeline 
import DataMiningPipeline

def main():
    pipeline = DataMiningPipeline(
        raw_path="data/raw/lens-export(1).csv",
        processed_path="data/processed/clean.csv",
        label_column="Document Type"
    )

    results = pipeline.run_all()

    print(" Pipeline terminé ")
    print("Meilleur k :", results["clustering"]["best_k"])
    print("Métriques classification :", results["classification"]["metrics"])

if __name__ == "__main__":
    main()
