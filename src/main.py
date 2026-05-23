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
