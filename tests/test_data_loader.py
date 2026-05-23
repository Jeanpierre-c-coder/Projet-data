from src.pipelines.data_loader import DataLoader
import pandas as pd
from pathlib import Path


def test_data_loader_basic(tmp_path):
    # Création d'un CSV temporaire
    csv_path = tmp_path / "test.csv"
    out_path = tmp_path / "processed.csv"

    df = pd.DataFrame({
        "Title": ["Test Title"],
        "Abstract": ["Some abstract text."]
    })
    df.to_csv(csv_path, index=False)

    loader = DataLoader(str(csv_path), str(out_path))
    df_clean = loader.run()

    assert "text_clean" in df_clean.columns
    assert len(df_clean) == 1
    assert out_path.exists()
