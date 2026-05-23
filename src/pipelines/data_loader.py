import pandas as pd
import re
from pathlib import Path
from typing import Optional
from src.core.utils import clean_text
class DataLoader:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.input_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_path, encoding="utf-8", low_memory=False)
        return df

    def prepare_text_column(self, df: pd.DataFrame) -> pd.DataFrame:

        df["Title"] = df["Title"].fillna("")
        df["Abstract"] = df["Abstract"].fillna("")
        df["text_raw"] = df["Title"] + " " + df["Abstract"]
        df["text_clean"] = df["text_raw"].apply(clean_text)

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["Title", "Abstract"])
        return df

    def save(self, df: pd.DataFrame):
        df.to_csv(self.output_path, index=False, encoding="utf-8")

    def run(self) -> pd.DataFrame:
        df = self.load()
        df = self.remove_duplicates(df)
        df = self.prepare_text_column(df)
        self.save(df)
        return df
