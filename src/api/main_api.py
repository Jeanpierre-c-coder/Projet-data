from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from src.pipelines.data_mining_pipeline import DataMiningPipeline
from src.core.utils import clean_text

app = FastAPI(
    title="Veille Technologique - API",
    description="API pour analyse, clustering et classification de brevets",
    version="1.0.0"
)

# Charger modèle et vectorizer
MODEL_PATH = "models/classifier.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except:
    model = None
    vectorizer = None

# Schémas d'entrée
class PredictRequest(BaseModel):
    text: str


# Endpoint /analyze

@app.post("/analyze")
def analyze():
    pipeline = DataMiningPipeline(
        raw_path="data/raw/lens-export.csv",
        processed_path="data/processed/clean.csv",
        label_column="Document Type"
    )

    results = pipeline.run_all()

    return {
        "clustering": {
            "best_k": results["clustering"]["best_k"],
            "silhouette_scores": results["clustering"]["silhouette_scores"],
            "cluster_keywords": results["clustering"]["cluster_keywords"]
        },
        "classification": results["classification"]["metrics"],
        "keywords_global": results["tfidf"]["keywords_global"][:20]
    }

# Endpoint /predict

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or vectorizer is None:
        return {"error": "Modèle non entraîné. Lancez /analyze d'abord."}

    text_clean = clean_text(req.text)
    X = vectorizer.transform([text_clean])
    pred = model.predict(X)[0]

    return {
        "input": req.text,
        "cleaned": text_clean,
        "prediction": pred
    }
