import streamlit as st
import pandas as pd
import joblib

from src.pipelines.data_mining_pipeline import DataMiningPipeline
from src.core.utils import clean_text


st.set_page_config(
    page_title="Plateforme de Veille Technologique",
    layout="wide"
)

st.title("🔍 Plateforme de Veille Technologique")
st.write("Analyse, clustering et classification de brevets Lens.org")


# ---------------------------------------------------------
# 1. Bouton pour lancer le pipeline complet
# ---------------------------------------------------------
st.header("⚙️ Exécuter le pipeline complet")

if st.button("Lancer l'analyse"):
    pipeline = DataMiningPipeline(
        raw_path="data/raw/lens-export.csv",
        processed_path="data/processed/clean.csv",
        label_column="Document Type"
    )

    results = pipeline.run_all()

    st.success("Pipeline exécuté avec succès !")

    # Stocker les résultats en session
    st.session_state["results"] = results


# ---------------------------------------------------------
# 2. Affichage des résultats
# ---------------------------------------------------------
if "results" in st.session_state:
    results = st.session_state["results"]

    # -----------------------------
    # Analyse descriptive
    # -----------------------------
    st.header("📊 Analyse descriptive")

    df = results["data"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de documents", len(df))
    col2.metric("Années uniques", df["Publication Year"].nunique())
    col3.metric("Types de documents", df["Document Type"].nunique())

    st.subheader("Répartition par pays")
    st.bar_chart(df["Jurisdiction"].value_counts())

    st.subheader("Répartition par année")
    st.line_chart(df["Publication Year"].value_counts().sort_index())

    # -----------------------------
    # Mots-clés globaux
    # -----------------------------
    st.header("🧠 Mots-clés dominants (TF‑IDF)")

    keywords = results["tfidf"]["keywords_global"]
    kw_df = pd.DataFrame(keywords, columns=["mot", "score"])

    st.bar_chart(kw_df.set_index("mot"))

    # -----------------------------
    # Clustering
    # -----------------------------
    st.header("🧩 Clustering")

    st.write(f"**Meilleur k :** {results['clustering']['best_k']}")

    for k, words in results["clustering"]["cluster_keywords"].items():
        st.subheader(f"Cluster {k}")
        st.write([w for w, _ in words])

    # -----------------------------
    # Classification
    # -----------------------------
    st.header("🎯 Classification supervisée")

    metrics = results["classification"]["metrics"]
    st.json(metrics)

    # -----------------------------
    # Prédiction en direct
    # -----------------------------
    st.header("🔮 Prédiction en direct")

    model = joblib.load("models/classifier.joblib")
    vectorizer = joblib.load("models/vectorizer.joblib")

    user_input = st.text_area("Entrez un texte (titre + résumé)")

    if st.button("Prédire"):
        text_clean = clean_text(user_input)
        X = vectorizer.transform([text_clean])
        pred = model.predict(X)[0]

        st.success(f"Type de document prédit : **{pred}**")
