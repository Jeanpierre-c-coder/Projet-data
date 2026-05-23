import re
import unicodedata
from typing import Optional

def clean_text(text: Optional[str]) -> str:
  
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Supprimer accents
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

    # Supprimer tout sauf lettres/chiffres/espace
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normaliser espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
