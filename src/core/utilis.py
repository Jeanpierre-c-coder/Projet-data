import re
import unicodedata
from typing import Optional

def clean_text(text: Optional[str]) -> str:
   
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9\s]", " ", tex
    text = re.sub(r"\s+", " ", text).strip()

    return text
