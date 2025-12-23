from sentence_transformers import SentenceTransformer
import numpy as np
import re
from typing import List

_model = SentenceTransformer('all-MiniLM-L6-v2')

def _simple_terms(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]

def extract(text: str, top_k: int = 10) -> List[str]:
    terms = _simple_terms(text)
    if not terms:
        return []
    
    seen = []
    for t in terms:
        if t not in seen:
            seen.append(t)

    terms = seen
    doc_emb = _model.encode([text], convert_to_numpy=True)[0]
    term_emb = _model.encode(terms, convert_to_numpy=True)
    scores = term_emb @ doc_emb
    idx = np.argsort(scores)[::-1][:top_k]
    return [terms[i] for i in idx]
