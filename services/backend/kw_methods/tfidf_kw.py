from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import re

STOPWORDS = set([
    "we","our","paper","show","result","results","approach", "experiments", "also",
    "method","methods","propose","introduction","related","work", "systems",
    "section","figure","table","study","based","using","model",
    "data","analysis","problem","task","algorithm","performance"
])

def _preprocess(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _tokenize(text: str) -> List[str]:
    text = _preprocess(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

def extract(text: str, top_k: int = 10) -> List[str]:
    if not text or not text.strip():
        return []
    
    tokens = _tokenize(text)
    if not tokens:
        return []
    
    doc = " ".join(tokens)

    vec = TfidfVectorizer(
        lowercase=False,
        tokenizer=lambda s: s.split(),
        token_pattern=None
    )

    try:
        X = vec.fit_transform([doc])
    except ValueError:
        return []
    
    feature_names = vec.get_feature_names_out()
    scores = X.toarray()[0]

    ids = [i for i, s in enumerate(scores) if s > 0]
    if not ids:
        return []
    
    ids = sorted(ids, key=lambda i: scores[i], reverse=True)[:top_k]
    keywords = [feature_names[i] for i in ids]
    return keywords
