from .tfidf_kw import extract as tfidf_extract
from .textrank_kw import extract as textrank_extract
from .keybert_kw import extract as keybert_extract

METHODS = {
    "tfidf": tfidf_extract,
    "textrank": textrank_extract,
    "keybert": keybert_extract
}