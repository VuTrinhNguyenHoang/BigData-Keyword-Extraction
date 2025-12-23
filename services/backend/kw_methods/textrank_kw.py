import re
import networkx as nx
from typing import List

def _simple_terms(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def extract(text: str, top_k: int = 10) -> List[str]:
    terms = _simple_terms(text)
    if not terms:
        return []
    
    # build co-occurrence
    G = nx.Graph()
    G.add_nodes_from(terms)
    window = 4
    n = len(terms)
    for i in range(n):
        for j in range(i+1, min(i+window, n)):
            if terms[i] == terms[j]:
                continue
            if G.has_edge(terms[i], terms[j]):
                G[terms[i]][terms[j]]["weight"] += 1.0
            else:
                G.add_edge(terms[i], terms[j], weight=1.0)
    if G.number_of_edges() == 0:
        return terms[:top_k]
    scores = nx.pagerank(G, weight="weight")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [t for t,_ in ranked]
