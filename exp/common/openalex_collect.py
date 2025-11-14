import requests, time
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Any

from .utils import norm_space, doc_text, slug_list, approx_token_count, hash_id

BASE_URL = "https://api.openalex.org/works"

def _reconstruct_abstract(inv: Dict[str, List[int]]) -> str:
    if not isinstance(inv, Dict) or not inv:
        return ""
    
    max_pos = 0
    for positions in inv.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    
    tokens = [""] * (max_pos + 1)
    for token, positions in inv.items():
        for pos in positions:
            if 0 <= pos < len(tokens):
                tokens[pos] = token
    
    txt = " ".join(t for t in tokens if t)
    return norm_space(txt)

def fetch_openalex(
    year_from: int = 2010, year_to: int = 2025, per_page: int = 200, 
    max_per_year: int = 5000, sleep_sec: float = 1.0
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for year in tqdm(range(year_from, year_to + 1), desc="Collecting OpenAlex..."):
        page = 1
        collected = 0

        while True:
            filters = [f"publication_year:{year}"]
            filter_str = ",".join(filters)

            params = {
                "filter": filter_str,
                "per-page": per_page,
                "page": page
            }

            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for w in results:
                wid = w.get("id") or ""
                short_id = wid.split("/")[-1] if wid else ""

                title = w.get("title") or ""
                title = norm_space(title)

                # abstract
                inv = w.get("abstract_inverted_index")
                abstract = _reconstruct_abstract(inv)

                pub_year = w.get("publication_year") or year

                # authors
                authorships = w.get("authorships") or []
                authors = [a["author"]["display_name"]
                           for a in authorships
                           if a.get("author") and a["author"].get("display_name")]

                # keywords_gold
                kws = w.get("keywords") or []
                keywords_gold = [k.get("display_name", "").strip()
                                 for k in kws
                                 if k.get("display_name")]

                rows.append({
                    "id": short_id,
                    "title": title,
                    "abstract": abstract,
                    "year": pub_year,
                    "authors": authors,
                    "keywords_gold": keywords_gold,
                })

                collected += 1
                if max_per_year is not None and collected >= max_per_year:
                    break

            if max_per_year is not None and collected >= max_per_year:
                break

            meta = data.get("meta", {})
            count = meta.get("count", 0)
            if page * per_page >= count:
                break

            page += 1
            time.sleep(sleep_sec)
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    df["id"] = df["id"].astype(str)
    df["title"] = df["title"].map(norm_space)
    df["abstract"] = df["abstract"].map(norm_space)
    df["authors"] = df["authors"].map(slug_list)

    df["doc_text"] = df.apply(
        lambda r: doc_text(r["title"], r["abstract"]),
        axis=1
    )
    df["n_tokens"] = df["doc_text"].map(approx_token_count)
    df["uid"] = df["id"].map(hash_id)
    df.drop_duplicates(subset=["id"], keep="first", inplace=True)
    return df