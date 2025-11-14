from typing import Iterable, List, Dict
import re, hashlib, os, json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def slug_list(xs: Iterable[str]) -> List[str]:
    out = []
    for x in xs or []:
        if not x: continue

        x = str(x).strip()
        if x: out.append(x)

    return out

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def doc_text(title: str, abstract: str) -> str:
    title = norm_space(title)
    abstract = norm_space(abstract)
    return (title + "\n\n" + abstract).strip()

def approx_token_count(s: str) -> int:
    # xấp xỉ số token theo khoảng trống
    return len(re.findall(r"\S+", s or ""))

def hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_checkpoint(path: str) -> Dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except:
            return {}
    return {}

def save_checkpoint(path: str, data: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_partitioned_by_year(df: pd.DataFrame, out_base: str):
    os.makedirs(out_base, exist_ok=True)
    
    for y, g in df.groupby("year"):
        if pd.isna(y): continue

        out = os.path.join(out_base, f"year={int(y)}")
        os.makedirs(out, exist_ok=True)
        table = pa.Table.from_pandas(g, preserve_index=False)
        pq.write_table(table, os.path.join(out, f"part-{hash_id(str(y))[:8]}.parquet"))
