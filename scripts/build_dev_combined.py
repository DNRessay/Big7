import os
import re
import json
import hashlib
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

api = HfApi()

OUT_DIR = "dev_combined"
PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")
os.makedirs(OUT_DIR, exist_ok=True)

REPO_OWNER = "michsethowusu"

SOURCES = {
    "zu": [
        ("english-zulu_sentence-pairs", "sentence-pairs", "sp"),
        ("english-zulu_sentence-pairs_mt560", "mt560", "mt5"),
    ],
    "xh": [
        ("english-xhosa_sentence-pairs", "sentence-pairs", "sp"),
        ("english-xhosa_sentence-pairs_mt560", "mt560", "mt5"),
    ],
    "tn": [
        ("english-tswana_sentence-pairs", "sentence-pairs", "sp"),
        ("english-setswana_sentence-pairs_mt560", "mt560", "mt5"),
    ],
    "nso": [
        ("english-pedi_sentence-pairs", "sentence-pairs", "sp"),
        ("english-northern-sotho_sentence-pairs_mt560", "mt560", "mt5"),
    ],
    "st": [],
    "ss": [
        ("english-swati_sentence-pairs", "sentence-pairs", "sp"),
        ("english-swati_sentence-pairs_mt560", "mt560", "mt5"),
    ],
    "ts": [
        ("english-tsonga_sentence-pairs", "sentence-pairs", "sp"),
        ("english-tsonga_sentence-pairs_mt560", "mt560", "mt5"),
    ],
    "ve": [
        ("english-venda_sentence-pairs_mt560", "mt560", "mt5"),
    ],
    "nr": [],
}

COLUMN_GUESSES = [
    ("source", "target"),
    ("src", "tgt"),
    ("text", "en_text"),
    ("sentence", "english"),
]
NON_TEXT_COLS = {"similarity", "score", "id", "index"}
ENGLISH_NAMES = {"english", "eng", "en"}


def find_parquet_files(repo_id):
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]
    train_files = [f for f in parquet_files if "train" in f.lower()]
    return sorted(train_files or parquet_files)


def detect_columns(df):
    cols = list(df.columns)
    for src_c, en_c in COLUMN_GUESSES:
        if src_c in cols and en_c in cols:
            return src_c, en_c
    if "conversations" in cols:
        return "conversations", None

    en_col = next((c for c in cols if c.lower() in ENGLISH_NAMES), None)
    str_cols = [c for c in cols if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]

    if en_col:
        others = [c for c in str_cols if c != en_col and c.lower() not in NON_TEXT_COLS]
        if others:
            return others[0], en_col

    str_cols = [c for c in str_cols if c.lower() not in NON_TEXT_COLS]
    if len(str_cols) >= 2:
        return str_cols[0], str_cols[1]
    raise ValueError(f"Could not detect text columns in: {cols}")


def row_hash(text, en_text):
    return hashlib.md5(f"{text.lower()}|{en_text.lower()}".encode("utf-8")).hexdigest()


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed_shards": {}}


def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)


def load_existing_state(lang_code):
    out_path = os.path.join(OUT_DIR, f"{lang_code}-en.jsonl")
    seen_hashes = set()
    pid = 0
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                seen_hashes.add(row_hash(row["text"], row["en_text"]))
                pid += 1
    return out_path, seen_hashes, pid


def clean_and_stream(df, src_col, en_col, lang_code, source_type, prefix, seen_hashes, pid, fh):
    local_idx = 0
    for text, en_text in zip(df[src_col].astype(str), df[en_col].astype(str)):
        text = text.strip()
        en_text = en_text.strip()
        local_idx += 1
        if not text or not en_text:
            continue
        if text.lower() == en_text.lower():
            continue
        if not re.search(r"[A-Za-z]", text):
            continue
        h = row_hash(text, en_text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        row = {
            "lang": lang_code,
            "text": text,
            "en_text": en_text,
            "doc_id": f"{prefix}_{local_idx}",
            "pid": pid,
            "source": source_type,
        }
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        pid += 1
    return pid


def build_language(lang_code, entries, progress):
    if not entries:
        return "no public translation sources available"

    out_path, seen_hashes, pid = load_existing_state(lang_code)
    completed = set(progress["completed_shards"].get(lang_code, []))

    with open(out_path, "a", encoding="utf-8") as fh:
        for slug, source_type, prefix in entries:
            repo_id = f"{REPO_OWNER}/{slug}"
            try:
                files = find_parquet_files(repo_id)
            except Exception as e:
                print(f"  ! could not list {slug}: {e}")
                continue

            for f in files:
                shard_key = f"{slug}::{f}"
                if shard_key in completed:
                    continue
                try:
                    path = hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset")
                    df = pd.read_parquet(path)
                    src_col, en_col = detect_columns(df)
                    if en_col is None:
                        print(f"  ? {shard_key} uses '{src_col}' format (chat-style) - skipping")
                        completed.add(shard_key)
                        progress["completed_shards"][lang_code] = list(completed)
                        save_progress(progress)
                        del df
                        continue
                    pid = clean_and_stream(
                        df, src_col, en_col, lang_code, source_type, prefix, seen_hashes, pid, fh
                    )
                    del df
                    completed.add(shard_key)
                    progress["completed_shards"][lang_code] = list(completed)
                    save_progress(progress)
                    print(f"  done: {shard_key} (running total {pid:,})")
                except Exception as e:
                    print(f"  ! skip {shard_key}: {e}")
                    continue

    return f"{pid:,} rows in {out_path}"


def main():
    progress = load_progress()
    summary = {}
    for lang_code, entries in SOURCES.items():
        print(f"\n{lang_code}")
        msg = build_language(lang_code, entries, progress)
        print(f"  -> {msg}")
        summary[lang_code] = msg

    print("\n" + "=" * 30)
    print("SUMMARY")
    print("=" * 30)
    for lang, msg in summary.items():
        print(f"{lang:5} {msg}")


if __name__ == "__main__":
    main()
