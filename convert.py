import os
import sys
import json
import re
import time
import math
import hashlib
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Set

import pymysql
from dotenv import load_dotenv
import tiktoken
import chromadb
import openai

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# ------------------ Load Environment ------------------
load_dotenv()

# ------------------ Environment Config ----------------
OPENAI_API_KEY = os.environ.get("")
# if not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env or export it.")
openai.api_key = OPENAI_API_KEY

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "#")
MYSQL_DB = os.environ.get("MYSQL_DB", "dash_db")

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "applicant_chunks_embedded")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./data")
WRITE_JSONL = os.environ.get("WRITE_JSONL", "1") not in {"0", "false", "False"}
WRITE_SQLITE = os.environ.get("WRITE_SQLITE", "0") not in {"0", "false", "False"}

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
MAX_CHUNKS_PER_APPLICANT = int(os.environ.get("MAX_CHUNKS_PER_APPLICANT", "300"))
MAX_TOKENS_PER_APPLICANT = int(os.environ.get("MAX_TOKENS_PER_APPLICANT", "30000"))
SECTION_HARD_TRUNCATE_TOKENS = int(os.environ.get("SECTION_HARD_TRUNCATE_TOKENS", "8000"))

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
DRY_RUN = os.environ.get("DRY_RUN", "0") in {"1", "true", "True"}
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "1") in {"1", "true", "True"}
MAX_APPLICANTS = int(os.environ.get("MAX_APPLICANTS", "0"))

MODEL_PRICING_PER_1K = {
    "text-embedding-3-large": 0.00013,
    "text-embedding-3-small": 0.00002
}
PRICE_PER_1K = MODEL_PRICING_PER_1K.get(EMBED_MODEL)

# ------------------ Tokenizer / Splitter ----------------
splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
ENCODER = tiktoken.get_encoding("cl100k_base")
Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

# ------------------ Utility Functions ------------------
def connect_mysql():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

def fetch_applicants(limit=None):
    conn = connect_mysql()
    cur = conn.cursor()
    sql = "SELECT * FROM applicants"
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()

def truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    half = max_tokens // 2
    head = ENCODER.decode(tokens[:half])
    tail = ENCODER.decode(tokens[-half:])
    return f"[TRUNCATED]\nHEAD:\n{head}\n...\nTAIL:\n{tail}"

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def parse_currency(value):
    if not value:
        return 0.0
    try:
        v = str(value).upper().replace("$", "").replace(",", "").replace("+", "").strip()
        if "K" in v: return float(v.replace("K", "")) * 1000
        if "M" in v: return float(v.replace("M", "")) * 1_000_000
        return float(v)
    except:
        return 0.0

def parse_number(value):
    if not value:
        return 0.0
    try:
        return float(str(value).replace(",", "").strip())
    except:
        return 0.0

def estimate_years_from_hours(hours_str):
    hours = parse_number(hours_str)
    return hours / 2000.0 if hours else 0.0

def extract_country_code(location: str):
    if not location:
        return ""
    parts = [p.strip() for p in location.split(",") if p.strip()]
    return parts[-1] if parts else ""

def normalize_skills(raw):
    if not raw:
        return []
    skills = []
    try:
        if isinstance(raw, str):
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, list):
                skills = parsed
        elif isinstance(raw, list):
            skills = raw
    except:
        if isinstance(raw, str):
            skills = [s.strip() for s in raw.split(",") if s.strip()]
    return [s.strip().lower() for s in skills if isinstance(s, str) and s.strip()]

def isoformat(dt):
    if dt and hasattr(dt, "isoformat"):
        return dt.isoformat()
    return datetime.utcnow().isoformat()

def serialize_metadata(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict, tuple)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out

# ------------------ Chunk Builder ------------------
def build_chunks(applicant: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    seen_hashes: Set[str] = set()
    total_tokens = 0
    chunk_counter = 0
    normalized_skills = normalize_skills(applicant.get("skills"))

    def add_section(raw_text: str, section_type: str, primary_title=None, src_field=None):
        nonlocal total_tokens, chunk_counter
        if not raw_text or chunk_counter >= MAX_CHUNKS_PER_APPLICANT or total_tokens >= MAX_TOKENS_PER_APPLICANT:
            return
        cleaned = clean_text(raw_text)
        truncated = truncate_tokens(cleaned, SECTION_HARD_TRUNCATE_TOKENS)
        token_length = len(ENCODER.encode(truncated))
        candidate_chunks = [truncated] if token_length <= CHUNK_SIZE else splitter.split_text(truncated)

        for piece in candidate_chunks:
            if chunk_counter >= MAX_CHUNKS_PER_APPLICANT or total_tokens >= MAX_TOKENS_PER_APPLICANT:
                break
            piece_hash = sha256_text(piece)
            if piece_hash in seen_hashes:
                continue
            piece_tokens = len(ENCODER.encode(piece))
            if total_tokens + piece_tokens > MAX_TOKENS_PER_APPLICANT:
                break

            chunk_counter += 1
            total_tokens += piece_tokens
            seen_hashes.add(piece_hash)
            chunk_id = f"{applicant['id']}_{section_type}_{chunk_counter}_{piece_hash[:12]}"
            result.append({
                "chunk_id": chunk_id,
                "applicant_id": applicant["id"],
                "chunk_type": section_type,
                "chunk_seq": chunk_counter,
                "primary_title": primary_title or applicant.get("title"),
                "text": piece,
                "tokens": piece_tokens,
                "skills": normalized_skills,
                "job_success_pct": parse_number(applicant.get("job_success")),
                "hourly_rate_num": parse_currency(applicant.get("hourly_rate")),
                "earnings_bucket": "100_plus" if parse_currency(applicant.get("total_earnings")) > 100 else "0_100",
                "experience_years_est": estimate_years_from_hours(applicant.get("total_hours")),
                "location_region": applicant.get("location"),
                "country_code": extract_country_code(applicant.get("location")),
                "scraped_at": isoformat(applicant.get("scraped_at")),
                "content_hash": piece_hash,
                "source_fields": [src_field] if src_field else []
            })

    add_section(applicant.get("overview"), "overview", primary_title=applicant.get("title"), src_field="overview")

    # Work history
    raw_wh = applicant.get("work_history")
    work_items = []
    if raw_wh:
        try:
            obj = raw_wh
            if isinstance(obj, str):
                obj = json.loads(obj)
            if isinstance(obj, list):
                work_items = obj
        except:
            work_items = []
    for entry in work_items:
        if chunk_counter >= MAX_CHUNKS_PER_APPLICANT or total_tokens >= MAX_TOKENS_PER_APPLICANT:
            break
        if isinstance(entry, dict):
            content = entry.get("description") or entry.get("title") or ""
            title = entry.get("title") or "work_item"
        else:
            content = str(entry)
            title = content[:40]
        if content.strip():
            add_section(content, "work_history", primary_title=title, src_field="work_history")

    return result

# ------------------ SQLite (Optional) ------------------
def init_sqlite(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS applicant_chunks (
            chunk_id TEXT PRIMARY KEY,
            applicant_id INTEGER,
            chunk_type TEXT,
            chunk_seq INTEGER,
            primary_title TEXT,
            text TEXT,
            tokens INTEGER,
            skills TEXT,
            job_success_pct REAL,
            hourly_rate_num REAL,
            earnings_bucket TEXT,
            experience_years_est REAL,
            location_region TEXT,
            country_code TEXT,
            scraped_at TEXT,
            content_hash TEXT,
            source_fields TEXT
        )
    """)
    conn.commit()
    return conn

def insert_sqlite(conn, chunk_records: List[Dict[str, Any]]):
    if not chunk_records:
        return
    cur = conn.cursor()
    rows = [
        (
            r["chunk_id"], r["applicant_id"], r["chunk_type"], r["chunk_seq"], r["primary_title"],
            r["text"], r["tokens"], json.dumps(r["skills"], ensure_ascii=False),
            r["job_success_pct"], r["hourly_rate_num"], r["earnings_bucket"], r["experience_years_est"],
            r["location_region"], r["country_code"], r["scraped_at"], r["content_hash"],
            json.dumps(r["source_fields"], ensure_ascii=False)
        )
        for r in chunk_records
    ]
    cur.executemany("""
        INSERT OR REPLACE INTO applicant_chunks
        (chunk_id, applicant_id, chunk_type, chunk_seq, primary_title, text, tokens,
         skills, job_success_pct, hourly_rate_num, earnings_bucket, experience_years_est,
         location_region, country_code, scraped_at, content_hash, source_fields)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()

# ------------------ Chroma ------------------
def init_chroma():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return client, collection

def get_existing_ids(collection):
    if not SKIP_EXISTING:
        return set()
    try:
        res = collection.get()
        return set(res.get("ids", []))
    except Exception:
        return set()

def embed_texts(texts: List[str], max_retries=5, backoff=2.0):
    for attempt in range(max_retries):
        try:
            resp = openai.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff ** attempt
            print(f"Embedding error attempt {attempt+1}/{max_retries}: {e} - retrying in {wait:.1f}s")
            time.sleep(wait)
    return []

def batch(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

# ------------------ Main Pipeline ------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunk_dir = os.path.join(OUTPUT_DIR, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    raw_jsonl_path = os.path.join(chunk_dir, "applicant_chunks_raw.jsonl")
    post_jsonl_path = os.path.join(chunk_dir, "applicant_chunks_embedded.jsonl")

    print("Fetching applicants...")
    applicants = fetch_applicants(limit=MAX_APPLICANTS if MAX_APPLICANTS > 0 else None)
    print(f"Loaded {len(applicants)} applicants")

    sqlite_conn = None
    if WRITE_SQLITE:
        sqlite_conn = init_sqlite(os.path.join(chunk_dir, "applicant_chunks.sqlite"))
        print("SQLite storage enabled.")

    all_chunks: List[Dict[str, Any]] = []
    per_applicant_stats = []
    start_time = time.time()

    for idx, applicant in enumerate(applicants, start=1):
        print(f"\n[{idx}/{len(applicants)}] Applicant ID={applicant['id']} Name={applicant.get('name')}")
        chunks = build_chunks(applicant)
        if not chunks:
            print("  No chunks produced.")
            continue
        all_chunks.extend(chunks)
        per_applicant_stats.append({
            "applicant_id": applicant["id"],
            "name": applicant.get("name"),
            "chunks": len(chunks)
        })
        print(f"  Produced {len(chunks)} chunks (cumulative {len(all_chunks)})")

    total_tokens = sum(c.get("tokens", 0) for c in all_chunks)
    print("\n========== Chunking Summary ==========")
    print(f"Applicants processed: {len(per_applicant_stats)}")
    print(f"Total chunks built : {len(all_chunks)}")
    print(f"Approx total tokens: {total_tokens}")
    if PRICE_PER_1K and not DRY_RUN:
        est_cost = (total_tokens / 1000.0) * PRICE_PER_1K
        print(f"Estimated embedding cost @ {PRICE_PER_1K:.8f}/1K tokens: ~${est_cost:.4f}")
    elif DRY_RUN:
        print("DRY_RUN=1 (no embedding will occur)")
    else:
        print("No pricing entry for this model; cost estimate skipped.")

    if WRITE_JSONL:
        with open(raw_jsonl_path, "w", encoding="utf-8") as f_raw:
            for c in all_chunks:
                f_raw.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Raw chunk JSONL written: {raw_jsonl_path}")

    if WRITE_SQLITE and sqlite_conn and all_chunks:
        insert_sqlite(sqlite_conn, all_chunks)
        print("Chunks inserted into SQLite.")

    if DRY_RUN:
        print("\nDRY_RUN set — stopping before embedding.")
        return

    if not all_chunks:
        print("No chunks to embed. Exiting.")
        return

    chroma_client, collection = init_chroma()
    existing_ids = get_existing_ids(collection)
    print(f"Existing vectors in collection ({COLLECTION_NAME}): {len(existing_ids)}")

    if SKIP_EXISTING and existing_ids:
        before = len(all_chunks)
        all_chunks = [c for c in all_chunks if c["chunk_id"] not in existing_ids]
        print(f"Skipping {before - len(all_chunks)} already embedded chunks.")

    if not all_chunks:
        print("Nothing new to embed after skipping existing. Done.")
        return

    print("\nStarting embedding + ingestion into Chroma...")
    batches = list(batch(all_chunks, EMBED_BATCH_SIZE))
    for bi, bch in enumerate(batches, start=1):
        texts = [r["text"] for r in bch]
        ids = [r["chunk_id"] for r in bch]
        metadatas = [serialize_metadata({k: v for k, v in r.items() if k != "text"}) for r in bch]
        embeddings = embed_texts(texts)
        if len(embeddings) != len(texts):
            print(f"[WARN] Embedding count mismatch batch {bi}, skipping.")
            continue
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        print(f"  Embedded batch {bi}/{len(batches)} ({len(bch)} chunks)")

    # DO NOT call chroma_client.persist() for PersistentClient (automatic persistence)
    if hasattr(chroma_client, "persist"):
        # Only triggers in legacy Client(Settings(...)) usage
        try:
            chroma_client.persist()
            print("Manual persist() called (legacy client).")
        except Exception as e:
            print("persist() existed but failed:", e)
    else:
        print("Automatic persistence (PersistentClient) – no manual persist call needed.")

    if WRITE_JSONL:
        with open(post_jsonl_path, "w", encoding="utf-8") as f_post:
            for c in all_chunks:
                f_post.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Post-embedding JSONL written: {post_jsonl_path}")

    elapsed = time.time() - start_time
    print(f"\nEmbedding complete. Added {len(all_chunks)} chunks this run.")
    print(f"Chroma path: {CHROMA_PATH}")
    print("Top 5 applicants by chunk count:")
    for row in sorted(per_applicant_stats, key=lambda r: r['chunks'], reverse=True)[:5]:
        print(f"  ID={row['applicant_id']} Name={row['name']}: {row['chunks']} chunks")
    print(f"\nTotal elapsed time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()