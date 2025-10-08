

import os
import sys
import json
import re
import math
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter

import chromadb
import openai
from dotenv import load_dotenv

# Optional DB enrichment
import pymysql

# Groq client for answer generation
try:
    from groq import Groq
except ImportError:
    Groq = None
    # You must: pip install groq

load_dotenv()

# ------------------ Config ------------------
OPENAI_API_KEY = os.environ.get("")
# if not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY not set.")
# openai.api_key = OPENAI_API_KEY

GROQ_API_KEY = os.environ.get("")
# if not GROQ_API_KEY:
#     print("WARNING: GROQ_API_KEY not set. Answer generation will be skipped.")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "applicant_chunks_embedded")

TOP_K = int(os.environ.get("TOP_K", "12"))
MAX_APPLICANTS_IN_ANSWER = int(os.environ.get("MAX_APPLICANTS_IN_ANSWER", "15"))
MAX_SNIPPETS_PER_APPLICANT = int(os.environ.get("MAX_SNIPPETS_PER_APPLICANT", "3"))
MAX_SNIPPET_CHARS = int(os.environ.get("MAX_SNIPPET_CHARS", "350"))

ENABLE_DB_ENRICH = os.environ.get("ENABLE_DB_ENRICH", "1") in {"1", "true", "True"}

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DB = os.environ.get("MYSQL_DB", "dash_db")

# Domain / keyword maps (expand as needed)
DOMAIN_MAP = {
    "audio": ["audio", "speech", "asr", "voice", "sound"],
    "web": ["web", "frontend", "react", "django", "flask", "javascript", "next.js", "web app", "fullstack"],
    "pcb": ["pcb", "altium", "schematic", "circuit", "electronics", "hardware"],
    "devops": ["devops", "kubernetes", "docker", "terraform", "ansible", "cicd", "cloud"],
    "data_annot": ["annotation", "labeling", "labeling", "data annotation", "image annotation", "video annotation"]
}

# Country normalization (simple heuristic)
COUNTRY_ALIASES = {
    "japan": "Japan",
    "india": "India",
    "bangladesh": "Bangladesh",
    "indonesia": "Indonesia",
    "philippines": "Philippines",
    "pakistan": "Pakistan",
    "morocco": "Morocco",
    "italy": "Italy",
    "portugal": "Portugal"
}

REQUESTED_FIELD_KEYWORDS = {
    "phone": ["phone", "mobile", "contact number"],
    "email": ["email", "mail"],
    "profile": ["profile", "url", "link"],
    "hourly_rate_num": ["hourly", "rate", "price", "cost"],
    "location_region": ["location", "where based", "country"],
    "skills": ["skills", "expertise"],
}

# ------------------ Embedding ------------------
def embed_query(text: str) -> List[float]:
    resp = openai.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

# ------------------ Query Parsing ------------------
def parse_query(query: str) -> Dict[str, Any]:
    q_lower = query.lower()

    # Countries
    countries = []
    for key, proper in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(key)}\b", q_lower):
            countries.append(proper)

    # Domain intents
    domain_intents = []
    for domain, kws in DOMAIN_MAP.items():
        if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in kws):
            domain_intents.append(domain)

    # Requested explicit fields
    requested_fields = set()
    for field, triggers in REQUESTED_FIELD_KEYWORDS.items():
        if any(t in q_lower for t in triggers):
            requested_fields.add(field)

    # Limit if user asks "top N" or "first N"
    limit_match = re.search(r'\btop\s+(\d+)|first\s+(\d+)|list\s+(\d+)\b', q_lower)
    user_limit = None
    if limit_match:
        for g in limit_match.groups():
            if g and g.isdigit():
                user_limit = int(g)
                break

    return {
        "countries": countries,
        "domains": domain_intents,
        "requested_fields": list(requested_fields),
        "user_limit": user_limit
    }

# ------------------ Domain Skill Heuristic ------------------
def matches_domain(skills_list: List[str], domain_intents: List[str]) -> bool:
    if not domain_intents:
        return True
    s = set(skills_list)
    for d in domain_intents:
        for kw in DOMAIN_MAP.get(d, []):
            # check exact or substring in any skill
            if any(kw in skill for skill in s):
                return True
    return False

# ------------------ DB Enrichment ------------------
def fetch_applicant_details(applicant_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ENABLE_DB_ENRICH or not applicant_ids:
        return {}
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
        cur = conn.cursor()
        # Avoid huge IN clause if large
        placeholders = ",".join(["%s"] * len(applicant_ids))
        sql = f"""
            SELECT id, email, phone, upwork_profile_url, name, title, location, hourly_rate, total_earnings
            FROM applicants
            WHERE id IN ({placeholders})
        """
        cur.execute(sql, applicant_ids)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {r["id"]: r for r in rows}
    except Exception as e:
        print(f"DB enrichment failed: {e}")
        return {}

# ------------------ Retrieval & Grouping ------------------
def retrieve_candidates(query: str) -> Dict[str, Any]:
    parsed = parse_query(query)
    print("Parsed intent:", json.dumps(parsed, ensure_ascii=False))

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    coll = client.get_or_create_collection(COLLECTION_NAME)

    q_emb = embed_query(query)
    # Retrieve more than needed for filtering
    raw_res = coll.query(query_embeddings=[q_emb], n_results=TOP_K * 3)

    docs = raw_res.get("documents", [[]])[0]
    metas = raw_res.get("metadatas", [[]])[0]
    ids = raw_res.get("ids", [[]])[0]

    applicant_chunks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for meta, doc, cid in zip(metas, docs, ids):
        # Parse 'skills'
        skills_raw = meta.get("skills")
        skills_list = []
        if isinstance(skills_raw, str):
            try:
                skills_list = json.loads(skills_raw)
            except:
                skills_list = []
        elif isinstance(skills_raw, list):
            skills_list = skills_raw

        # Filters
        if parsed["countries"]:
            if meta.get("country_code") not in parsed["countries"]:
                continue
        if not matches_domain(skills_list, parsed["domains"]):
            continue

        applicant_id = meta.get("applicant_id")
        if applicant_id is None:
            continue

        snippet = doc[:MAX_SNIPPET_CHARS]
        applicant_chunks[applicant_id].append({
            "chunk_id": cid,
            "snippet": snippet,
            "skills": skills_list,
            "primary_title": meta.get("primary_title"),
            "country_code": meta.get("country_code"),
            "hourly_rate_num": meta.get("hourly_rate_num"),
            "location_region": meta.get("location_region"),
            "earnings_bucket": meta.get("earnings_bucket"),
            "experience_years_est": meta.get("experience_years_est"),
        })

    # Score applicants (simple: number of retained chunks + token variety heuristic)
    scored_applicants: List[Tuple[int, float]] = []
    for aid, chunk_list in applicant_chunks.items():
        score = len(chunk_list)
        # slight boost for domain skill intersections
        domain_hits = 0
        for c in chunk_list:
            for d in parsed["domains"]:
                for kw in DOMAIN_MAP.get(d, []):
                    if any(kw in sk for sk in c["skills"]):
                        domain_hits += 1
                        break
        score += 0.2 * domain_hits
        scored_applicants.append((aid, score))

    scored_applicants.sort(key=lambda x: x[1], reverse=True)

    # Apply user limit or default
    limit = parsed["user_limit"] if parsed["user_limit"] else MAX_APPLICANTS_IN_ANSWER
    top_applicants = scored_applicants[:limit]

    # Enrich with DB if enabled
    enrich_map = fetch_applicant_details([aid for aid, _ in top_applicants])

    structured_results = []
    for aid, score in top_applicants:
        chunks = applicant_chunks[aid]
        # keep only top N snippets for brevity
        top_snips = chunks[:MAX_SNIPPETS_PER_APPLICANT]
        # combine skills
        all_skills = Counter()
        for c in chunks:
            for s in c["skills"]:
                all_skills[s] += 1

        info = {
            "applicant_id": aid,
            "score": round(score, 3),
            "name": enrich_map.get(aid, {}).get("name"),
            "title": enrich_map.get(aid, {}).get("title") or (top_snips[0]["primary_title"] if top_snips else None),
            "country_code": top_snips[0]["country_code"] if top_snips else None,
            "location_region": top_snips[0]["location_region"] if top_snips else None,
            "hourly_rate_num": top_snips[0]["hourly_rate_num"] if top_snips else None,
            "earnings_bucket": top_snips[0]["earnings_bucket"] if top_snips else None,
            "experience_years_est": top_snips[0]["experience_years_est"] if top_snips else None,
            "top_skills": [s for s, _ in all_skills.most_common(12)],
            "snippets": [c["snippet"] for c in top_snips],
        }

        # Only include PII if requested & exists
        if ENABLE_DB_ENRICH:
            if "phone" in parsed["requested_fields"]:
                info["phone"] = enrich_map.get(aid, {}).get("phone")
            if "email" in parsed["requested_fields"]:
                info["email"] = enrich_map.get(aid, {}).get("email")
            if "profile" in parsed["requested_fields"]:
                info["profile_url"] = enrich_map.get(aid, {}).get("upwork_profile_url")

        structured_results.append(info)

    return {
        "parsed_intent": parsed,
        "results": structured_results
    }

# ------------------ Groq Answer Generation ------------------
def build_llm_prompt(user_query: str, structured: Dict[str, Any]) -> str:
    parsed = structured["parsed_intent"]
    results = structured["results"]

    if not results:
        return f"""The user asked: {user_query}
No matching applicants were found with the current filters: {json.dumps(parsed, ensure_ascii=False)}.
Explain politely and suggest relaxing filters."""

    lines = []
    lines.append(f"User Query: {user_query}")
    lines.append("Parsed Intent: " + json.dumps(parsed, ensure_ascii=False))
    lines.append("You are a helpful assistant summarizing applicant profiles. Provide only information present; if phone/email missing, state 'not provided'.")
    lines.append("Return a concise list first, then a brief summary. Avoid inventing data. If user asked for phone numbers and they are absent, explicitly say they are not in the dataset.\n")

    for r in results:
        lines.append(f"- ApplicantID: {r['applicant_id']}")
        lines.append(f"  Name: {r.get('name')}")
        lines.append(f"  Title: {r.get('title')}")
        lines.append(f"  Country: {r.get('country_code')}  Location: {r.get('location_region')}")
        lines.append(f"  HourlyRate: {r.get('hourly_rate_num')}  EarningsBucket: {r.get('earnings_bucket')}  ExperienceYears(est): {r.get('experience_years_est')}")
        if "phone" in r:
            lines.append(f"  Phone: {r.get('phone') or 'not provided'}")
        if "email" in r:
            lines.append(f"  Email: {r.get('email') or 'not provided'}")
        if "profile_url" in r:
            lines.append(f"  ProfileURL: {r.get('profile_url') or 'not provided'}")
        lines.append(f"  TopSkills: {', '.join(r['top_skills'])}")
        for sn in r["snippets"]:
            clean_sn = sn.replace("\n", " ").strip()
            lines.append(f"    Snippet: {clean_sn}")
        lines.append("")

    lines.append("Now produce the answer:")
    return "\n".join(lines)

def stream_llm_answer(prompt: str, model: str = "llama-3.3-70b-versatile"):
    if not GROQ_API_KEY or Groq is None:
        print("\n[Answer Generation Skipped: GROQ_API_KEY missing or groq library not installed]")
        return
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"You are an expert technical recruiting assistant."},
                  {"role":"user","content":prompt}],
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1.0,
        stream=True
    )
    print("\n--- LLM Answer (streaming) ---\n")
    for chunk in completion:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta and delta.content:
            print(delta.content, end="", flush=True)
    print("\n--- End of Answer ---\n")

# ------------------ Main CLI ------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_query.py \"your question about applicants\"")
        sys.exit(1)
    user_query = " ".join(sys.argv[1:])
    structured = retrieve_candidates(user_query)

    # Print structured JSON summary
    print("\n=== Structured Retrieval Results (JSON) ===")
    print(json.dumps(structured, ensure_ascii=False, indent=2))

    # Build prompt & stream answer
    prompt = build_llm_prompt(user_query, structured)
    stream_llm_answer(prompt)

if __name__ == "__main__":
    main()