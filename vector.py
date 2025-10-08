import pymysql
import json
import hashlib
from datetime import datetime
import chromadb
from groq import Groq

# -----------------------------
# 1️⃣ Connect to MySQL
# -----------------------------
conn = pymysql.connect(
    host='localhost',
    user='root',            
    password='#',  
    database='dash_db',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)
cursor = conn.cursor()
cursor.execute("SELECT * FROM applicants")
applicants_data = cursor.fetchall()
print(f"{len(applicants_data)} applicants loaded from MySQL")

# -----------------------------
# 2️⃣ Initialize Chroma DB
# -----------------------------
client = chromadb.Client()
collection = client.create_collection("applicants")

# -----------------------------
# 3️⃣ Initialize Groq client
# -----------------------------
groq_client = Groq(api_key="#")

def embed_text(text):
    """Generate embedding using Groq Qwen-Q3 embedding model."""
    completion = groq_client.embeddings.create(
        model="qwen/qwen3-32b",
        input=text
    )
    return completion.data[0].embedding

# -----------------------------
# Helpers
# -----------------------------
def parse_currency(value):
    """Convert strings like '$100K+', '$1.2M', '50' to float."""
    if not value:
        return 0.0
    try:
        value = str(value).replace("$", "").replace(",", "").replace("+", "").upper()
        if "K" in value:
            return float(value.replace("K", "")) * 1000
        elif "M" in value:
            return float(value.replace("M", "")) * 1_000_000
        else:
            return float(value)
    except:
        return 0.0

def parse_number(value):
    """Convert strings like '16,599' or None to float safely."""
    if not value:
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except:
        return 0.0

# -----------------------------
# 4️⃣ Chunk applicants & insert into Chroma
# -----------------------------
for applicant in applicants_data:
    chunks = []
    seq = 0

    # Chunk 1: Overview
    if applicant.get("overview"):
        seq += 1
        content = applicant["overview"]
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        chunk = {
            "applicant_id": applicant["id"],
            "chunk_type": "overview",
            "chunk_seq": seq,
            "primary_title": applicant.get("title"),
            "name": applicant.get("name"),
            "country_code": applicant.get("location", "").split(",")[-1].strip() if applicant.get("location") else None,
            "location_region": applicant.get("location"),
            "skills": json.loads(applicant["skills"]) if applicant.get("skills") else [],
            "job_success_pct": parse_number(applicant.get("job_success")),
            "hourly_rate_num": parse_currency(applicant.get("hourly_rate")),
            "earnings_bucket": "100_plus" if parse_currency(applicant.get("total_earnings")) > 100 else "0_100",
            "has_cloud_certs": False,
            "cloud_providers": [],
            "security_keywords": [],
            "experience_years_est": parse_number(applicant.get("total_hours"))/1000,
            "content_hash": content_hash,
            "version": 1,
            "scraped_at": applicant.get("scraped_at").isoformat() if applicant.get("scraped_at") else datetime.utcnow().isoformat()
        }
        chunks.append(chunk)

    # Chunk 2: Work History
    if applicant.get("work_history"):
        try:
            work_history_list = json.loads(applicant["work_history"])
        except:
            work_history_list = []

        for work_item in work_history_list:
            seq += 1
            if isinstance(work_item, str):
                content = work_item
                title = work_item[:50]
                skills = []
            elif isinstance(work_item, dict):
                content = work_item.get("description", "") or work_item.get("title", "")
                title = work_item.get("title")
                skills = work_item.get("skills", [])
            else:
                continue

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunk = {
                "applicant_id": applicant["id"],
                "chunk_type": "work_history",
                "chunk_seq": seq,
                "primary_title": title,
                "name": applicant.get("name"),
                "country_code": applicant.get("location", "").split(",")[-1].strip() if applicant.get("location") else None,
                "location_region": applicant.get("location"),
                "skills": skills,
                "job_success_pct": parse_number(applicant.get("job_success")),
                "hourly_rate_num": parse_currency(applicant.get("hourly_rate")),
                "earnings_bucket": "100_plus" if parse_currency(applicant.get("total_earnings")) > 100 else "0_100",
                "has_cloud_certs": False,
                "cloud_providers": [],
                "security_keywords": [],
                "experience_years_est": parse_number(applicant.get("total_hours"))/1000,
                "content_hash": content_hash,
                "version": 1,
                "scraped_at": applicant.get("scraped_at").isoformat() if applicant.get("scraped_at") else datetime.utcnow().isoformat()
            }
            chunks.append(chunk)

    # -----------------------------
    # Embed & insert into Chroma
    # -----------------------------
    for chunk in chunks:
        text_to_embed = (chunk.get("primary_title") or "") + " " + " ".join(chunk.get("skills", [])) + " " + (chunk.get("overview") or "")
        embedding_vector = embed_text(text_to_embed)

        collection.add(
            ids=[f"{chunk['applicant_id']}_{chunk['chunk_type']}_{chunk['chunk_seq']}"],
            embeddings=[embedding_vector],
            documents=[text_to_embed],
            metadatas=[chunk]
        )

print("✅ All applicants indexed in Chroma with embeddings!")
