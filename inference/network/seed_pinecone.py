from pinecone import Pinecone, ServerlessSpec
import os
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import numpy as np
import time

# =========================
# ENV
# =========================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "security-knowledge")

existing_indexes = [i.name for i in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# wait until ready

while True:
    desc = pc.describe_index(index_name)
    if desc.status["ready"]:
        break
    time.sleep(1)

index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# LOAD PARQUET
# =========================
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "UNSW_Flow.parquet"

df = pd.read_parquet(DATA_PATH).sample(10_000, random_state=42)

df = (
    df.groupby("attack_label", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 50), random_state=42))
)
print(df["attack_label"].value_counts())

print(f"Loaded {len(df)} rows")


# =========================
# HELPER: FLOW → TEXT
# =========================
def flow_to_text(row):
    return f"""
    Network flow:
    duration {row.get('dur', 0)} seconds,
    source bytes {row.get('sbytes', 0)},
    destination bytes {row.get('dbytes', 0)},
    source packets {row.get('spkts', 0)},
    destination packets {row.get('dpkts', 0)},
    source ttl {row.get('sttl', 0)},
    destination ttl {row.get('dttl', 0)},
    source load {row.get('sload', 0)},
    destination load {row.get('dload', 0)},
    tcp round trip time {row.get('tcprtt', 0)},
    synack {row.get('synack', 0)},
    ackdat {row.get('ackdat', 0)},
    source jitter {row.get('sjit', 0)},
    destination jitter {row.get('djit', 0)},
    source mean packet size {row.get('smeansz', 0)},
    destination mean packet size {row.get('dmeansz', 0)}
    """

# =========================
# BUILD TEXTS
# =========================
texts = df.apply(flow_to_text, axis=1).tolist()

print("Encoding...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

print(f"Embeddings shape: {len(embeddings)}")

# =========================
# BUILD VECTORS
# =========================
vectors = []

for i in range(len(df)):
    row = df.iloc[i]
    emb = embeddings[i]

    metadata = {
        "label": int(row.get("label", 0)),
        "attack_label": str(row.get("attack_label", "")),
        "dur": float(row.get("dur", 0)),
        "spkts": int(row.get("spkts", 0)),
        "dpkts": int(row.get("dpkts", 0)),
        "sbytes": float(row.get("sbytes", 0)),
        "dbytes": float(row.get("dbytes", 0)),
    }

    vectors.append({
        "id": f"flow-{i}",
        "values": emb.tolist(),
        "metadata": metadata
    })

print(f"Built {len(vectors)} vectors")
# =========================
# UPSERT
# =========================
BATCH_SIZE = 50

for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i:i+BATCH_SIZE]
    index.upsert(
        vectors=batch,
        namespace="network-flows"
    )
print(f"Done seeding {len(vectors)} flows into Pinecone")