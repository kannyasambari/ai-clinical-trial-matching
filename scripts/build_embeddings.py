import pickle
import psycopg2
import pandas as pd
import numpy as np
import faiss

from backend.config import DB_CONFIG, EMBEDDINGS_PKL_PATH, FAISS_INDEX_PATH
from backend.embedding_model import get_embedding


def fetch_data():
    print("🔌 Connecting to DB...")

    conn = psycopg2.connect(**DB_CONFIG)

    print("📥 Fetching data from trials table...")

    # 🔥 LIMIT ADDED (VERY IMPORTANT FOR SPEED)
    query = """
    SELECT 
        t.nct_id,
        t.title,
        t.brief_summary,
        e.full_criteria_text AS eligibility_criteria
    FROM trials t
    LEFT JOIN eligibility_criteria e
    ON t.nct_id = e.nct_id
    LIMIT 500
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"✅ Data loaded: {len(df)} rows")
    print(df.head(2))  # preview

    return df


def extract_criteria(raw):
    try:
        return (
            raw.get("protocolSection", {})
               .get("eligibilityModule", {})
               .get("eligibilityCriteria", "")
        )
    except Exception:
        return ""


def build_embeddings():
    print("🚀 Starting embedding pipeline...\n")

    df = fetch_data()

    print("\n🧠 Preparing text data...")

    texts = []

    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"📄 Processing row {idx}")

        criteria = row.get("eligibility_criteria", "") or ""

        # store for RAG
        df.at[idx, "cleaned_criteria"] = criteria

        text = f"""
        Title: {row.get('title', '')}

        Summary:
        {row.get('brief_summary', '')}

        Eligibility Criteria:
        {criteria}
        """

        texts.append(text)

    print("\n⚡ Starting embedding generation...")

    embeddings = []
    total = len(texts)

    for i, text in enumerate(texts):
        print(f"⚡ Embedding {i+1}/{total}")

        try:
            emb = get_embedding(text)
        except Exception as e:
            print(f"⚠️ Error at {i}: {e}")
            emb = [0.0] * 1536  # fallback

        embeddings.append(emb)

    print("\n📦 Converting to numpy array...")
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    print("🧠 Building FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("💾 Saving FAISS index...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("💾 Saving dataframe...")
    with open(EMBEDDINGS_PKL_PATH, "wb") as f:
        pickle.dump(df, f)

    print("\n🔥 DONE — Everything saved successfully!")


if __name__ == "__main__":
    build_embeddings()