from eligibility_reasoner import evaluate_trial

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import faiss
import numpy as np
import pickle
from embedding_model import get_embedding


# -----------------------------
# Load FAISS index
# -----------------------------
print("Loading FAISS index...")
index = faiss.read_index("embeddings_output/trial_faiss.index")
print("Index loaded:", index.ntotal)


# -----------------------------
# Load trial dataframe
# -----------------------------
print("\nLoading trial dataframe...")

with open("embeddings_output/df_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

print("Trials loaded:", len(df))
print("Available columns:", df.columns)


# -----------------------------
# Column containing eligibility text
# -----------------------------
text_col = "cleaned_criteria"


# -----------------------------
# Example patient profile
# -----------------------------
patient_profile = {
    "age": 55,
    "sex": "male",
    "conditions": ["type 2 diabetes", "hypertension"],
    "medications": ["metformin", "lisinopril"]
}


# -----------------------------
# Convert patient profile to text
# -----------------------------
patient_text = f"""
Patient eligibility summary:
Male patient aged {patient_profile['age']}.
Diagnosed with type 2 diabetes mellitus and hypertension.
Currently taking metformin and lisinopril.
Looking for clinical trials for diabetes management.
"""


# -----------------------------
# Generate embedding
# -----------------------------
print("\nGenerating embedding...")

patient_embedding = get_embedding(patient_text)

patient_embedding = np.array(patient_embedding, dtype=np.float32).reshape(1, -1)
patient_embedding = np.ascontiguousarray(patient_embedding)

print("Embedding shape:", patient_embedding.shape)


# -----------------------------
# FAISS search
# -----------------------------
print("\nSearching FAISS...")

k = 50
D, I = index.search(patient_embedding, k)

print("Search complete")


# -----------------------------
# Retrieve trials
# -----------------------------
results = df.iloc[I[0]]


# -----------------------------
# Structured filtering
# -----------------------------
patient_age = patient_profile["age"]
patient_sex = patient_profile["sex"]

structured_filter = results[
    (results["min_age_years"].fillna(0) <= patient_age) &
    ((results["max_age_years"].isna()) | (results["max_age_years"] >= patient_age)) &
    ((results["sex"].str.lower() == "all") | (results["sex"].str.lower() == patient_sex))
]


# -----------------------------
# Condition filtering
# -----------------------------
conditions = "|".join(patient_profile["conditions"])

condition_filter = structured_filter[
    structured_filter[text_col].str.contains(conditions, case=False, na=False)
]

filtered = condition_filter


print("\nFiltered trials:", len(filtered))


# -----------------------------
# Display results
# -----------------------------
patient_summary = patient_text

for rank, (_, trial) in enumerate(filtered.head(5).iterrows(), start=1):

    print(f"\nRank #{rank}")
    print("Trial ID:", trial["nct_id"])

    eligibility_text = trial["cleaned_criteria"]

    print("\nRunning eligibility analysis...")

    explanation = evaluate_trial(patient_summary, eligibility_text)

    print("\nLLM Evaluation:")
    print(explanation)

    print("-" * 60)