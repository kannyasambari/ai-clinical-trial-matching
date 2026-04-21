import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from backend.rag_pipeline import TrialRetriever
from backend.rag_reasoner import rag_reason

# -----------------------------
# Load retriever
# -----------------------------
retriever = TrialRetriever()


# -----------------------------
# Get patient input
# -----------------------------
print("\nEnter patient information\n")

age = input("Age: ")
sex = input("Sex (male/female): ")
conditions = input("Conditions (comma separated): ")
medications = input("Medications (comma separated): ")


# -----------------------------
# Build patient text
# -----------------------------
patient_text = f"""
Patient summary:
Age: {age}
Sex: {sex}
Conditions: {conditions}
Medications: {medications}
"""


print("\nSearching trials...\n")

# -----------------------------
# Retrieve trials
# -----------------------------
trials = retriever.retrieve_trials(patient_text, conditions, k=5)

# -----------------------------
# Run RAG reasoning
# -----------------------------
results = rag_reason(patient_text, trials)


# -----------------------------
# Display results
# -----------------------------
for r in results:

    print("\nTrial:", r["trial_id"])
    print(r["analysis"])
    print("-"*50)