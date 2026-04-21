# 🧠 AI-Powered Clinical Trial Matching System

## 📌 Overview

This project is an AI-based system that matches patients with relevant clinical trials using **Retrieval-Augmented Generation (RAG)**.

The system combines:

* Semantic search using vector embeddings
* Structured filtering using clinical criteria
* AI-based reasoning for eligibility evaluation

It improves over traditional keyword-based systems by providing **accurate, relevant, and explainable trial recommendations**.

---

## 🚀 Features

* 🔍 **Semantic Search (FAISS)**

  * Retrieves trials based on meaning, not just keywords

* 🧠 **RAG Pipeline**

  * Combines retrieval + AI reasoning

* 🏥 **Condition-Based Filtering**

  * Eliminates irrelevant trials (major improvement)

* 📊 **Eligibility Reasoning**

  * Uses AI to explain if a patient qualifies

* 💬 **Chat Interface**

  * User-friendly interaction via frontend

---

## 🏗️ System Architecture

```
User Input
   ↓
NLU Extraction (condition, age, etc.)
   ↓
FAISS Semantic Search
   ↓
Condition-Based Filtering
   ↓
RAG Reasoner (LLM)
   ↓
Final Trial Recommendations
```

---

## 🛠️ Tech Stack

**Backend**

* FastAPI (Python)
* PostgreSQL (Database)

**AI / ML**

* FAISS (Vector Search)
* OpenAI GPT (Eligibility Reasoning)

**Frontend**

* React (Vite)

**Data Processing**

* Pandas, NumPy

---

## 📂 Project Structure

```
clinical_trial_system_clean/
├── backend/        # RAG pipeline, APIs, reasoning modules
├── frontend/       # React-based UI
├── scripts/        # Data ingestion & embedding generation
├── data_sample/    # Sample clinical trial data (subset)
├── Data Validation.pdf  # Data preprocessing & validation
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/ai-clinical-trial-matching.git
cd ai-clinical-trial-matching
```

---

### 2️⃣ Backend Setup

```
pip install -r requirements.txt
```

---

### 3️⃣ Generate Embeddings (IMPORTANT)

Before running the system, generate vector embeddings:

```
python3 -m scripts.build_embeddings
```

---

### 4️⃣ Run Backend

```
uvicorn backend.api:app --reload
```

---

### 5️⃣ Run Frontend

```
cd frontend
npm install
npm run dev
```

---

## 📊 Sample Usage

**Input:**

```
My doctor thinks I have Crohn's disease. I'm 34, male.
```

**Output:**

* Relevant clinical trials
* Eligibility classification (YES / NO / POSSIBLE)
* Clear reasoning based on criteria

---

## 📁 Dataset Note

* The full dataset (~500,000 clinical trials) was used during development.
* Due to size constraints, only a **small sample dataset** is included in this repository.
* The system is designed to scale with large datasets.

---

## 📌 Key Improvements

* Reduced irrelevant trial retrieval using condition filtering
* Improved reasoning clarity by removing ambiguous outputs
* Optimized pipeline for structured + semantic matching

---

## ⚠️ Limitations

* Uses static dataset (not real-time)
* Requires embedding generation before use
* Eligibility results are AI-assisted and not clinically validated

---

## 🔮 Future Work

* Real-time integration with ClinicalTrials.gov API
* Faster response optimization
* Improved ranking of trials
* Personalized patient profiling

---

## 👩‍💻 Author

**Kannya Sambari**

---

## ⭐ Acknowledgements

* ClinicalTrials.gov dataset
* OpenAI API
* FAISS (Facebook AI Research)
