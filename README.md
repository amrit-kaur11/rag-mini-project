# Prompt Engineering & RAG Mini Project 

A **minimal, fully local Retrieval-Augmented Generation (RAG) pipeline** built with  
**FAISS**, **Sentence-Transformers**, and **Ollama** — no cloud APIs, no paid keys.


This project was developed as part of an **AI Engineer Intern – Take-Home Assignment**, with a strong focus on:
- Correct retrieval
- Clean prompting
- Reproducibility
- Evaluation


---


## ✨ Features


🔹 Deterministic text chunking with overlap
🔹 Sentence-Transformer embeddings (`all-MiniLM-L6-v2`)
🔹 FAISS vector similarity search
🔹 Local LLM inference via **Ollama**
🔹 Strict JSON-based prompting (reduces hallucinations)
🔹 End-to-end CLI demo
🔹 Automatic evaluation & CSV results


---


## 🧱 Project Structure



rag-mini/
├── data/
│ ├── source.txt # Raw input document
│ └── cleaned/
│ └── chunks.jsonl # Chunked text
├── scripts/
│ ├── prepare_data.py # Text chunking
│ └── index_vectors.py # FAISS index creation
├── src/
│ └── rag.py # Retrieval + generation logic
├── eval/
│ ├── evaluate.py # Evaluation script
│ └── results.csv # Evaluation output
├── query.py # Single-question CLI
├── run_demo.py # Demo questions
├── requirements.txt
└── README.md



---


## ⚙️ Setup Instructions


### 1️⃣ Create virtual environment


```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Install & run Ollama

Download from: https://ollama.com

Pull model:

```bash
ollama pull llama3.1:8b
```

---

## 📄 Data Preparation

Place your raw document inside:

data/source.txt

### Generate chunks:

python scripts/prepare_data.py --input data --output data/cleaned

---

## 🔎 Build FAISS Index
python scripts/index_vectors.py --chunks data/cleaned/chunks.jsonl --out faiss_index

---

## ❓ Ask Questions (CLI)
python query.py --question "What is the refund processing time?"

Example output:

{
  "answer": "14 days after the request is approved",
  "source_chunks": ["source_chunk_0"],
  "answerable": "yes"
}
### ▶️ Run Demo
python run_demo.py

Outputs are saved to:

demo_outputs.json

---

## 📊 Evaluation

Run automated evaluation:

python -m eval.evaluate

Results are saved to:

eval/results.csv

---

## 🧠 Design Decisions

Local-first: No OpenAI / cloud APIs

FAISS Flat index: Simple & deterministic

JSON-only prompting: Enforces structured outputs

Explicit retrieval → generation separation

Reproducible embeddings & indexing

---

## 🚧 Limitations

Single-document ingestion

No reranking stage

No streaming responses

Basic evaluation metrics

---

## 🌱 Future Improvements

Multi-document ingestion

Cross-encoder reranking

Hybrid (BM25 + vector) retrieval

FastAPI backend

Web UI (Streamlit / Next.js)

---

## 👩‍💻 Author

Amrit Kaur
AI / ML Engineer (Internship Candidate)

📌 Notes for Reviewers

This project emphasizes correctness, clarity, and reproducibility over scale.
All components run fully offline using open-source tools.


