from src.rag import run_query
import json

questions = [
    "What is the refund processing time?",
    "How can I cancel an order?",
    "Do you ship internationally?"
]

results = []
for q in questions:
    print("Running:", q)
    res = run_query(q, "faiss_index")
    results.append({"question": q, **res})

json.dump(results, open("demo_outputs.json", "w"), indent=2)
print("Saved demo_outputs.json")