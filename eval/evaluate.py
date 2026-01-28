# eval/evaluate.py

import json
import csv
from src.rag import run_query


def score_answer(answerable):
    """
    Scoring rubric:
    yes     -> 2 points
    partial -> 1 point
    no      -> 0 points
    """
    if answerable == "yes":
        return 2
    if answerable == "partial":
        return 1
    return 0


def main():
    with open("eval/eval_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    for q in questions:
        question_text = q["question"]

        res = run_query(question_text, "faiss_index")

        # Ollama JSON comes as a STRING → parse it
        try:
            model_json = json.loads(res["raw_response"])
        except json.JSONDecodeError:
            model_json = {
                "answer": "",
                "source_chunks": [],
                "answerable": "no",
                "notes": "Model output was not valid JSON"
            }

        answerable = model_json.get("answerable", "no")
        score = score_answer(answerable)

        results.append({
            "question": question_text,
            "answer": model_json.get("answer", ""),
            "answerable": answerable,
            "score": score,
            "notes": model_json.get("notes", "")
        })

    # Write CSV
    with open("eval/results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "answerable", "score", "notes"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("Evaluation complete. Results saved to eval/results.csv")


if __name__ == "__main__":
    main()