# query.py

import argparse
from src.rag import run_query

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="Question to ask")
    args = parser.parse_args()

    result = run_query(args.question, "faiss_index")

    # IMPORTANT: PRINT RESULT
    print("\n=== ANSWER ===")
    print(result)

if __name__ == "__main__":
    main()