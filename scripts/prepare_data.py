import argparse, json, re
from pathlib import Path
from tqdm import tqdm

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def chunk_text(text, chunk_size=2500, overlap=1250):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((start, end, chunk.strip()))
        if end == len(text):
            break
        start = end - overlap
    return chunks

def main(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / "chunks.jsonl"

    with out_file.open("w", encoding="utf-8") as f:
        for file in tqdm(list(input_dir.iterdir())):
            if file.suffix.lower() not in [".txt", ".md"]:
                continue
            text = clean_text(file.read_text(encoding="utf-8"))
            for start, end, chunk in chunk_text(text):
                record = {
                    "id": f"{file.stem}_chunk_{start}",
                    "source": file.name,
                    "text": chunk
                }
                f.write(json.dumps(record) + "\n")

    print("Chunks saved to", out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)