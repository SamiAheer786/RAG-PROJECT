"""
Part 1: Corpus Preparation and Chunking
Medical FAQ Assistant RAG System
NUCES Chiniot-Faisalabad Campus
"""

import os
import re
import json
import string

# ─────────────────────────────────────────────
# 1. Load raw documents
# ─────────────────────────────────────────────

def load_documents(folder_path: str) -> list[dict]:
    """Load all .txt files from the corpus/raw folder."""
    documents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({
                "doc_id": filename.replace(".txt", ""),
                "filename": filename,
                "raw_text": text,
                "source": filepath,
            })
            print(f"  Loaded: {filename} ({len(text)} chars)")
    return documents


# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Clean and normalise a document string."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Remove non-ASCII characters (keep common punctuation)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def preprocess_documents(documents: list[dict]) -> list[dict]:
    """Preprocess all documents and save to corpus/processed."""
    processed = []
    os.makedirs("corpus/processed", exist_ok=True)
    for doc in documents:
        cleaned = preprocess_text(doc["raw_text"])
        doc_processed = {**doc, "processed_text": cleaned}
        processed.append(doc_processed)
        # Save to disk
        out_path = os.path.join("corpus/processed", doc["filename"])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
    print(f"\n  {len(processed)} documents preprocessed and saved to corpus/processed/")
    return processed


# ─────────────────────────────────────────────
# 3. Chunking strategies
# ─────────────────────────────────────────────

def chunk_by_fixed_size(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into fixed-size word chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_by_paragraph(text: str, min_len: int = 100) -> list[str]:
    """Split text by paragraphs; merge short ones."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text)]
    chunks, buffer = [], ""
    for para in paragraphs:
        if not para:
            continue
        buffer = (buffer + " " + para).strip() if buffer else para
        if len(buffer.split()) >= min_len:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_by_sentence(text: str, sentences_per_chunk: int = 5, overlap: int = 1) -> list[str]:
    """Split text into chunks of N sentences with sentence-level overlap."""
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + sentences_per_chunk
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += sentences_per_chunk - overlap
    return chunks


# ─────────────────────────────────────────────
# 4. Build the chunked corpus
# ─────────────────────────────────────────────

def build_chunks(processed_docs: list[dict], strategy: str = "paragraph") -> list[dict]:
    """
    Build a flat list of chunk records.
    Each record contains: chunk_id, doc_id, source, chunk_text, strategy.
    """
    all_chunks = []
    chunk_id = 0
    for doc in processed_docs:
        text = doc["processed_text"]
        if strategy == "fixed":
            raw_chunks = chunk_by_fixed_size(text)
        elif strategy == "sentence":
            raw_chunks = chunk_by_sentence(text)
        else:  # paragraph (default)
            raw_chunks = chunk_by_paragraph(text)

        for c in raw_chunks:
            if len(c.split()) < 20:   # skip tiny fragments
                continue
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "source": doc["filename"],
                "chunk_text": c,
                "strategy": strategy,
            })
            chunk_id += 1

    print(f"  Total chunks created ({strategy}): {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: list[dict], out_path: str = "corpus/chunks/chunks.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Chunks saved to {out_path}")


# ─────────────────────────────────────────────
# 5. Corpus statistics
# ─────────────────────────────────────────────

def print_corpus_stats(documents: list[dict], chunks: list[dict]):
    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    total_words = sum(len(d["processed_text"].split()) for d in documents)
    print(f"  Total documents : {len(documents)}")
    print(f"  Total words     : {total_words:,}")
    print(f"  Total chunks    : {len(chunks)}")
    print(f"  Avg chunk length: {sum(len(c['chunk_text'].split()) for c in chunks)//max(len(chunks),1)} words")
    print()
    for doc in documents:
        n_chunks = sum(1 for c in chunks if c["doc_id"] == doc["doc_id"])
        words    = len(doc["processed_text"].split())
        print(f"  {doc['doc_id']:40s}  {words:5d} words  {n_chunks:3d} chunks")
    print("=" * 60)


# ─────────────────────────────────────────────
# 6. Main entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[PART 1] Corpus Preparation and Chunking\n")

    RAW_DIR = "corpus/raw"

    print("Step 1: Loading raw documents...")
    docs = load_documents(RAW_DIR)

    print("\nStep 2: Preprocessing documents...")
    processed_docs = preprocess_documents(docs)

    print("\nStep 3: Chunking (paragraph strategy)...")
    chunks_para = build_chunks(processed_docs, strategy="paragraph")
    save_chunks(chunks_para, "corpus/chunks/chunks_paragraph.json")

    print("\nStep 3b: Chunking (fixed-size strategy)...")
    chunks_fixed = build_chunks(processed_docs, strategy="fixed")
    save_chunks(chunks_fixed, "corpus/chunks/chunks_fixed.json")

    print("\nStep 3c: Chunking (sentence strategy)...")
    chunks_sent = build_chunks(processed_docs, strategy="sentence")
    save_chunks(chunks_sent, "corpus/chunks/chunks_sentence.json")

    # Use paragraph chunks as the default
    save_chunks(chunks_para, "corpus/chunks/chunks.json")

    print_corpus_stats(processed_docs, chunks_para)
    print("\n[PART 1] Done.\n")
