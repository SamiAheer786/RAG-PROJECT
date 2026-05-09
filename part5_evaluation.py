"""
Part 5: Experimental Evaluation
Medical FAQ Assistant RAG System
NUCES Chiniot-Faisalabad Campus

Evaluates the RAG system on 10 questions with:
  - Correct responses
  - Failure/edge cases
  - Observed limitations
Metrics: retrieval precision@k, answer relevance (keyword overlap)
"""

import os
import sys
import json
import re
import textwrap
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────
# Ground-truth Q&A pairs for evaluation
# ─────────────────────────────────────────────

EVAL_QA = [
    {
        "id": 1,
        "question": "What are the two main types of diabetes?",
        "expected_keywords": ["type 1", "type 2", "autoimmune", "insulin"],
        "relevant_doc": "doc1_diabetes",
        "category": "factual",
    },
    {
        "id": 2,
        "question": "What blood pressure level is considered hypertensive crisis?",
        "expected_keywords": ["180", "120", "crisis", "immediate"],
        "relevant_doc": "doc2_hypertension",
        "category": "specific fact",
    },
    {
        "id": 3,
        "question": "Name three common asthma triggers.",
        "expected_keywords": ["pollen", "dust", "smoke", "exercise", "mold", "pet"],
        "relevant_doc": "doc3_asthma",
        "category": "list",
    },
    {
        "id": 4,
        "question": "What is the DASH diet?",
        "expected_keywords": ["dietary", "approaches", "stop", "hypertension", "sodium", "fruits"],
        "relevant_doc": "doc2_hypertension",
        "category": "definition",
    },
    {
        "id": 5,
        "question": "How does coronary heart disease develop?",
        "expected_keywords": ["atherosclerosis", "plaque", "arteries", "cholesterol", "narrowed"],
        "relevant_doc": "doc4_heart_disease",
        "category": "mechanism",
    },
    {
        "id": 6,
        "question": "What BMI is classified as obese?",
        "expected_keywords": ["30", "bmi", "obese"],
        "relevant_doc": "doc5_obesity",
        "category": "specific fact",
    },
    {
        "id": 7,
        "question": "What is cognitive behavioral therapy used for?",
        "expected_keywords": ["depression", "negative", "thought", "patterns", "cbt"],
        "relevant_doc": "doc6_depression",
        "category": "treatment",
    },
    {
        "id": 8,
        "question": "What is herd immunity?",
        "expected_keywords": ["herd", "immunity", "community", "immune", "percentage"],
        "relevant_doc": "doc7_vaccines",
        "category": "definition",
    },
    {
        "id": 9,
        "question": "Why should you complete the full course of antibiotics?",
        "expected_keywords": ["resistance", "complete", "course", "bacteria", "reproduce"],
        "relevant_doc": "doc8_antibiotics",
        "category": "reasoning",
    },
    {
        "id": 10,
        "question": "Can type 2 diabetes be prevented?",  # tests cross-document reasoning
        "expected_keywords": ["prevent", "lifestyle", "weight", "diet", "exercise"],
        "relevant_doc": "doc1_diabetes",
        "category": "prevention",
    },
    # Failure / edge cases
    {
        "id": 11,
        "question": "What is the cure for cancer?",   # out-of-scope
        "expected_keywords": [],
        "relevant_doc": None,
        "category": "out-of-scope",
    },
    {
        "id": 12,
        "question": "asdflkj random gibberish xyz123",  # nonsense
        "expected_keywords": [],
        "relevant_doc": None,
        "category": "adversarial",
    },
]


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def keyword_overlap(text: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in text (case-insensitive)."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def retrieval_hit(retrieved: list[dict], relevant_doc: Optional[str]) -> bool:
    """True if the relevant doc appears in the retrieved chunks."""
    if relevant_doc is None:
        return False
    return any(relevant_doc in r["doc_id"] for r in retrieved)


def precision_at_k(retrieved: list[dict], relevant_doc: Optional[str]) -> float:
    """Precision@k (treating the one relevant doc as ground truth)."""
    if relevant_doc is None or not retrieved:
        return 0.0
    relevant = sum(1 for r in retrieved if relevant_doc in r["doc_id"])
    return relevant / len(retrieved)


# ─────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────

def evaluate(pipeline, eval_qa: list[dict]) -> dict:
    results = []
    hits, total_kw_score, total_prec = 0, 0.0, 0.0
    in_scope = [q for q in eval_qa if q["relevant_doc"] is not None]

    print("\n" + "=" * 70)
    print("EXPERIMENTAL EVALUATION RESULTS")
    print("=" * 70)

    for qa in eval_qa:
        record = pipeline.answer(qa["question"], verbose=False)
        answer = record["answer"]
        retrieved = record["retrieved_chunks"]

        kw_score = keyword_overlap(answer + " " + " ".join(r["chunk_text"] for r in retrieved),
                                   qa["expected_keywords"])
        ret_hit  = retrieval_hit(retrieved, qa["relevant_doc"])
        prec     = precision_at_k(retrieved, qa["relevant_doc"])

        if qa["relevant_doc"]:
            hits           += int(ret_hit)
            total_kw_score += kw_score
            total_prec     += prec

        result = {
            "id":            qa["id"],
            "question":      qa["question"],
            "category":      qa["category"],
            "relevant_doc":  qa["relevant_doc"],
            "retrieval_hit": ret_hit,
            "precision_at_k": round(prec, 3),
            "keyword_score": round(kw_score, 3),
            "answer":        answer[:300],
            "top_sources":   [r["source"] for r in retrieved],
        }
        results.append(result)

        # Console output
        status = "✅" if ret_hit else "❌"
        print(f"\nQ{qa['id']:02d} [{qa['category']:15s}] {status}  "
              f"kw={kw_score:.2f}  P@k={prec:.2f}")
        print(f"  Question : {qa['question']}")
        print(f"  Sources  : {result['top_sources']}")
        wrapped = textwrap.fill(answer[:200], width=66, initial_indent="  Answer   : ",
                                subsequent_indent="            ")
        print(wrapped)

    # Aggregate metrics (only in-scope questions)
    n = len(in_scope)
    metrics = {
        "total_questions":    len(eval_qa),
        "in_scope_questions": n,
        "retrieval_accuracy": round(hits / n, 3) if n else 0,
        "avg_precision_at_k": round(total_prec / n, 3) if n else 0,
        "avg_keyword_score":  round(total_kw_score / n, 3) if n else 0,
    }

    print("\n" + "=" * 70)
    print("AGGREGATE METRICS (in-scope questions only)")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

    return {"per_question": results, "metrics": metrics}


def save_eval_results(results: dict, path: str = "reports/evaluation_results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Evaluation results saved to {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from part2_retrieval import load_chunks, TFIDFRetriever, DenseRetriever
    from part3_generation import RAGPipeline, TemplateLLM, HuggingFaceLLM

    print("\n[PART 5] Experimental Evaluation\n")

    chunks = load_chunks("corpus/chunks/chunks.json")

    # Use Dense retriever if available, else TF-IDF
    try:
        retriever = DenseRetriever()
        retriever.load_model()
        retriever.build_index(chunks)
    except Exception:
        retriever = TFIDFRetriever()
        retriever.build_index(chunks)

    # LLM
    try:
        llm = HuggingFaceLLM()
        llm.load()
    except Exception:
        llm = TemplateLLM()

    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=3)
    results = evaluate(pipeline, EVAL_QA)
    save_eval_results(results)

    print("\n[PART 5] Done.\n")
