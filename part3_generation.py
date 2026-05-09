"""
Part 3: Generation Module
Medical FAQ Assistant RAG System
NUCES Chiniot-Faisalabad Campus

Uses retrieved context + open-source LLM (via HuggingFace or Ollama)
to generate grounded answers. Falls back to a template-based approach
when no GPU / model is available (for demo/testing).
"""

import os
import json
import textwrap
from typing import Optional

# ─────────────────────────────────────────────
# 1. Prompt builder
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable medical FAQ assistant.
Answer the user's question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Do not fabricate facts. Be concise but thorough.
Always recommend consulting a healthcare professional for personal medical advice."""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build a RAG prompt with retrieved context injected."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['chunk_text']}"
        for c in context_chunks
    )
    prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
    return prompt


# ─────────────────────────────────────────────
# 2. LLM backends
# ─────────────────────────────────────────────

class HuggingFaceLLM:
    """
    Uses a small open-source model from HuggingFace.
    Default: google/flan-t5-base (fast, no GPU needed).
    For better quality use: google/flan-t5-large
    """

    DEFAULT_MODEL = "google/flan-t5-base"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.pipeline = None

    def load(self):
        from transformers import pipeline
        print(f"  [LLM] Loading {self.model_name} ...")
        self.pipeline = pipeline(
            "text2text-generation",
            model=self.model_name,
            max_new_tokens=256,
            do_sample=False,
        )
        print("  [LLM] Model ready.")

    def generate(self, prompt: str) -> str:
        if self.pipeline is None:
            self.load()
        # flan-t5 works best with instruction-style prompts; trim if needed
        truncated = prompt[:2000]
        output = self.pipeline(truncated)
        return output[0]["generated_text"].strip()


class OllamaLLM:
    """
    Uses a locally-running Ollama instance (e.g. ollama run llama3).
    Requires: pip install ollama  and  ollama server running.
    """

    def __init__(self, model: str = "llama3"):
        self.model = model

    def generate(self, prompt: str) -> str:
        import ollama
        response = ollama.generate(model=self.model, prompt=prompt)
        return response["response"].strip()


class TemplateLLM:
    """
    Fallback: returns a template answer that cites the retrieved chunks.
    Used when no GPU / internet is available. Demonstrates the RAG pipeline
    correctly even without a heavy model.
    """

    def generate(self, prompt: str) -> str:
        # Extract the ANSWER section and the context
        lines = prompt.split("\n")
        q_line = next((l for l in lines if l.startswith("QUESTION:")), "")
        question = q_line.replace("QUESTION:", "").strip()

        # Find context sources
        sources = []
        for l in lines:
            if l.startswith("[Source:"):
                src = l.replace("[Source:", "").replace("]", "").strip()
                sources.append(src)

        answer = (
            f"Based on the retrieved medical documents ({', '.join(set(sources))}), "
            f"here is a summary relevant to your question about '{question}':\n\n"
            "The retrieved context contains detailed information on this topic. "
            "Key points from the knowledge base have been identified and presented above. "
            "Please review the context sections for specific details. "
            "For personalised medical advice, consult a qualified healthcare professional."
        )
        return answer


# ─────────────────────────────────────────────
# 3. RAG pipeline
# ─────────────────────────────────────────────

class RAGPipeline:
    """End-to-end RAG: retrieve → prompt → generate."""

    def __init__(self, retriever, llm, top_k: int = 3):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.history: list[dict] = []   # chat history (bonus)

    def answer(self, query: str, verbose: bool = True) -> dict:
        # Step 1 – retrieve
        context_chunks = self.retriever.retrieve(query, top_k=self.top_k)

        # Step 2 – build prompt
        prompt = build_rag_prompt(query, context_chunks)

        # Step 3 – generate
        answer = self.llm.generate(prompt)

        record = {
            "query":          query,
            "retrieved_chunks": context_chunks,
            "prompt":         prompt,
            "answer":         answer,
        }
        self.history.append(record)

        if verbose:
            _print_qa(query, context_chunks, answer)

        return record


def _print_qa(query, chunks, answer):
    print("\n" + "=" * 70)
    print(f"USER QUESTION:\n  {query}")
    print("\nRETRIEVED CONTEXT:")
    for i, c in enumerate(chunks, 1):
        print(f"\n  [{i}] Source: {c['source']}  (score={c['score']:.4f})")
        wrapped = textwrap.fill(c["chunk_text"], width=70, initial_indent="      ",
                                subsequent_indent="      ")
        print(wrapped[:500] + ("..." if len(wrapped) > 500 else ""))
    print(f"\nGENERATED ANSWER:\n")
    wrapped_ans = textwrap.fill(answer, width=70, initial_indent="  ",
                                subsequent_indent="  ")
    print(wrapped_ans)
    print("=" * 70)


# ─────────────────────────────────────────────
# 4. Demo queries
# ─────────────────────────────────────────────

DEMO_QUERIES = [
    "What is type 2 diabetes?",
    "How does high blood pressure affect the body?",
    "What are the treatment options for asthma?",
    "What foods should people with heart disease avoid?",
    "How do vaccines work to prevent disease?",
]


def save_generation_results(records: list[dict], path: str = "reports/generation_results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Remove the raw prompt from saved output to keep file readable
    clean = []
    for r in records:
        clean.append({
            "query": r["query"],
            "sources": [c["source"] for c in r["retrieved_chunks"]],
            "answer": r["answer"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\n  Generation results saved to {path}")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from part2_retrieval import load_chunks, TFIDFRetriever, DenseRetriever

    print("\n[PART 3] Generation Module\n")

    # Load retrieval layer (use Dense by default; fall back to TF-IDF if needed)
    chunks = load_chunks("corpus/chunks/chunks.json")

    try:
        retriever = DenseRetriever()
        retriever.load_model()
        retriever.build_index(chunks)
        print("  Using Dense retriever.")
    except Exception as e:
        print(f"  Dense retriever failed ({e}); falling back to TF-IDF.")
        retriever = TFIDFRetriever()
        retriever.build_index(chunks)

    # Choose LLM backend
    llm_backend = os.getenv("RAG_LLM_BACKEND", "hf")   # "hf" | "ollama" | "template"

    if llm_backend == "ollama":
        llm = OllamaLLM(model="llama3")
        print("  Using Ollama LLM (llama3).")
    elif llm_backend == "hf":
        try:
            llm = HuggingFaceLLM()
            llm.load()
            print("  Using HuggingFace LLM (flan-t5-base).")
        except Exception as e:
            print(f"  HuggingFace LLM failed ({e}); using TemplateLLM.")
            llm = TemplateLLM()
    else:
        llm = TemplateLLM()
        print("  Using Template LLM (demo mode).")

    # Run pipeline
    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=3)
    records = []
    for query in DEMO_QUERIES:
        record = pipeline.answer(query)
        records.append(record)

    save_generation_results(records)
    print("\n[PART 3] Done.\n")
