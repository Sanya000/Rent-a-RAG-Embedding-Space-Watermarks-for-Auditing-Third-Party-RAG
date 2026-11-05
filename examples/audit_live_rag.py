"""Audit a live, black-box RAG system: ask it questions until the evidence decides.

The system under audit is any function ``query -> answer``. Here it is a small
toy RAG service that indexes two of the three providers; with a real stack you
pass its entry point instead:

    report = auditor.audit(chain.invoke, queries)                       # LangChain
    report = auditor.audit(query_engine.query, queries)                 # LlamaIndex
    report = auditor.audit(lambda q: requests.post(URL, json={"query": q}).json()["answer"], queries)

Runs offline (toy paraphraser and toy encoder). Use ``--llm`` / ``--embedder``
for real components, e.g. ``--llm openai/gpt-4o-mini --embedder sentence-transformers/all-mpnet-base-v2``.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

import rentarag
from demo_corpus import CORPORA


class ToyRAG:
    """Retrieve the three sentences closest to the query and return them as the answer."""

    def __init__(self, documents: list[str], embedder: rentarag.Embedder, seed: int = 0):
        self.sentences = [s for d in documents for s in rentarag.split_sentences(d)]
        self.embedder = embedder
        self.X = embedder(self.sentences)
        self.rng = random.Random(seed)

    def answer(self, query: str) -> str:
        sims = self.X @ self.embedder([query])[0]
        top = [self.sentences[i] for i in np.argsort(-sims)[:3]]
        words = " ".join(top).split()
        return " ".join(w for w in words if self.rng.random() > 0.05)  # light rewording


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="toy", help="LLM spec for watermarking and query generation")
    ap.add_argument("--embedder", default="hashing", help="encoder spec (provider and auditor share it)")
    args = ap.parse_args()
    embedder = rentarag.get_embedder(args.embedder)

    # Providers watermark and publish their corpora.
    keys = rentarag.ProviderKeys.generate(list(CORPORA), seed=7)  # seeded for a reproducible demo
    marked = rentarag.watermark_corpus(CORPORA, keys, paraphraser=args.llm, embedder=embedder)

    # A third-party RAG service indexes two of them.
    published = marked.by_provider()
    rag = ToyRAG(published["harbour-news"] + published["city-culture"], embedder)

    # The auditor writes questions about passages of the providers' documents (paper prompt) ...
    passages = [s for d in marked for s in rentarag.split_sentences(d.original)]
    queries = rentarag.generate_audit_queries(passages, args.llm, n=30, seed=0)
    # ... and asks them one by one. By default an audit stops at the first detection;
    # here it asks all 30 questions to show how evidence accumulates for every provider.
    auditor = rentarag.Auditor.from_corpus(marked, keys, "small-corpus", embedder)
    report = auditor.audit(rag.answer, queries, stop_when_detected=False)

    for step in report.trajectory:
        z = "  ".join(f"{p}: Z={step.z[p]:5.2f} T={step.support[p]:3d}" for p in keys)
        print(f"query {step.step:2d} | {z} | detected: {', '.join(step.detected) or '-'}")
    print(f"\n{report}\n\nqueries asked: {len(report.trajectory)}; first detected at query: {report.first_detected_at}")


if __name__ == "__main__":
    main()
