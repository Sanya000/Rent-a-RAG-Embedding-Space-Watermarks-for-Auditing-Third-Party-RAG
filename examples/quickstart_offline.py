"""End-to-end demo in a few lines: watermark three providers, simulate RAG answers, audit.

Runs offline in seconds: candidates come from the rule-based "toy" paraphraser
(not fluent text) and embeddings from the "hashing" toy encoder. Swap in real
components with two strings, e.g.

    python examples/quickstart_offline.py --llm openai/gpt-4o-mini --embedder sentence-transformers/all-mpnet-base-v2

The last block re-audits the same answers with 100 impostor keys for one
provider and summarises their pooled Z.
"""

from __future__ import annotations

import argparse
import random

import rentarag
from demo_corpus import CORPORA


def simulate_answers(marked: rentarag.WatermarkedCorpus, used: list[str], seed: int = 0) -> list[str]:
    """RAG answers that reuse (and lightly edit) sentences of the ``used`` providers, each sentence once."""
    rng = random.Random(seed)
    pools = {p: [s for d in marked if d.provider == p for s in rentarag.split_sentences(d.text)] for p in used}
    for sents in pools.values():
        rng.shuffle(sents)
    answers = []
    while all(len(s) >= 2 for s in pools.values()):
        sents = [pools[p].pop() for p in used for _ in range(2)]
        rng.shuffle(sents)
        answers.append(" ".join(w for w in " ".join(sents).split() if rng.random() > 0.05))
    return answers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="toy", help="paraphraser spec (default: offline toy)")
    ap.add_argument("--embedder", default="hashing", help="encoder spec (default: offline toy)")
    args = ap.parse_args()

    # 1. One secret key per provider. (Seeded here so the demo is reproducible;
    #    real deployments call ProviderKeys.generate(names) without a seed.)
    keys = rentarag.ProviderKeys.generate(list(CORPORA), seed=2026)

    # 2. Each provider's documents are watermarked with its own key.
    marked = rentarag.watermark_corpus(CORPORA, keys, paraphraser=args.llm, embedder=args.embedder)
    for provider, s in marked.summary().items():
        print(f"[{provider}] {s['documents']} docs, {s['changed_frac']:.0%} of sentences rewritten, mean gain {s['gain_mean']:.3f}")
    print(f"  before: {marked[0].original.split('. ')[0]}.\n  after:  {marked[0].text.split('. ')[0]}.")

    # 3. A RAG service answers questions with content from two of the three providers.
    answers = simulate_answers(marked, used=["harbour-news", "city-culture"])

    # 4. The auditor pools evidence over the answers and tests every provider.
    #    Each provider has only 20 sentences here, so the support gate is lowered to 10.
    auditor = rentarag.Auditor.from_corpus(marked, keys, "small-corpus", args.embedder)
    auditor.add_answers(answers)
    report = auditor.report()
    print(f"\n{report}\n\ndetected providers: {report.detected}")

    # 5. Control: the same answers audited with 100 impostor keys for harbour-news.
    #    Without the right key the pooled Z behaves like a standard normal.
    zs = []
    for s in range(100):
        impostor = dict(keys)
        impostor["harbour-news"] = rentarag.generate_provider_keys(1, seed=1000 + s)[0]
        control = rentarag.Auditor.from_corpus(marked, impostor, "small-corpus", args.embedder)
        control.add_answers(answers)
        zs.append(control.report()["harbour-news"].z_raw)
    mean = sum(zs) / len(zs)
    sd = (sum((z - mean) ** 2 for z in zs) / len(zs)) ** 0.5
    print(f"harbour-news Z with its key: {report['harbour-news'].z_raw:.1f}; with 100 impostor keys: mean {mean:.2f}, sd {sd:.2f}")

if __name__ == "__main__":
    main()
