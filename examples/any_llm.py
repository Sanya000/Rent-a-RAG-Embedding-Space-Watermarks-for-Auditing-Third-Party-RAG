"""Watermark with any LLM: one spec string, or your own function.

    python examples/any_llm.py                                    # offline toy paraphraser
    python examples/any_llm.py --llm openai/gpt-4o-mini           # OPENAI_API_KEY      pip install "rentarag[openai]"
    python examples/any_llm.py --llm anthropic/claude-haiku-5-5   # ANTHROPIC_API_KEY   pip install "rentarag[anthropic]"
    python examples/any_llm.py --llm gemini/gemini-2.5-flash      # GEMINI_API_KEY      pip install "rentarag[openai]"
    python examples/any_llm.py --llm ollama/llama3.1              # local `ollama serve`
    python examples/any_llm.py --llm vllm/Qwen/Qwen2.5-7B-Instruct  # local `vllm serve Qwen/Qwen2.5-7B-Instruct`
    python examples/any_llm.py --llm hf/Qwen/Qwen2.5-1.5B-Instruct  # local transformers  pip install "rentarag[hf]"
    python examples/any_llm.py --llm litellm/mistral/mistral-small-latest   # pip install "rentarag[litellm]"
    python examples/any_llm.py --llm custom                       # your own Python function (below)

Any OpenAI-compatible server works with ``--llm openai/<model> --base-url http://host:port/v1``.
With the toy paraphraser the demo uses the toy "hashing" encoder; with a real
LLM it uses the paper's encoder (all-mpnet-base-v2, downloaded on first use).
"""

from __future__ import annotations

import argparse

import rentarag
from demo_corpus import CORPORA


def my_paraphraser(sentence: str, n: int) -> list[str]:
    """Bring your own generator: any function returning up to n rewrites of a sentence.

    This one is rule-based so the demo runs offline; in practice it would call
    your model, your gateway, a cache, ...
    """
    variants = [
        sentence,
        sentence.replace(" the ", " this ", 1),
        "In short, " + sentence[0].lower() + sentence[1:],
        sentence.replace(" and ", " as well as ", 1),
        sentence.replace(" said ", " stated ", 1),
        "Notably, " + sentence[0].lower() + sentence[1:],
    ]
    return [v for v in variants if v != sentence][:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="toy", help="LLM spec (see above), 'toy' or 'custom'")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for --llm openai/<model>")
    ap.add_argument("--embedder", default=None, help="encoder spec (default: hashing for toy/custom, else all-mpnet-base-v2)")
    ap.add_argument("--prompt", default="corpus", help="corpus | k-variants | your template with {sentence} and {n}")
    ap.add_argument("--preset", default="fast", help="watermark preset: paper | fast | strong | paper-corpus")
    ap.add_argument("--workers", type=int, default=4, help="parallel documents (API backends)")
    args = ap.parse_args()

    offline = args.llm in ("toy", "custom")
    embedder = args.embedder or ("hashing" if offline else "sentence-transformers/all-mpnet-base-v2")
    if args.llm == "custom":
        paraphraser = rentarag.get_paraphraser(my_paraphraser)
    elif args.llm == "toy":
        paraphraser = rentarag.get_paraphraser("toy")
    else:
        llm_options = {"base_url": args.base_url} if args.base_url else {}
        paraphraser = rentarag.get_paraphraser(args.llm, prompt=args.prompt, **llm_options)
    print(f"paraphraser: {paraphraser!r}\nencoder:     {embedder}")

    keys = rentarag.ProviderKeys.generate(list(CORPORA))  # fresh secret keys
    docs = {p: texts[:1] for p, texts in CORPORA.items()}  # one document per provider keeps API costs small
    marked = rentarag.watermark_corpus(
        docs, keys, paraphraser=paraphraser, embedder=embedder, config=args.preset, max_workers=args.workers, keep_sentences=True
    )
    for d in marked:
        print(f"\n[{d.provider}] {d.stats['changed_sentences']}/{d.stats['n_sentences']} sentences rewritten, mean gain {d.stats['gain_mean']:+.4f}")
        for s in d.result.sentences[:2]:
            print(f"  - {s.original}\n  + {s.text}  (gain {s.gain:+.4f}, cos {s.sem_sim:.2f})")

    # The same LLM can write audit queries with the paper's prompt.
    if args.llm != "custom":
        llm = paraphraser.llm if isinstance(paraphraser, rentarag.LLMParaphraser) else "toy"
        print("\naudit queries:", *rentarag.generate_audit_queries(marked, llm, n=2), sep="\n  ")


if __name__ == "__main__":
    main()
