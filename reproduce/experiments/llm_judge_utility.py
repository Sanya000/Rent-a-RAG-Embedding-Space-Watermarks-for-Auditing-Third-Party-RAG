#!/usr/bin/env python3
"""Blind LLM-judge utility evaluation (paper App. app:llm-judge, Sec. 5.8).

Sentence level: (source, watermarked) sentence pairs are drawn from the
cost-ablation candidate log (``watermark_cost_ablation.py``; the watermarked
sentence is the K=24, rho=6 hill-climb result from ``simulate_krho``), shuffled
with ``--seed``, and the first ``--n_sent`` changed pairs are rated for meaning
(1-5), naturalness (1-5), factual consistency (0/1) and no-omission (0/1).

Answer level: for ``--n_per_target`` random trials per target of the primary
trial log of ``--trial_seed`` (default seed 01), the logged watermarked-corpus
answer (gpt-4o-mini) is compared blind (random A/B order) with a source-corpus
answer regenerated from the same retrieved doc ids using the source texts.

Paper run (results/llm_judge_gpt4o): ``--n_sent 200 --n_per_target 15
--model gpt-4o --seed 42`` -> 200 sentence pairs and 90 answer pairs. The source
answer is regenerated with ``--source_answer_model`` (default: the judge model)
while the watermarked answer comes from the gpt-4o-mini log;
``--source_answer_model gpt-4o-mini`` makes the comparison symmetric. A/B order
follows async completion order.
"""
from __future__ import annotations

import argparse, asyncio, json, os, random, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from watermark_cost_ablation import simulate_krho   # reuse for getting final WM text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SENTENCE_JUDGE_PROMPT = """\
You are a precise linguistic evaluator. You will be shown an original sentence
and a paraphrased version of it. Evaluate the paraphrase on four criteria.

Original:
{source}

Paraphrase:
{watermarked}

Rate the paraphrase:
1. Meaning Preservation (1–5): Does it convey the same core information as the original?
   (1 = completely different meaning, 3 = partially preserved, 5 = fully preserved)
2. Naturalness (1–5): Does it read naturally and fluently in English?
   (1 = very awkward or ungrammatical, 3 = acceptable, 5 = completely natural)
3. Factual Consistency (1/0): Are ALL facts in the paraphrase also present in the original?
   (1 = no new or contradictory facts, 0 = at least one factual error or hallucination)
4. No Important Omission (1/0): Are the key facts from the original preserved in the paraphrase?
   (1 = nothing important is missing, 0 = important information was omitted)

Respond with valid JSON only — no preamble, no explanation outside the JSON:
{{"meaning": <int 1-5>, "naturalness": <int 1-5>, "factual": <0 or 1>, "complete": <0 or 1>, "note": "<15 words max>"}}"""

ANSWER_GEN_PROMPT = """\
You are a RAG assistant. Answer using ONLY the provided documents.
Synthesize across multiple docs. Use your OWN WORDS; avoid quoting exact spans.

Here are retrieved documents:
{docs}

Question: {question}"""

ANSWER_JUDGE_PROMPT = """\
You are a fair, neutral judge evaluating two AI-generated answers to the same question.
The answers were produced by different knowledge bases. Do not favour brevity or verbosity.

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Evaluate each answer independently, then state your preference.
1. Relevance (1–5): Does the answer directly address the question?
2. Accuracy (1–5): Does the answer appear factually reliable and consistent?
3. Completeness (1–5): Does it cover the key points a reader needs?
4. Fluency (1–5): Is it well-written, clear, and coherent?

Return valid JSON only:
{{"a_relevance":<1-5>,"a_accuracy":<1-5>,"a_completeness":<1-5>,"a_fluency":<1-5>,
  "b_relevance":<1-5>,"b_accuracy":<1-5>,"b_completeness":<1-5>,"b_fluency":<1-5>,
  "preference":"A"|"B"|"equivalent","reason":"<one sentence>"}}"""


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------
async def _achat(client: Any, prompt: str, model: str,
                 semaphore: asyncio.Semaphore, temperature: float = 0.0) -> str:
    async with semaphore:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=temperature,
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == 2:
                    return ""
                await asyncio.sleep(2.0 * (attempt + 1))
    return ""


def _parse_json_safe(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    # strip markdown code fences if present
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(l for l in lines if not l.startswith("```")).strip()
    try:
        return json.loads(s)
    except Exception:
        # try to extract JSON object
        import re
        m = re.search(r'\{.*\}', s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# Phase 1: sentence-level judge
# ---------------------------------------------------------------------------
def load_sentence_pairs(debug_path: Path, n: int, seed: int) -> List[Dict[str, str]]:
    """Load changed (source, watermarked) pairs from the ablation debug log."""
    recs = [json.loads(l) for l in debug_path.open("r", encoding="utf-8")]
    pairs: List[Dict[str, str]] = []
    for rec in recs:
        for s in rec["sentences"]:
            _, changed, accepted = simulate_krho(s, 24, 6)
            if changed:
                pairs.append({
                    "source": s["base_text"],
                    "watermarked": accepted,
                    "doc_id": rec["doc_id"],
                    "provider_id": str(rec["provider_id"]),
                    "gain": s.get("rounds", [{}])[0][0].get("proj", 0) if s.get("rounds") and s["rounds"][0] else 0,
                })
    rng = np.random.RandomState(seed)
    rng.shuffle(pairs)
    return pairs[:n]


def summarise_sentence(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    meaning, natural, factual, complete = [], [], [], []
    failed = 0
    for r in results:
        p = r.get("parsed")
        if not p:
            failed += 1
            continue
        if "meaning" in p:    meaning.append(int(p["meaning"]))
        if "naturalness" in p: natural.append(int(p["naturalness"]))
        if "factual" in p:    factual.append(int(p["factual"]))
        if "complete" in p:   complete.append(int(p["complete"]))
    def _s(lst: List[int]) -> Dict[str, float]:
        if not lst: return {"mean": float("nan"), "n": 0}
        return {"mean": float(np.mean(lst)), "sd": float(np.std(lst)), "n": len(lst)}
    return {
        "n_pairs": len(results),
        "n_failed_parse": failed,
        "meaning_preservation": _s(meaning),
        "naturalness": _s(natural),
        "factual_consistency_rate": float(np.mean(factual)) if factual else float("nan"),
        "no_omission_rate": float(np.mean(complete)) if complete else float("nan"),
    }


# ---------------------------------------------------------------------------
# Phase 2: answer-level judge
# ---------------------------------------------------------------------------
def load_answer_trials(
    trial_logs_dir: Path,
    manifest_path: Path,
    corpus_path: Path,
    n_per_target: int,
    targets: List[int],
    seed: int,
    trial_seed: int = 1,
) -> List[Dict[str, Any]]:
    """
    Load trials from trial logs, attaching source texts for retrieved docs.
    """
    manifest = {
        r["id"]: r["text_src"]
        for r in (json.loads(l) for l in manifest_path.open("r", encoding="utf-8"))
    }

    selected: List[Dict[str, Any]] = []
    rng = np.random.RandomState(seed)
    for target in targets:
        path = trial_logs_dir / f"rag_multi_t{target}_seed{trial_seed:02d}.trials.jsonl"
        if not path.exists():
            print(f"  [WARN] Missing: {path}")
            continue
        rows = [json.loads(l) for l in path.open("r", encoding="utf-8")]
        # Filter to trials where we have source text for ALL retrieved docs
        ok_rows = [
            r for r in rows
            if all(d["doc_id"] in manifest for d in r.get("retrieved_docs", []))
            and r.get("retrieved_docs")
        ]
        rng.shuffle(ok_rows)
        for row in ok_rows[:n_per_target]:
            src_docs = [
                f"[Document {i+1}]\n{manifest[d['doc_id']]}"
                for i, d in enumerate(row["retrieved_docs"])
            ]
            selected.append({
                "target": target,
                "question": row["question"],
                "wm_answer": row["answer"],
                "source_docs_text": "\n\n".join(src_docs),
                "retrieved_doc_ids": [d["doc_id"] for d in row["retrieved_docs"]],
            })
    return selected


def summarise_answers(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    wm_scores: Dict[str, List[float]] = {k: [] for k in ["relevance","accuracy","completeness","fluency"]}
    src_scores: Dict[str, List[float]] = {k: [] for k in ["relevance","accuracy","completeness","fluency"]}
    prefs = {"watermarked": 0, "source": 0, "equivalent": 0, "failed": 0}
    failed = 0

    for r in results:
        p = r.get("parsed")
        if not p:
            failed += 1
            prefs["failed"] += 1
            continue
        flip = r["flipped"]
        # A = watermarked if not flipped
        wm_key, src_key = ("a", "b") if not flip else ("b", "a")
        for dim in ["relevance","accuracy","completeness","fluency"]:
            wm_val = p.get(f"{wm_key}_{dim}")
            sr_val = p.get(f"{src_key}_{dim}")
            if wm_val is not None: wm_scores[dim].append(float(wm_val))
            if sr_val is not None: src_scores[dim].append(float(sr_val))
        pref = r.get("preference_label", "equivalent")
        prefs[pref] = prefs.get(pref, 0) + 1

    def _s(lst): return {"mean": float(np.mean(lst)), "n": len(lst)} if lst else {"mean": float("nan"), "n": 0}

    return {
        "n_trials": len(results),
        "n_failed_parse": failed,
        "watermarked_scores": {k: _s(v) for k, v in wm_scores.items()},
        "source_scores": {k: _s(v) for k, v in src_scores.items()},
        "preference_counts": prefs,
        "preference_rate_watermarked": prefs["watermarked"] / max(1, len(results)-failed),
        "preference_rate_source": prefs["source"] / max(1, len(results)-failed),
        "preference_rate_equivalent": prefs["equivalent"] / max(1, len(results)-failed),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_log", type=Path,
                    default=Path("results/watermark_cost_ablation/candidates_debug.jsonl"))
    ap.add_argument("--trial_logs_dir", type=Path,
                    default=Path("results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("work/farad_shared_manifest_p6_d400_seed13.jsonl"))
    ap.add_argument("--corpus", type=Path,
                    default=Path("work/farad_dirbucket_naive/corpus.jsonl"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/llm_judge_utility"))
    ap.add_argument("--n_sent", type=int, default=100)
    ap.add_argument("--n_per_target", type=int, default=10)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trial_seed", type=int, default=1,
                    help="Primary-run seed whose trial logs feed the answer phase.")
    ap.add_argument("--source_answer_model", type=str, default=None,
                    help="Model that regenerates the source-corpus answer (default: --model, "
                         "as in the paper run; gpt-4o-mini makes the comparison symmetric).")
    return ap.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI
    from tqdm import tqdm
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[FATAL] OPENAI_API_KEY is not set.")
    client = AsyncOpenAI()
    source_model = args.source_answer_model or args.model
    sem = asyncio.Semaphore(args.concurrency)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1 ----
    print(f"[PHASE 1] Loading {args.n_sent} sentence pairs from debug log...")
    pairs = load_sentence_pairs(args.debug_log, args.n_sent, args.seed)
    print(f"  Loaded {len(pairs)} pairs (from {args.n_sent} requested)")

    print(f"[PHASE 1] Running sentence judge ({len(pairs)} pairs)...")
    t0 = time.perf_counter()
    sent_results = []

    async def _judge_sent(p):
        prompt = SENTENCE_JUDGE_PROMPT.format(source=p["source"], watermarked=p["watermarked"])
        raw = await _achat(client, prompt, args.model, sem)
        parsed = _parse_json_safe(raw)
        return {"pair": p, "raw": raw, "parsed": parsed}

    tasks = [_judge_sent(p) for p in pairs]
    pbar = tqdm(total=len(tasks), desc="Sentence judge")
    for coro in asyncio.as_completed(tasks):
        sent_results.append(await coro)
        pbar.update(1)
    pbar.close()
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    sent_summary = summarise_sentence(sent_results)
    print(f"  Meaning: {sent_summary['meaning_preservation']['mean']:.2f}/5  "
          f"Naturalness: {sent_summary['naturalness']['mean']:.2f}/5  "
          f"Factual: {sent_summary['factual_consistency_rate']*100:.0f}%  "
          f"Complete: {sent_summary['no_omission_rate']*100:.0f}%")

    # ---- Phase 2 ----
    print(f"\n[PHASE 2] Loading answer trials...")
    targets = list(range(6))
    trials = load_answer_trials(
        args.trial_logs_dir, args.manifest, args.corpus,
        args.n_per_target, targets, args.seed, trial_seed=args.trial_seed,
    )
    print(f"  Loaded {len(trials)} trials across {len(targets)} providers")

    print(f"[PHASE 2] Generating source-corpus answers + judging ({len(trials)} trials)...")
    t1 = time.perf_counter()
    ans_results = []

    rng = random.Random(args.seed)
    ans_pbar = tqdm(total=len(trials), desc="Answer judge")

    async def _process_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
        # Generate source-corpus answer
        gen_prompt = ANSWER_GEN_PROMPT.format(
            docs=trial["source_docs_text"], question=trial["question"]
        )
        src_answer = await _achat(client, gen_prompt, source_model, sem, temperature=0.55)

        # Blind A/B
        flip = rng.random() < 0.5
        a_text = trial["wm_answer"] if not flip else src_answer
        b_text = src_answer if not flip else trial["wm_answer"]
        label_a = "watermarked" if not flip else "source"

        judge_p = ANSWER_JUDGE_PROMPT.format(
            question=trial["question"], answer_a=a_text, answer_b=b_text
        )
        raw_j = await _achat(client, judge_p, args.model, sem, temperature=0.0)
        parsed_j = _parse_json_safe(raw_j)

        pref_raw = (parsed_j or {}).get("preference", "equivalent")
        if pref_raw == "A":   pref = label_a
        elif pref_raw == "B": pref = "source" if label_a == "watermarked" else "watermarked"
        else:                  pref = "equivalent"

        return {
            "trial_meta": {"target": trial["target"], "question": trial["question"][:80]},
            "src_answer_excerpt": src_answer[:150],
            "wm_answer_excerpt": trial["wm_answer"][:150],
            "flipped": flip,
            "preference_label": pref,
            "parsed": parsed_j,
        }

    trial_tasks = [_process_trial(t) for t in trials]
    for coro in asyncio.as_completed(trial_tasks):
        ans_results.append(await coro)
        ans_pbar.update(1)
    ans_pbar.close()
    print(f"  Done in {time.perf_counter()-t1:.1f}s")

    ans_summary = summarise_answers(ans_results)
    pc = ans_summary["preference_counts"]
    print(f"  WM scores: "
          f"rel={ans_summary['watermarked_scores']['relevance']['mean']:.2f} "
          f"acc={ans_summary['watermarked_scores']['accuracy']['mean']:.2f} "
          f"cmp={ans_summary['watermarked_scores']['completeness']['mean']:.2f} "
          f"flu={ans_summary['watermarked_scores']['fluency']['mean']:.2f}")
    print(f"  Preferences: WM={pc['watermarked']} Src={pc['source']} Eq={pc['equivalent']}")

    # Save raw results
    raw_out = args.out_dir / "judge_results.json"
    raw_out.write_text(json.dumps({
        "config": {"judge_model": args.model, "source_answer_model": source_model,
                   "n_sent": args.n_sent, "n_per_target": args.n_per_target,
                   "seed": args.seed, "trial_seed": args.trial_seed},
        "sentence_summary": sent_summary,
        "answer_summary": ans_summary,
        "sentence_results": [{"pair": r["pair"], "parsed": r["parsed"]} for r in sent_results],
        "answer_results": ans_results,
    }, indent=2), encoding="utf-8")
    print(f"\n[WRITE] {raw_out}")

    print("\nDone.")


def main() -> None:
    args = _parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
