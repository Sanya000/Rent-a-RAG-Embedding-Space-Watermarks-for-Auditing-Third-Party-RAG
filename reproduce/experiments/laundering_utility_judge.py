#!/usr/bin/env python3
"""LLM-judge utility of laundered answers (paper App. app:laundering-utility, Table tab:laundering-utility).

For each laundering condition, the first ``--n_per_pair`` trials (by trial_idx)
of every (seed, target) primary trial log are looked up in the laundering
answer cache (``post_answer_laundering_attack.py``) and a judge rates the
laundered answer against the original answer for accuracy, completeness and
fluency (1-5) plus a preference. Only the four laundering conditions are
judged; "original" is the reference, not a rated condition. Judgements are
requested sequentially; unparseable judge outputs are dropped.

Paper run: ``--cache_dir results/laundering_attack_seeds45/answer_cache
--seeds 4,5 --targets 0,1,2,3,4,5 --n_per_pair 10 --judge_model gpt-4o``
(2 x 6 x 10 = 120 judgements per condition).

Output: <out_dir>/utility_judge_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List


def cache_key(question: str, answer: str, mode: str = "question") -> str:
    """Same keying as post_answer_laundering_attack.cache_key."""
    text = question if mode == "question" else question + "\x1f" + answer
    return hashlib.sha256(text.encode()).hexdigest()[:16]


JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of a RAG system answer.

Question: {question}

Reference answer (original): {reference}

Answer to evaluate: {answer}

Rate the answer-to-evaluate on:
- accuracy (1-5): are all stated facts correct relative to the reference?
- completeness (1-5): does it cover the key information in the reference?
- fluency (1-5): is it well-written and easy to read?

Also give an overall preference: "reference_better", "evaluate_better", or "equivalent".

Respond in JSON:
{{"accuracy": <1-5>, "completeness": <1-5>, "fluency": <1-5>, "preference": "<choice>", "reason": "<one sentence>"}}
"""


async def _achat(client, prompt, model, sem, temperature=0.0, max_tokens=200):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"})
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == 2:
                    return ""
                await asyncio.sleep(2 ** attempt)
    return ""


async def judge_condition(
    condition: str,
    cache_dir: Path,
    trial_logs_dir: Path,
    seeds: List[int],
    targets: List[int],
    n_per_pair: int,
    client,
    model: str,
    sem,
    key_mode: str = "question",
) -> Dict[str, Any]:
    cond_cache = cache_dir / condition
    if not cond_cache.exists():
        print(f"  [SKIP] no cache for {condition}")
        return {}

    results = []
    for seed in seeds:
        for target in targets:
            log_path = trial_logs_dir / f"rag_multi_t{target}_seed{seed:02d}.trials.jsonl"
            if not log_path.exists():
                continue
            trials = sorted([json.loads(l) for l in log_path.open()],
                            key=lambda r: int(r.get("trial_idx", 0)))[:n_per_pair]

            for trial in trials:
                q = trial["question"]
                orig_ans = (trial.get("answer") or "").strip()
                if not orig_ans:
                    continue
                q_hash = cache_key(q, trial.get("answer") or "", key_mode)
                cached = cond_cache / f"{q_hash}.txt"
                if not cached.exists():
                    continue
                laundered = cached.read_text(encoding="utf-8").strip()
                if not laundered:
                    continue

                prompt = JUDGE_PROMPT.format(
                    question=q,
                    reference=orig_ans[:1000],
                    answer=laundered[:1000])
                raw = await _achat(client, prompt, model, sem)
                try:
                    parsed = json.loads(raw)
                    parsed["condition"] = condition
                    parsed["seed"] = seed
                    parsed["target"] = target
                    parsed["orig_words"] = len(orig_ans.split())
                    parsed["laundered_words"] = len(laundered.split())
                    results.append(parsed)
                except Exception:
                    continue

    if not results:
        return {}

    acc = [r.get("accuracy", 0) for r in results]
    comp = [r.get("completeness", 0) for r in results]
    flu = [r.get("fluency", 0) for r in results]
    orig_w = [r.get("orig_words", 0) for r in results]
    lau_w = [r.get("laundered_words", 0) for r in results]
    prefs = [r.get("preference", "") for r in results]

    return {
        "condition": condition,
        "n": len(results),
        "accuracy_mean": statistics.mean(acc),
        "completeness_mean": statistics.mean(comp),
        "fluency_mean": statistics.mean(flu),
        "orig_words_median": statistics.median(orig_w),
        "laundered_words_median": statistics.median(lau_w),
        "word_ratio": statistics.mean(lw/ow if ow else 0 for ow, lw in zip(orig_w, lau_w)),
        "pref_orig": prefs.count("reference_better") / len(prefs),
        "pref_equal": prefs.count("equivalent") / len(prefs),
        "pref_laundered": prefs.count("evaluate_better") / len(prefs),
        "per_result": results,
    }


async def main_async(args):
    from openai import AsyncOpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[FATAL] OPENAI_API_KEY is not set.")
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    seeds = [int(s) for s in args.seeds.split(",")]
    targets = [int(t) for t in args.targets.split(",")]

    all_results = {}
    for cond in args.conditions.split(","):
        print(f"[JUDGE] {cond} ...", flush=True)
        r = await judge_condition(
            condition=cond,
            cache_dir=args.cache_dir,
            trial_logs_dir=args.trial_logs_dir,
            seeds=seeds,
            targets=targets,
            n_per_pair=args.n_per_pair,
            client=client,
            model=args.judge_model,
            sem=sem,
            key_mode=args.cache_key)
        if r:
            all_results[cond] = r
            print(f"  n={r['n']} acc={r['accuracy_mean']:.2f} "
                  f"comp={r['completeness_mean']:.2f} "
                  f"words={r['laundered_words_median']:.0f} "
                  f"(orig={r['orig_words_median']:.0f}, ratio={r['word_ratio']:.2f})",
                  flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "utility_judge_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n[WRITE] {out}")

    # Print summary table
    print("\n=== Adversary's Dilemma Summary ===")
    print(f"{'Condition':<20} {'Words':>6} {'Ratio':>6} {'Acc':>5} {'Comp':>5} {'Pref:Orig':>9}")
    for cond, r in all_results.items():
        print(f"{cond:<20} {r['laundered_words_median']:>6.0f} "
              f"{r['word_ratio']:>6.2f} {r['accuracy_mean']:>5.2f} "
              f"{r['completeness_mean']:>5.2f} {r['pref_orig']:>9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path,
                    default=Path("results/laundering_attack_seeds45/answer_cache"))
    ap.add_argument("--trial_logs_dir", type=Path,
                    default=Path("results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive"))
    ap.add_argument("--conditions", type=str,
                    default="neutral,aggressive,compress50,compress25")
    ap.add_argument("--seeds", type=str, default="4,5")
    ap.add_argument("--targets", type=str, default="0,1,2,3")
    ap.add_argument("--n_per_pair", type=int, default=15,
                    help="Number of trials per (seed, target) pair to judge")
    ap.add_argument("--judge_model", type=str, default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--out_dir", type=Path, default=Path("results/laundering_attack"))
    ap.add_argument("--cache_key", choices=["question", "trial"], default="question",
                    help="Must match the --cache_key used by post_answer_laundering_attack.py.")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
