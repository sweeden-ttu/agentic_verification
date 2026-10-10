import json
import random
from collections import Counter
from pathlib import Path

from . import bayes, terms
from .store import read_jsonl, run_jobs, write_json


def labels_for(model_ids, seed):
    order = list(model_ids)
    random.Random(seed).shuffle(order)
    return {m: f"M{i + 1}" for i, m in enumerate(order)}


def mcq_context(mcq_an, labels, top_items):
    rows = sorted(mcq_an["items"], key=lambda r: -r["mi"])[:top_items]
    if not rows:
        return ""
    lines = [
        "The same multiple-choice question (A legal, not banned; B legal, but banned; C illegal, and banned; "
        "D illegal, but not banned) was put to several anonymous models about statements from the Gemma 4 "
        "Developer Agent competition. Their answers diverged most on these statements:"
    ]
    for r in rows:
        parts = []
        for m in sorted(labels, key=lambda x: labels[x]):
            if m in r["models"]:
                pm = r["models"][m]
                parts.append(f"{labels[m]} {pm['letter']} ({round(100 * max(pm['p']))}%)")
        lines.append(f"- \"{r['text']}\": " + ", ".join(parts))
    return "\n".join(lines)


def level_context(prev, samples, labels, top_tokens, excerpt_chars, rng):
    lines = [
        "The question at the end was put to several anonymous models. Their answers diverged. These tokens "
        "carried the most information about which model wrote an answer (share of each model's answers that use the token):"
    ]
    for t in prev["boundary"][:top_tokens]:
        rates = prev["tokens"][t]["rates"]
        parts = [f"{labels[m]} {round(100 * rates[m])}%" for m in sorted(labels, key=lambda x: labels[x]) if m in rates]
        lines.append(f"- \"{t}\": " + ", ".join(parts))
    lines.append("One answer excerpt per model:")
    for m in sorted(labels, key=lambda x: labels[x]):
        if samples.get(m):
            text = " ".join(rng.choice(samples[m]).split())[:excerpt_chars]
            lines.append(f"{labels[m]}: \"{text}\"")
    return "\n".join(lines)


def analyze_level(samples, prompt, cfg, lexicon):
    bc = cfg["bayes"]
    sets = {m: [terms.units(t) for t in texts] for m, texts in samples.items() if texts}
    support = Counter(u for ss in sets.values() for s in ss for u in s)
    vocab = [u for u, c in support.items() if c >= bc["min_support"]]
    echo = terms.units(prompt)
    tokens = {}
    h_tot = h_in = mi_sum = 0.0
    for u in vocab:
        cm = {m: [sum(u in s for s in ss), len(ss) - sum(u in s for s in ss)] for m, ss in sets.items()}
        d = bayes.decompose(cm, bc["alpha"])
        tokens[u] = {
            "mi": d["mi"],
            "log_bf": d["log_bf"],
            "h_total": d["h_total"],
            "h_within": d["h_within"],
            "rates": {m: d["post"][m][0] for m in cm},
            "raw": {m: cm[m][0] for m in cm},
            "surprise": d["surprise"],
            "support": support[u],
            "echo": u in echo,
            "senses": terms.senses_for(u, lexicon),
        }
        h_tot += d["h_total"]
        h_in += d["h_within"]
        mi_sum += d["mi"]
    boundary = sorted(
        [u for u in vocab if tokens[u]["mi"] >= bc["mi_min_bits"] and tokens[u]["log_bf"] >= bc["log_bf_min"]],
        key=lambda u: (-tokens[u]["mi"], -tokens[u]["log_bf"], u),
    )[: bc["max_boundary_tokens"]]
    signatures = {}
    for m in sets:
        cand = [u for u in vocab if tokens[u]["rates"][m] > max(tokens[u]["rates"][o] for o in sets if o != m)] if len(sets) > 1 else []
        signatures[m] = sorted(cand, key=lambda u: -tokens[u]["surprise"][m])[:8]
    return {
        "models": list(sets),
        "n_samples": {m: len(ss) for m, ss in sets.items()},
        "vocab_size": len(vocab),
        "decomposition": {
            "h_total_bits": h_tot,
            "h_within_bits": h_in,
            "mi_bits": mi_sum,
            "meaningful_share": mi_sum / h_tot if h_tot else 0.0,
        },
        "boundary": boundary,
        "signatures": signatures,
        "tokens": tokens,
    }


def reanalyze(run_dir, specs, cfg, lexicon):
    run_dir = Path(run_dir)
    rdir = run_dir / "recurse"
    if not rdir.exists():
        return None
    ids = [s["id"] for s in specs]
    history = []
    prev = None
    for ldir in sorted(rdir.glob("level_*"), key=lambda p: int(p.name.split("_")[1])):
        if not (ldir / "prompt.txt").exists():
            continue
        level = int(ldir.name.split("_")[1])
        samples = {m: [r["text"] for r in read_jsonl(ldir / f"{m}.jsonl") if r.get("text")] for m in ids}
        if sum(1 for v in samples.values() if v) < 2:
            break
        an = analyze_level(samples, (ldir / "prompt.txt").read_text(), cfg, lexicon)
        an["level"] = level
        lp = ldir / "labels.json"
        an["labels"] = json.loads(lp.read_text()) if lp.exists() else {}
        an["jaccard_prev"] = bayes.jaccard(an["boundary"], prev["boundary"]) if prev else None
        write_json(ldir / "analysis.json", an)
        history.append({"level": level, "boundary": an["boundary"], "decomposition": an["decomposition"],
                        "jaccard_prev": an["jaccard_prev"], "vocab_size": an["vocab_size"]})
        prev = an
    old = run_dir / "recurse_summary.json"
    stop = json.loads(old.read_text())["stop_reason"] if old.exists() else "reanalyzed"
    out = {"levels": history, "stop_reason": stop}
    write_json(old, out)
    return out


def run(run_dir, specs, cfg, lexicon, mcq_an=None):
    rc = cfg["recurse"]
    root = cfg["root_question"]
    run_dir = Path(run_dir)
    ids = [s["id"] for s in specs]
    prev = prev_samples = None
    history = []
    stop = "depth reached"
    for level in range(rc["depth"]):
        rng = random.Random(cfg["seed"] + 1000 + level)
        labels = labels_for(ids, cfg["seed"] + level)
        if level == 0:
            context = mcq_context(mcq_an, labels, rc["top_items"]) if (rc["seed_from"] == "mcq" and mcq_an) else ""
        else:
            context = level_context(prev, prev_samples, labels, rc["top_tokens"], rc["excerpt_chars"], rng)
        prompt = f"{context}\n\n{root}" if context else root
        ldir = run_dir / "recurse" / f"level_{level}"
        ldir.mkdir(parents=True, exist_ok=True)
        (ldir / "prompt.txt").write_text(prompt)
        write_json(ldir / "labels.json", labels)
        print(f"recurse level {level}: prompt {len(prompt)} chars", flush=True)
        for spec in specs:
            jobs = [{"level": level, "sample": s} for s in range(rc["k"])]

            def call(client, job, prompt=prompt):
                r = client.chat(prompt, rc["max_tokens"], rc["temperature"])
                return {"text": r["text"], "finish": r.get("finish")}

            run_jobs(spec, jobs, call, ldir / f"{spec['id']}.jsonl", key=lambda r: r["sample"])
        samples = {m: [r["text"] for r in read_jsonl(ldir / f"{m}.jsonl") if r.get("text")] for m in ids}
        an = analyze_level(samples, prompt, cfg, lexicon)
        an["level"] = level
        an["labels"] = labels
        an["jaccard_prev"] = bayes.jaccard(an["boundary"], prev["boundary"]) if prev else None
        write_json(ldir / "analysis.json", an)
        history.append({
            "level": level,
            "boundary": an["boundary"],
            "decomposition": an["decomposition"],
            "jaccard_prev": an["jaccard_prev"],
            "vocab_size": an["vocab_size"],
        })
        print(f"  level {level}: {len(an['boundary'])} boundary tokens, meaningful share "
              f"{an['decomposition']['meaningful_share']:.3f}, jaccard {an['jaccard_prev']}", flush=True)
        if not an["boundary"]:
            stop = "no boundary tokens"
            break
        if an["jaccard_prev"] is not None and an["jaccard_prev"] >= rc["fixed_point_jaccard"]:
            stop = "fixed point"
            break
        prev, prev_samples = an, samples
    out = {"levels": history, "stop_reason": stop}
    write_json(run_dir / "recurse_summary.json", out)
    return out
