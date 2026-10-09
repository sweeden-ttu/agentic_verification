import json
import math
import re
from pathlib import Path

from . import bayes
from .store import read_jsonl, run_jobs, write_json

LETTERS = "ABCD"
LETTER_SET = set(LETTERS)
STRIP = " *_().:\n\t\"'`"


def load_items(path):
    return json.loads(Path(path).read_text())["items"]


def first_letter(text):
    m = re.search(r"\b([ABCD])\b", (text or "")[:400])
    return m.group(1) if m else None


def dist_from_logprobs(lp):
    for t in lp or []:
        if t["token"].strip(STRIP) in LETTER_SET:
            mass = {L: 0.0 for L in LETTERS}
            for alt, logp in t["top"]:
                a = alt.strip(STRIP)
                if a in LETTER_SET:
                    mass[a] += math.exp(logp)
            total = sum(mass.values())
            if total <= 0:
                return None, 0.0
            return {L: mass[L] / total for L in LETTERS}, total
    return None, 0.0


def run(run_dir, specs, items, cfg):
    mc = cfg["mcq"]
    by_id = {it["id"]: it for it in items}
    for spec in specs:
        use_lp = bool(spec.get("logprobs"))
        if use_lp:
            jobs = [{"item": it["id"], "sample": "lp"} for it in items]
        else:
            jobs = [{"item": it["id"], "sample": s} for it in items for s in range(mc["k"])]

        def call(client, job):
            prompt = mc["template"].format(statement=by_id[job["item"]]["text"])
            if job["sample"] == "lp":
                r = client.chat(prompt, min(4, mc["max_tokens"]), mc["temperature"], logprobs=True)
                dist, mass = dist_from_logprobs(r["logprobs"])
                if dist is not None and mass >= 0.5:
                    return {"text": r["text"][:200], "dist": dist, "mass": mass, "letter": max(dist, key=dist.get)}
                letters = [first_letter(client.chat(prompt, mc["max_tokens"], mc["temperature"])["text"]) for _ in range(mc["k"])]
                return {"text": r["text"][:200], "letters": letters, "mass": mass}
            r = client.chat(prompt, mc["max_tokens"], mc["temperature"])
            return {"text": r["text"][:200], "letter": first_letter(r["text"])}

        print(f"mcq: {spec['id']} ({len(jobs)} calls)", flush=True)
        run_jobs(spec, jobs, call, Path(run_dir) / "mcq" / f"{spec['id']}.jsonl", key=lambda r: (r["item"], r["sample"]))


def counts_for(records, pseudocount):
    out = {}
    invalid = {}
    for r in records:
        c = out.setdefault(r["item"], [0.0, 0.0, 0.0, 0.0])
        if r.get("dist"):
            for i, L in enumerate(LETTERS):
                c[i] += r["dist"][L] * pseudocount
            continue
        letters = r.get("letters", [r.get("letter")])
        for L in letters:
            if L in LETTER_SET:
                c[LETTERS.index(L)] += 1
            else:
                invalid[r["item"]] = invalid.get(r["item"], 0) + 1
    return out, invalid


def analyze(run_dir, specs, items, cfg):
    bc = cfg["bayes"]
    alpha = bc["alpha"]
    data, invalid = {}, {}
    for spec in specs:
        recs = read_jsonl(Path(run_dir) / "mcq" / f"{spec['id']}.jsonl")
        if recs:
            data[spec["id"]], invalid[spec["id"]] = counts_for(recs, cfg["mcq"]["logprob_pseudocount"])
    models = [s["id"] for s in specs if s["id"] in data]
    rows = []
    for it in items:
        cm = {m: data[m][it["id"]] for m in models if it["id"] in data[m] and sum(data[m][it["id"]]) > 0}
        if len(cm) < 2:
            continue
        full = bayes.decompose(cm, alpha)
        lg = bayes.decompose({m: [c[0] + c[1], c[2] + c[3]] for m, c in cm.items()}, alpha)
        bn = bayes.decompose({m: [c[1] + c[2], c[0] + c[3]] for m, c in cm.items()}, alpha)
        per = {}
        for m in cm:
            p = full["post"][m]
            per[m] = {
                "p": p,
                "letter": LETTERS[max(range(4), key=lambda i: p[i])],
                "p_legal": lg["post"][m][0],
                "p_banned": bn["post"][m][0],
                "surprise_bits": full["surprise"][m],
                "n": sum(cm[m]),
            }
        rows.append({
            "item": it["id"],
            "kind": it["kind"],
            "expected": it.get("expected"),
            "text": it["text"],
            "h_total": full["h_total"],
            "h_within": full["h_within"],
            "mi": full["mi"],
            "meaningful_share": full["meaningful_share"],
            "log_bf": full["log_bf"],
            "mi_legal": lg["mi"],
            "mi_banned": bn["mi"],
            "axis": "legal" if lg["mi"] >= bn["mi"] else "banned",
            "boundary": full["mi"] >= bc["item_jsd_min_bits"] and full["log_bf"] >= bc["log_bf_min"],
            "models": per,
        })
    summary = {}
    for m in models:
        mine = [r for r in rows if m in r["models"]]
        quad = {L: 0 for L in LETTERS}
        for r in mine:
            quad[r["models"][m]["letter"]] += 1
        exp_b = [r for r in mine if r["expected"] == "b"]
        anchors = [r for r in mine if r["expected"] in ("a", "b")]
        summary[m] = {
            "items": len(mine),
            "quadrants": quad,
            "mean_p_legal": sum(r["models"][m]["p_legal"] for r in mine) / max(1, len(mine)),
            "mean_p_banned": sum(r["models"][m]["p_banned"] for r in mine) / max(1, len(mine)),
            "conflation_banned_as_illegal": sum(r["models"][m]["letter"] == "C" for r in exp_b) / max(1, len(exp_b)),
            "anchor_accuracy": sum(r["models"][m]["letter"].lower() == r["expected"] for r in anchors) / max(1, len(anchors)),
            "invalid_answers": sum(invalid.get(m, {}).values()),
        }
    tot = sum(r["h_total"] for r in rows)
    out = {
        "models": models,
        "items": rows,
        "summary": summary,
        "totals": {
            "h_total_bits": tot,
            "h_within_bits": sum(r["h_within"] for r in rows),
            "mi_bits": sum(r["mi"] for r in rows),
            "meaningful_share": sum(r["mi"] for r in rows) / tot if tot else 0.0,
            "boundary_items": [r["item"] for r in sorted(rows, key=lambda r: -r["mi"]) if r["boundary"]],
        },
    }
    write_json(Path(run_dir) / "mcq_analysis.json", out)
    return out
