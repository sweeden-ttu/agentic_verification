import json
from pathlib import Path

from .store import write_json


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return f"{round(100 * x)}%"


def build(run_dir, specs, cfg):
    run_dir = Path(run_dir)
    mcq = _load(run_dir / "mcq_analysis.json")
    rec = _load(run_dir / "recurse_summary.json")
    dbn = _load(run_dir / "dbn_summary.json") or []
    pl = _load(run_dir / "prolog_results.json") or {}
    ranks = {s["id"]: s["rank"] for s in specs}
    md = ["# Boundary analysis", "", f"Root question (verbatim): \"{cfg['root_question']}\"", ""]
    md += [
        "Every quantity below splits the same total surprise two ways: within-model noise (the average entropy of each model's own answers) "
        "and between-model divergence (the mutual information between the answer and which model gave it). Only the second part is "
        "meaningful surprise; boundary items and boundary tokens are where it is large and the Bayes factor says the models really differ.",
        "",
    ]
    if mcq:
        t = mcq["totals"]
        md += ["## Legal and banned: the same question to every model", "",
               f"{len(mcq['items'])} statements. Total surprise {t['h_total_bits']:.1f} bits = within-model noise {t['h_within_bits']:.1f} "
               f"+ between-model divergence {t['mi_bits']:.1f} ({pct(t['meaningful_share'])} meaningful). Boundary items: {len(t['boundary_items'])}.", ""]
        md += ["| Model | Rank | A legal, not banned | B legal, but banned | C illegal, banned | D illegal, not banned | Mean P(legal) | Mean P(banned) | Banned read as illegal | Anchor accuracy |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for m in sorted(mcq["summary"], key=lambda x: -ranks.get(x, 0)):
            s = mcq["summary"][m]
            qd = s["quadrants"]
            md.append(f"| {m} | {ranks.get(m, '')} | {qd['A']} | {qd['B']} | {qd['C']} | {qd['D']} | {s['mean_p_legal']:.2f} | {s['mean_p_banned']:.2f} | "
                      f"{pct(s['conflation_banned_as_illegal'])} | {pct(s['anchor_accuracy'])} |")
        md.append("")
        top = sorted(mcq["items"], key=lambda r: -r["mi"])[:12]
        order = sorted(mcq["models"], key=lambda x: -ranks.get(x, 0))
        md += ["### Statements with the most between-model divergence", "",
               "| Item | Kind | Divergence (bits) | log BF | Axis | " + " | ".join(order) + " | Statement |",
               "|---|---|---|---|---|" + "---|" * len(order) + "---|"]
        for r in top:
            cells = [r["models"][m]["letter"] if m in r["models"] else "" for m in order]
            md.append(f"| {r['item']} | {r['kind']} | {r['mi']:.2f} | {r['log_bf']:.1f} | {r['axis']} | " + " | ".join(cells) + f" | {r['text']} |")
        md.append("")
        if pl.get("tier_pivot"):
            md += ["### Size pivots (larger models on one side, smaller on the other)", ""]
            for i, axis, rk in pl["tier_pivot"]:
                md.append(f"- {i} on the {axis} axis splits at rank {rk}")
            md.append("")
        if pl.get("universal"):
            md += ["### Universal answers (every model agrees)", "", ", ".join(f"{i}={qd.upper()}" for i, qd in pl["universal"]), ""]
    if rec:
        md += ["## Recursion of the root question", "", f"Stopped: {rec['stop_reason']}.", "",
               "| Level | Vocabulary | Total bits | Noise bits | Divergence bits | Meaningful share | Jaccard to previous | Top boundary tokens |",
               "|---|---|---|---|---|---|---|---|"]
        for lv in rec["levels"]:
            d = lv["decomposition"]
            j = "" if lv["jaccard_prev"] is None else f"{lv['jaccard_prev']:.2f}"
            md.append(f"| {lv['level']} | {lv['vocab_size']} | {d['h_total_bits']:.1f} | {d['h_within_bits']:.1f} | {d['mi_bits']:.1f} | "
                      f"{pct(d['meaningful_share'])} | {j} | {', '.join(lv['boundary'][:8])} |")
        md.append("")
        last = rec["levels"][-1]["level"] if rec["levels"] else None
        if last is not None:
            an = _load(run_dir / "recurse" / f"level_{last}" / "analysis.json")
            md += [f"### Boundary tokens at level {last}", "", "| Token | Divergence (bits) | log BF | " + " | ".join(an["models"]) + " | Sense |",
                   "|---|---|---|" + "---|" * len(an["models"]) + "---|"]
            for t in an["boundary"][:15]:
                tk = an["tokens"][t]
                md.append(f"| {t} | {tk['mi']:.2f} | {tk['log_bf']:.1f} | " + " | ".join(pct(tk["rates"][m]) for m in an["models"]) +
                          f" | {', '.join(tk['senses'])} |")
            md += ["", "Signature tokens (used more by this model than by any other, ranked by the surprise they cause the others):", ""]
            for m, sig in an["signatures"].items():
                md.append(f"- {m}: {', '.join(sig)}")
            md.append("")
        for key, title in (("fixed_point_token", "Fixed-point tokens (boundary at the last two levels)"),
                           ("persistent_token", "Tokens that stay on the boundary from one level to the next"),
                           ("axis_token", "Boundary tokens on the legal, banned or ambiguous axis")):
            if pl.get(key):
                md += [f"### {title}", "", ", ".join(" ".join(x) for x in pl[key]), ""]
    if dbn:
        md += ["## Deep belief network (stacked RBMs over token presence)", "",
               "Exploratory: with a few answers per model, the RBM units are unstable; read them next to the Bayesian boundary tokens, not instead of them.", ""]
        for d in dbn:
            u = d["units"][0]
            md.append(f"- Level {d['level']}: layers {d['layers']}, most model-specific unit carries {u['mi_bits']:.2f} of "
                      f"{d['max_mi_possible_bits']:.2f} bits; its top terms: {', '.join(u['top_terms'])}; overlap with Bayesian boundary: "
                      f"{', '.join(d['overlap_with_bayes_boundary']) or 'none'}")
        md.append("")
    (run_dir / "report.md").write_text("\n".join(md))
    write_json(run_dir / "report.json", {"mcq": mcq and {"totals": mcq["totals"], "summary": mcq["summary"]}, "recurse": rec,
                                         "dbn": dbn, "prolog": pl})
    return run_dir / "report.md"
