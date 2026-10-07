import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from .providers import ROOT
from .store import write_json


def q(s):
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ") + "'"


def f(x):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        x = 0.0
    return f"{x:.6f}"


def swipl_path():
    local = ROOT / ".tools" / "env" / "bin" / "swipl"
    if local.exists():
        return str(local)
    found = shutil.which("swipl")
    if found:
        return found
    raise RuntimeError("SWI-Prolog not found: run scripts/install_prolog.sh")


def build(run_dir, specs, items, cfg):
    run_dir = Path(run_dir)
    lines = [":- discontiguous model/2, item/3, answer/6, axis_p/4, item_div/5, level/2, token_stat/5, token_rate/4, boundary_token/2, lexicon/2, dbn_unit/4, threshold/2."]
    bc = cfg["bayes"]
    lines.append(f"threshold(item_jsd, {f(bc['item_jsd_min_bits'])}).")
    lines.append(f"threshold(log_bf, {f(bc['log_bf_min'])}).")
    lines.append(f"threshold(token_mi, {f(bc['mi_min_bits'])}).")
    for s in specs:
        lines.append(f"model({q(s['id'])}, {int(s['rank'])}).")
    for it in items:
        lines.append(f"item({q(it['id'])}, {q(it['kind'])}, {q(it.get('expected') or 'none')}).")
    mcq_path = run_dir / "mcq_analysis.json"
    if mcq_path.exists():
        an = json.loads(mcq_path.read_text())
        for r in an["items"]:
            lines.append(f"item_div({q(r['item'])}, {f(r['mi'])}, {f(r['log_bf'])}, {f(r['mi_legal'])}, {f(r['mi_banned'])}).")
            for m, pm in r["models"].items():
                p = pm["p"]
                lines.append(f"answer({q(m)}, {q(r['item'])}, {f(p[0])}, {f(p[1])}, {f(p[2])}, {f(p[3])}).")
                lines.append(f"axis_p({q(m)}, {q(r['item'])}, {f(pm['p_legal'])}, {f(pm['p_banned'])}).")
    rdir = run_dir / "recurse"
    if rdir.exists():
        for ldir in sorted(rdir.glob("level_*"), key=lambda p: int(p.name.split("_")[1])):
            ap = ldir / "analysis.json"
            if not ap.exists():
                continue
            an = json.loads(ap.read_text())
            L = an["level"]
            lines.append(f"level({L}, {f(an['decomposition']['meaningful_share'])}).")
            for t in an["boundary"]:
                tk = an["tokens"][t]
                lines.append(f"boundary_token({L}, {q(t)}).")
                lines.append(f"token_stat({L}, {q(t)}, {f(tk['mi'])}, {f(tk['log_bf'])}, {int(tk['support'])}).")
                for m, r in tk["rates"].items():
                    lines.append(f"token_rate({L}, {q(m)}, {q(t)}, {f(r)}).")
                for sense in tk["senses"]:
                    lines.append(f"lexicon({q(t)}, {q(sense)}).")
            dp = ldir / "dbn.json"
            if dp.exists():
                dn = json.loads(dp.read_text())
                for u in dn["units"][:3]:
                    terms = "[" + ", ".join(q(t) for t in u["top_terms"]) + "]"
                    lines.append(f"dbn_unit({L}, {u['unit']}, {f(u['mi_bits'])}, {terms}).")
    facts = run_dir / "facts.pl"
    facts.write_text("\n".join(lines) + "\n")
    return facts


def query(run_dir):
    run_dir = Path(run_dir)
    rules = ROOT / "prolog" / "horizon.pl"
    facts = run_dir / "facts.pl"
    cmd = [swipl_path(), "-q", "-g", f"consult('{rules}'), consult('{facts}'), report", "-t", "halt"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split("\t")
        out.setdefault(parts[0], []).append(parts[1:])
    write_json(run_dir / "prolog_results.json", out)
    return out
