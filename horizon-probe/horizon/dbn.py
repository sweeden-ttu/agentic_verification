import json
import math
from pathlib import Path

import numpy as np
from sklearn.neural_network import BernoulliRBM

from . import terms
from .store import read_jsonl, write_json


def mutual_info_bits(b, y):
    n = len(y)
    mi = 0.0
    for bv in set(b.tolist()):
        pb = float(np.mean(b == bv))
        for yv in set(y.tolist()):
            pyv = float(np.mean(y == yv))
            pj = float(np.mean((b == bv) & (y == yv)))
            if pj > 0:
                mi += pj * math.log2(pj / (pb * pyv))
    return mi


def level(run_dir, level_idx, cfg):
    dc = cfg["dbn"]
    ldir = Path(run_dir) / "recurse" / f"level_{level_idx}"
    an = json.loads((ldir / "analysis.json").read_text())
    models = an["models"]
    vocab = sorted(an["tokens"], key=lambda u: (-an["tokens"][u]["support"], u))[: dc["max_terms"]]
    idx = {u: i for i, u in enumerate(vocab)}
    rows, ys = [], []
    for mi_, m in enumerate(models):
        for r in read_jsonl(ldir / f"{m}.jsonl"):
            if not r.get("text"):
                continue
            row = np.zeros(len(vocab))
            for u in terms.units(r["text"]):
                if u in idx:
                    row[idx[u]] = 1.0
            rows.append(row)
            ys.append(mi_)
    if len(rows) < 4 or len(vocab) < 4 or len(set(ys)) < 2:
        return None
    X = np.array(rows)
    y = np.array(ys)
    h = X
    rbms = []
    for n_h in dc["layers"]:
        n_h = int(min(n_h, max(2, h.shape[1] // 2)))
        rbm = BernoulliRBM(n_components=n_h, learning_rate=dc["learning_rate"], n_iter=dc["n_iter"],
                           batch_size=min(10, X.shape[0]), random_state=dc["seed"])
        h = rbm.fit_transform(h)
        rbms.append(rbm)
    W = rbms[0].components_
    for r in rbms[1:]:
        W = r.components_ @ W
    units = []
    for j in range(h.shape[1]):
        col = h[:, j]
        b = (col > np.median(col)).astype(int) if col.max() > col.min() else np.zeros(len(col), dtype=int)
        top = [vocab[i] for i in np.argsort(-W[j])[:8]]
        means = {m: float(h[y == k, j].mean()) for k, m in enumerate(models)}
        units.append({"unit": j, "mi_bits": mutual_info_bits(b, y), "top_terms": top, "mean_activation": means})
    units.sort(key=lambda u: -u["mi_bits"])
    boundary = set(an["boundary"])
    top_terms = set(t for u in units[:2] for t in u["top_terms"])
    out = {
        "level": level_idx,
        "n_samples": int(X.shape[0]),
        "n_terms": len(vocab),
        "layers": [r.n_components for r in rbms],
        "max_mi_possible_bits": math.log2(len(models)),
        "units": units,
        "overlap_with_bayes_boundary": sorted(top_terms & boundary),
    }
    write_json(ldir / "dbn.json", out)
    return out


def run(run_dir, cfg):
    out = []
    rdir = Path(run_dir) / "recurse"
    if not rdir.exists():
        return out
    for ldir in sorted(rdir.glob("level_*"), key=lambda p: int(p.name.split("_")[1])):
        if (ldir / "analysis.json").exists():
            r = level(run_dir, int(ldir.name.split("_")[1]), cfg)
            if r:
                out.append(r)
    write_json(Path(run_dir) / "dbn_summary.json", out)
    return out
