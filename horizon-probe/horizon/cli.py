import argparse
import json
import time
from pathlib import Path

from . import dbn, facts, mcq, recurse, report
from .providers import ROOT, load_models
from .store import write_json


def _cfg(args):
    cfg = json.loads(Path(args.config).read_text())
    if args.k:
        cfg["mcq"]["k"] = args.k
        cfg["recurse"]["k"] = args.k
    if args.depth:
        cfg["recurse"]["depth"] = args.depth
    if args.max_tokens:
        cfg["recurse"]["max_tokens"] = args.max_tokens
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="horizon")
    ap.add_argument("command", choices=["mcq", "recurse", "analyze", "all"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--models-file", default=str(ROOT / "config" / "models.json"))
    ap.add_argument("--config", default=str(ROOT / "config" / "run.json"))
    ap.add_argument("--items", default=str(ROOT / "data" / "items.json"))
    ap.add_argument("--lexicon", default=str(ROOT / "data" / "lexicon.json"))
    ap.add_argument("--models", default=None)
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = _cfg(a)
    specs = load_models(a.models_file, a.models.split(",") if a.models else None)
    items = mcq.load_items(a.items)
    if a.limit_items:
        items = items[: a.limit_items]
    lexicon = json.loads(Path(a.lexicon).read_text())["senses"]
    run_dir = Path(a.run) if a.run else ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta.update({"models": [{k: v for k, v in s.items() if k != "launch"} for s in specs], "config": cfg,
                 "items_file": a.items, "n_items": len(items), "last_command": a.command})
    write_json(meta_path, meta)
    if a.command in ("mcq", "all"):
        mcq.run(run_dir, specs, items, cfg)
    mcq_an = None
    if a.command in ("mcq", "recurse", "analyze", "all") and (run_dir / "mcq").exists():
        mcq_an = mcq.analyze(run_dir, specs, items, cfg)
    if a.command in ("recurse", "all"):
        recurse.run(run_dir, specs, cfg, lexicon, mcq_an)
    if a.command == "analyze":
        recurse.reanalyze(run_dir, specs, cfg, lexicon)
    if a.command in ("analyze", "all", "recurse", "mcq"):
        dbn.run(run_dir, cfg)
        facts.build(run_dir, specs, items, cfg)
        facts.query(run_dir)
        path = report.build(run_dir, specs, cfg)
        print(f"report: {path}")


if __name__ == "__main__":
    main()
