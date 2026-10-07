import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .providers import Session


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def run_jobs(spec, jobs, call, out_path, key):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {key(r) for r in read_jsonl(out_path)}
    todo = [j for j in jobs if key(j) not in done]
    if not todo:
        return 0
    lock = threading.Lock()
    err_path = out_path.with_suffix(".errors.jsonl")
    t0 = time.time()
    n_ok = 0
    with Session(spec) as client:
        def work(job):
            rec = call(client, job)
            rec.update({k: v for k, v in job.items() if k not in rec})
            rec["model"] = spec["id"]
            return rec

        with ThreadPoolExecutor(max_workers=max(1, spec.get("concurrency", 1))) as ex:
            futs = {ex.submit(work, j): j for j in todo}
            for i, f in enumerate(as_completed(futs), 1):
                job = futs[f]
                try:
                    rec = f.result()
                    with lock, out_path.open("a") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1
                except Exception as e:
                    with lock, err_path.open("a") as fh:
                        fh.write(json.dumps({"job": job, "error": str(e)[:500]}) + "\n")
                if i % 5 == 0 or i == len(todo):
                    print(f"  {spec['id']}: {i}/{len(todo)} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return n_ok
