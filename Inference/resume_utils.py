"""
Inference/resume_utils.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
import shutil
import socket
import subprocess
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


class MissingInput(Exception):
    pass


class IncompatibleArtifact(Exception):
    pass


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent(path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return parent


def write_json_atomic(path: str, obj) -> None:
    ensure_parent(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv_atomic(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_parent(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=index)
    with open(tmp, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_npz_atomic(path: str, **arrays) -> None:
    ensure_parent(path)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def write_torch_atomic(obj, path: str) -> None:
    import torch
    ensure_parent(path)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fingerprint_file(path: str) -> Optional[Dict]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {"path": os.path.abspath(path), "size": int(st.st_size),
            "mtime": round(float(st.st_mtime), 3)}


def fingerprint_done_flag(done_flag_path: str) -> Optional[str]:
    try:
        with open(done_flag_path) as f:
            return f.read().strip()
    except OSError:
        return None


def params_path(artifact_path: str) -> str:
    return artifact_path + ".params.json"


def write_build_params(artifact_path: str, params: Dict) -> None:
    write_json_atomic(params_path(artifact_path), params)


def read_build_params(artifact_path: str) -> Optional[Dict]:
    p = params_path(artifact_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _moved_keys(stored: Dict, expected: Dict, prefix: str = "") -> List:
    moved = []
    for k, want in expected.items():
        if k not in stored:
            continue
        was = stored[k]
        name = f"{prefix}{k}"
        if isinstance(was, dict) and isinstance(want, dict):
            moved.extend(_moved_keys(was, want, prefix=f"{name}."))
        elif was != want:
            moved.append((name, was, want))
    return moved


def check_build_params(artifact_path: str, expected: Dict, owner: str,
                       raise_on_mismatch: bool = False) -> bool:
    stored = read_build_params(artifact_path)
    if stored is None:
        return True
    moved = _moved_keys(stored, expected)
    if not moved:
        return True
    lines = "\n  ".join(f"{k}: {was!r} -> {now!r}" for k, was, now in moved)
    msg = (f"[{owner}] REBUILD: {os.path.basename(artifact_path)} was built "
           f"under different parameters, so it does not answer the question "
           f"being asked now.\n  {lines}")
    if raise_on_mismatch:
        raise IncompatibleArtifact(msg)
    print(msg, flush=True)
    return False


def append_jsonl(path: str, record: Dict) -> None:
    ensure_parent(path)
    line = json.dumps(record, ensure_ascii=True, default=_json_default) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("ascii"))
    finally:
        os.close(fd)


def write_jsonl_atomic(path: str, records: List[Dict]) -> None:
    ensure_parent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


_JSONL_WARNED = set()


def read_jsonl(path: str, owner: str = "resume_utils") -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.exists(path):
        return rows
    bad = 0
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                bad += 1
    if bad and path not in _JSONL_WARNED:
        _JSONL_WARNED.add(path)
        print(f"[{owner}] {bad} torn or unparsable line(s) skipped in {path}. "
              f"Those records are absent from the done-set, so the owning "
              f"stage regenerates exactly them on this run.", flush=True)
    return rows


def done_keys(path: str, key_fields: Sequence[str],
              owner: str = "resume_utils") -> set:
    return {tuple(r.get(k) for k in key_fields) for r in read_jsonl(path, owner)}


def unit_cache_path(partial_dir: str, group: str, unit: str) -> str:
    return os.path.join(partial_dir, f"{_safe(group)}__{_safe(unit)}.jsonl")


def unit_done(partial_dir: str, group: str, unit: str,
              require_rows: bool = False) -> bool:
    path = unit_cache_path(partial_dir, group, unit)
    if not os.path.exists(path):
        return False
    if not require_rows:
        return True
    return len(read_jsonl(path)) > 0


def write_unit_rows(partial_dir: str, group: str, unit: str,
                    rows: List[Dict]) -> None:
    ensure_dir(partial_dir)
    final = unit_cache_path(partial_dir, group, unit)
    tmp = final + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)


def load_group_rows(partial_dir: str, group: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isdir(partial_dir):
        return rows
    prefix = f"{_safe(group)}__"
    for name in sorted(os.listdir(partial_dir)):
        if not name.startswith(prefix) or not name.endswith(".jsonl"):
            continue
        rows.extend(read_jsonl(os.path.join(partial_dir, name)))
    return rows


def run_units_resumable(
    partial_dir: str,
    group: str,
    units: Sequence[str],
    compute_unit: Callable[[str], List[Dict]],
    progress_desc: Optional[str] = None,
    require_rows: bool = False,
    use_claims: bool = False,
    status_file: Optional[str] = None,
) -> pd.DataFrame:
    ensure_dir(partial_dir)
    todo = [u for u in units if not unit_done(partial_dir, group, u, require_rows)]
    print(f"[{group}] {len(units)} unit(s), {len(units) - len(todo)} already done, "
          f"{len(todo)} to run.", flush=True)
    try:
        from tqdm import tqdm
        iterator = tqdm(todo, desc=progress_desc or f"[{group}]", unit="unit")
    except ImportError:
        iterator = todo
    n_skipped = 0
    for unit in iterator:
        if unit_done(partial_dir, group, unit, require_rows):
            continue
        if use_claims and not claim_unit(partial_dir, f"{group}__{unit}"):
            print(f"[{group}] {unit}: claimed by a live job, leaving it alone.",
                  flush=True)
            continue
        try:
            rows = compute_unit(unit)
        except MissingInput as e:
            n_skipped += 1
            print(f"[{group}] SKIP {unit}: {e}", flush=True)
            continue
        finally:
            if use_claims:
                release_claim(partial_dir, f"{group}__{unit}")
        write_unit_rows(partial_dir, group, unit, rows)
        if status_file:
            append_status(status_file, f"{group}: {unit} wrote {len(rows)} row(s)")
    if n_skipped:
        print(f"[{group}] {n_skipped} unit(s) could not run and were left "
              f"uncached, so they re-run for free once their input exists.",
              flush=True)
    return pd.DataFrame(load_group_rows(partial_dir, group))


def clear_partial_dir(partial_dir: str) -> None:
    shutil.rmtree(partial_dir, ignore_errors=True)


def done_flag_path(cell_dir: str) -> str:
    return os.path.join(cell_dir, "done.flag")


def cell_is_done(cell_dir: str, required_files: Iterable[str] = ()) -> bool:
    flag = done_flag_path(cell_dir)
    if not os.path.exists(flag):
        return False
    for name in required_files:
        p = os.path.join(cell_dir, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"[resume] {cell_dir} carries a done.flag but {name} is "
                  f"missing or empty. Treating the cell as unfinished.",
                  flush=True)
            return False
    return True


def write_done_flag(cell_dir: str, meta: Optional[Dict] = None) -> None:
    ensure_dir(cell_dir)
    payload = {"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "host": socket.gethostname()}
    if meta:
        payload.update(meta)
    write_json_atomic(done_flag_path(cell_dir), payload)


def clear_cell(cell_dir: str) -> None:
    for name in ("done.flag", "resume.pt", "final.pt"):
        p = os.path.join(cell_dir, name)
        if os.path.exists(p):
            os.remove(p)
    for name in os.listdir(cell_dir) if os.path.isdir(cell_dir) else []:
        if name.endswith(".params.json"):
            os.remove(os.path.join(cell_dir, name))


def print_projected_peak(owner: str, parts: Dict[str, float],
                        multiplier: float = 1.0) -> float:
    total = float(sum(parts.values())) * float(multiplier)
    detail = ", ".join(f"{k} {v / 1024 ** 3:.2f} GiB" for k, v in sorted(parts.items()))
    print(f"[{owner}] projected peak {total / 1024 ** 3:.2f} GiB ({detail}"
          f"{f', x{multiplier:g} for library temporaries' if multiplier != 1.0 else ''}). "
          f"Request at least this much with --mem.", flush=True)
    return total


def append_status(status_file: str, message: str) -> None:
    ensure_parent(status_file)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n"
    with open(status_file, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def status_path(cfg: Dict, stage: str) -> str:
    return os.path.join(cfg["outputs"]["status_dir"], f"{stage}_status.txt")


_CLAIM_STALE_AFTER_S = 21600.0


def _claim_path(partial_dir: str, tag: str) -> str:
    return os.path.join(partial_dir, f"claim__{_safe(tag)}.json")


def _slurm_job_is_gone(job_id: str) -> bool:
    if not job_id:
        return False
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job_id)],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return "invalid job id" in (out.stderr or "").lower()
    return out.stdout.strip() == ""


def claim_unit(partial_dir: str, tag: str,
               stale_after_s: float = _CLAIM_STALE_AFTER_S) -> bool:
    ensure_dir(partial_dir)
    path = _claim_path(partial_dir, tag)
    record = {"host": socket.gethostname(), "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
              "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        pass
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    try:
        with open(path) as f:
            claim = json.load(f)
    except (OSError, json.JSONDecodeError):
        claim = {}
    try:
        age = time.time() - os.path.getmtime(path)
    except OSError:
        return claim_unit(partial_dir, tag, stale_after_s)
    owner_gone = _slurm_job_is_gone(str(claim.get("slurm_job_id", "")))
    if not owner_gone and age < stale_after_s:
        return False
    print(f"[claim] taking over {tag}: "
          f"{'the owning SLURM job is gone' if owner_gone else f'the claim is {age / 3600:.1f} h old'}.",
          flush=True)
    write_json_atomic(path, record)
    return True


def claim_cell(cell_dir: str, tag: str, owner: str) -> bool:
    if claim_unit(cell_dir, tag):
        return True
    print(f"[{owner}] {tag}: a live job owns this cell "
          f"({_claim_path(cell_dir, tag)}). Leaving it alone and moving on.",
          flush=True)
    return False


def heartbeat_claim(partial_dir: str, tag: str) -> None:
    path = _claim_path(partial_dir, tag)
    if os.path.exists(path):
        os.utime(path, None)


def release_claim(partial_dir: str, tag: str) -> None:
    path = _claim_path(partial_dir, tag)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
