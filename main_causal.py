"""
main_causal.py
Created on September 14, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import functools
import hashlib
import inspect
import os
import pdb
import resource
import sys
import time
import warnings
warnings.filterwarnings("ignore")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass


GLOBAL_CONFIG_PATH = "/PATH/causal/config/config.yaml"


def _raise_open_file_limit():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (ValueError, OSError):
        pass


def _memory_ceiling_bytes():
    try:
        rel = ""
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[1] in ("", "memory"):
                    rel = parts[2].lstrip("/")
                    break
        best = None
        node = os.path.join("/sys/fs/cgroup", rel)
        while True:
            for name in ("memory.max", "memory/memory.limit_in_bytes"):
                p = os.path.join(node, name)
                if os.path.exists(p):
                    with open(p) as f:
                        raw = f.read().strip()
                    if raw and raw != "max":
                        v = int(raw)
                        if v > 0 and (best is None or v < best):
                            best = v
            if os.path.normpath(node) in ("/sys/fs/cgroup", "/"):
                break
            node = os.path.dirname(node)
        if best:
            return best
    except (OSError, ValueError):
        pass
    try:
        if os.environ.get("SLURM_MEM_PER_NODE"):
            return int(os.environ["SLURM_MEM_PER_NODE"]) * 1024 ** 2
        if os.environ.get("SLURM_MEM_PER_CPU") and os.environ.get("SLURM_CPUS_PER_TASK"):
            return int(os.environ["SLURM_MEM_PER_CPU"]) * int(os.environ["SLURM_CPUS_PER_TASK"]) * 1024 ** 2
    except (ValueError, TypeError):
        pass
    return None


def _print_memory_ceiling():
    try:
        from config.serde import read_config
        if not read_config(GLOBAL_CONFIG_PATH)["CausalAudit"]["compute"]["print_memory_ceiling"]:
            return
        limit = _memory_ceiling_bytes()
        if limit:
            print(f"[main_causal] memory ceiling for this job: {limit / 1024 ** 3:.1f} GiB.", flush=True)
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 0)
        threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS") if os.environ.get(k)}
        print(f"[main_causal] usable CPU core(s): {cores}" + (f"; {threads}" if threads else "")
              + ("  <-- ONE core: every fit is single-threaded and every bootstrap is serial. Raise --cpus-per-task." if cores == 1 else ""),
              flush=True)
    except Exception:
        pass


def _print_pinned_version_drift():
    try:
        import importlib.metadata as meta
        pins = {}
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "requirements.txt")) as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if "==" in line:
                    name, _, version = line.partition("==")
                    pins[name.strip()] = version.strip()
        drift = []
        for name, want in sorted(pins.items()):
            try:
                have = meta.version(name)
            except Exception:
                continue
            if have.split("+")[0] != want:
                drift.append(f"{name} {have} against the pinned {want}")
        if drift:
            print(f"[main_causal] {len(drift)} package(s) differ from requirements.txt: " + "; ".join(drift)
                  + ". The statistics stack is pinned to the base environment; inference stages run under sorooshllm.", flush=True)
    except Exception:
        pass


_raise_open_file_limit()
_print_memory_ceiling()
_print_pinned_version_drift()


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONTROLLER_FILE = os.path.basename(os.path.abspath(__file__))
_CONTROLLER_DRIFT_ANNOUNCED = False
_RESTART_ENV = "CAUSAL_SOURCE_RESTARTS"
_MAX_RESTARTS = 20


def _source_fingerprint() -> dict:
    fp = {}
    for root, dirs, files in os.walk(_REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".idea", "legacy")]
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                try:
                    with open(p, "rb") as fh:
                        digest = hashlib.blake2b(fh.read(), digest_size=16).hexdigest()
                except OSError:
                    continue
                fp[os.path.relpath(p, _REPO_ROOT)] = digest
    return fp


_SOURCE_AT_IMPORT = _source_fingerprint()


def _changed_modules(now: dict) -> list:
    return sorted(k for k in set(now) | set(_SOURCE_AT_IMPORT) if now.get(k) != _SOURCE_AT_IMPORT.get(k))


def _restart_on_new_source(stage_name: str, changed: list) -> None:
    n = int(os.environ.get(_RESTART_ENV, "0"))
    if n >= _MAX_RESTARTS:
        raise RuntimeError(f"[{stage_name}] the source changed {n} times under this job; refusing to restart again.")
    settle_start = time.time()
    last = _source_fingerprint()
    while True:
        time.sleep(20)
        now = _source_fingerprint()
        if now == last:
            break
        last = now
        if time.time() - settle_start > 600:
            raise RuntimeError(f"[{stage_name}] the repository kept changing for 10 minutes; not restarting on a moving tree.")
    print(f"[source_guard] {len(changed)} file(s) changed on disk since this job started ({', '.join(changed[:8])}"
          f"{', ...' if len(changed) > 8 else ''}). Restarting on the new source at this stage boundary (restart {n + 1}).", flush=True)
    os.environ[_RESTART_ENV] = str(n + 1)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.closerange(3, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    except (ValueError, OSError):
        pass
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])


def _assert_source_unchanged(stage_name: str):
    global _CONTROLLER_DRIFT_ANNOUNCED
    changed = _changed_modules(_source_fingerprint())
    if not changed:
        return
    if _CONTROLLER_FILE in changed and not _CONTROLLER_DRIFT_ANNOUNCED:
        _CONTROLLER_DRIFT_ANNOUNCED = True
        print(f"[source_guard] {_CONTROLLER_FILE} changed on disk since this job started, most likely an edit for another job. "
              f"This process keeps its own compiled copy. Continuing.", flush=True)
    modules = [k for k in changed if k != _CONTROLLER_FILE]
    if modules:
        _restart_on_new_source(stage_name, modules)


_INPUT_ROOTS = ("repo_root", "datasets_root", "user_home")
_OUTPUT_ROOTS = ("outputs_root",)


def _assert_roots(cfg_path: str):
    from config.serde import read_config
    cfg = read_config(cfg_path)["CausalAudit"]
    for key in _INPUT_ROOTS:
        if not os.path.isdir(cfg[key]):
            raise NotADirectoryError(f"[main_causal] {key} does not exist: {cfg[key]}. Switching machines is TWO switches: the machine block "
                                     f"at the top of config.yaml AND GLOBAL_CONFIG_PATH in this file. Comment and uncomment whole blocks.")
    for key in _OUTPUT_ROOTS:
        os.makedirs(cfg[key], exist_ok=True)
    declared = str(cfg.get("global_config_path", ""))
    if declared and os.path.realpath(declared) != os.path.realpath(cfg_path):
        raise ValueError(f"[main_causal] this run loaded {cfg_path}, and that file says its own home is {declared}. "
                         f"The machine block and GLOBAL_CONFIG_PATH disagree; switch both.")


def _guard(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _assert_source_unchanged(fn.__name__)
        try:
            bound = inspect.signature(fn).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            cfg_path = bound.arguments.get("cfg_path", GLOBAL_CONFIG_PATH)
        except TypeError:
            cfg_path = GLOBAL_CONFIG_PATH
        _assert_roots(cfg_path)
        return fn(*args, **kwargs)
    return wrapper


def main_fixture_pipeline(cfg_path: str = GLOBAL_CONFIG_PATH):
    import subprocess
    script = os.path.join(_REPO_ROOT, "tests", "test_fixture_pipeline.py")
    if subprocess.run([sys.executable, "-u", script], cwd=_REPO_ROOT).returncode != 0:
        raise RuntimeError("[main_causal] the fixture failed, so no stage after it ran.")


def main_raddino_cache(dataset: str, cfg_path: str = GLOBAL_CONFIG_PATH):
    from models.raddino import main_raddino_cache
    main_raddino_cache(cfg_path, dataset)


def main_raddino_heads(cfg_path: str = GLOBAL_CONFIG_PATH):
    from models.raddino import main_raddino_heads
    main_raddino_heads(cfg_path)


def main_extend_manifest_mimic(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.extend_manifest import main_extend_manifest_mimic
    main_extend_manifest_mimic(cfg_path)


def main_extend_manifest_chexpert(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.extend_manifest import main_extend_manifest_chexpert
    main_extend_manifest_chexpert(cfg_path)


def main_build_assets(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_assets import main_build_assets
    main_build_assets(cfg_path)


def main_check_frozen_manifests(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_probe_set import main_check_frozen_manifests
    main_check_frozen_manifests(cfg_path)


def main_import_delivered(cfg_path: str = GLOBAL_CONFIG_PATH):
    from Inference.import_delivered import main_import_delivered
    main_import_delivered(cfg_path)


def main_run(model_key: str, cfg_path: str = GLOBAL_CONFIG_PATH):
    from Inference.inference_runner import main_run
    main_run(cfg_path, model_key)


def main_run_raddino(cfg_path: str = GLOBAL_CONFIG_PATH):
    from Inference.inference_runner import main_run_raddino
    main_run_raddino(cfg_path)


def main_rerun_failed(model_key: str, cfg_path: str = GLOBAL_CONFIG_PATH):
    from Inference.inference_runner import main_rerun_failed
    main_rerun_failed(cfg_path, model_key)


def main_reparse_records(cfg_path: str = GLOBAL_CONFIG_PATH):
    from Inference.reparse_records import main_reparse_records
    main_reparse_records(cfg_path)


def main_compute_metrics(cfg_path: str = GLOBAL_CONFIG_PATH):
    from analysis.compute_metrics import main_compute_metrics
    main_compute_metrics(cfg_path)


def main_paired_comparisons(cfg_path: str = GLOBAL_CONFIG_PATH):
    from analysis.paired_comparisons import main_paired_comparisons
    main_paired_comparisons(cfg_path)


def main_reanalyses(cfg_path: str = GLOBAL_CONFIG_PATH):
    from analysis.reanalyses import main_reanalyses
    main_reanalyses(cfg_path)


def main_per_case_export(cfg_path: str = GLOBAL_CONFIG_PATH):
    from analysis.per_case_export import main_per_case_export
    main_per_case_export(cfg_path)


def main_build_final_tables(cfg_path: str = GLOBAL_CONFIG_PATH):
    from aggregate.build_final_tables import main_build_final_tables
    main_build_final_tables(cfg_path)


def main_build_reader_packets(cfg_path: str = GLOBAL_CONFIG_PATH):
    from reader_study.build_packets import main_build_reader_packets
    main_build_reader_packets(cfg_path)


def main_analyze_reader_study(cfg_path: str = GLOBAL_CONFIG_PATH):
    from reader_study.analyze_reader_study import main_analyze_reader_study
    main_analyze_reader_study(cfg_path)


def main_build_figures(cfg_path: str = GLOBAL_CONFIG_PATH):
    from figures.build_figures import main_build_figures
    main_build_figures(cfg_path)


for _n, _f in list(globals().items()):
    if _n.startswith("main_") and callable(_f) and getattr(_f, "__module__", None) == __name__:
        globals()[_n] = _guard(_f)
del _n, _f


