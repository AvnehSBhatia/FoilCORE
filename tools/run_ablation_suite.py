#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.ablations import VariantSpec, get_variant_spec
from src.data import warmup_aero_dataset_caches


_ACTIVE_PROCS: set[subprocess.Popen] = set()
_ACTIVE_PROCS_LOCK = threading.Lock()
_STARTUP_GATE: threading.Semaphore | None = None

_SUITE_LOG_LOCK = threading.Lock()
_SUITE_ACTIVE_LOCK = threading.Lock()
# variant_id -> ("train"|"eval", slug) for heartbeats
_SUITE_ACTIVE: dict[int, tuple[str, str]] = {}
_SUITE_HB_STOP = threading.Event()


def _shutdown_executor(ex: ThreadPoolExecutor) -> None:
    """Don’t block on worker threads (each runs a long subprocess)."""
    kw: dict = {"wait": False}
    if sys.version_info >= (3, 9):
        kw["cancel_futures"] = True
    try:
        ex.shutdown(**kw)
    except TypeError:
        ex.shutdown(wait=False)


def _parse_variant_ids(expr: str) -> list[int]:
    out: set[int] = set()
    for part in expr.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(p))
    return sorted(out)


def _parse_parallel(value: str, n_trainable: int) -> int:
    v = str(value).strip().lower()
    if v in {"all", "auto"}:
        return max(1, n_trainable)
    out = int(v)
    if out < 1:
        raise ValueError("--parallel must be >= 1, 'all', or 'auto'")
    return out


def _visible_cuda_device_count() -> int | None:
    """How many GPUs this process will see (respects CUDA_VISIBLE_DEVICES)."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        stripped = str(raw).strip()
        if stripped != "":
            parts = [p for p in stripped.split(",") if p.strip() != ""]
            return max(1, len(parts))
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except ImportError:
        pass
    return None


def _mps_only_accelerator() -> bool:
    """True when this machine has MPS and no CUDA (Apple GPU / unified memory)."""
    try:
        import torch

        return bool(torch.backends.mps.is_available()) and (not torch.cuda.is_available())
    except Exception:
        return False


def _default_gpu_slots(n_trainable: int) -> int:
    """Concurrent trainers that share one pool of GPU memory (not VRAM per process)."""
    if os.environ.get("FOILCORE_ABLATION_NO_GPU_CAP", "").strip() in ("1", "true", "yes"):
        return max(1, n_trainable)
    env_v = os.environ.get("FOILCORE_ABLATION_GPU_SLOTS")
    if env_v is not None and str(env_v).strip() != "":
        return max(1, min(int(env_v), n_trainable))
    vis = _visible_cuda_device_count()
    if vis is None:
        # MPS: multiple full training jobs usually exhaust unified memory — default to 1.
        if _mps_only_accelerator():
            cap = int(os.environ.get("FOILCORE_ABLATION_MPS_SLOTS", "1"))
            return max(1, min(max(1, cap), n_trainable))
        return max(1, n_trainable)
    # One visible GPU: many concurrent full training jobs each claim VRAM → OOM kills mid-epoch.
    # Default is conservative; raise with --gpu-slots or FOILCORE_ABLATION_GPU_SLOTS (e.g. 12 on 192GB).
    per = 8
    return max(1, min(per * vis, n_trainable))


def _variant_dir(root: Path, spec: VariantSpec) -> Path:
    return root / f"{spec.idx:02d}_{spec.slug}"


def _terminate_process_tree(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=timeout_s)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass

    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass


def _terminate_active_processes() -> None:
    with _ACTIVE_PROCS_LOCK:
        procs = list(_ACTIVE_PROCS)
    for proc in procs:
        _terminate_process_tree(proc)


def _suite_log(root: Path, message: str, *, also_tqdm: bool = True) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} {message}\n"
    with _SUITE_LOG_LOCK:
        path = root / "suite_events.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    if also_tqdm:
        tqdm.write(message.rstrip("\n"))


def _suite_active_set(vid: int, phase: str, slug: str) -> None:
    with _SUITE_ACTIVE_LOCK:
        _SUITE_ACTIVE[vid] = (phase, slug)


def _suite_active_clear(vid: int) -> None:
    with _SUITE_ACTIVE_LOCK:
        _SUITE_ACTIVE.pop(vid, None)


def _suite_heartbeat_loop(root: Path) -> None:
    interval = max(5.0, float(os.environ.get("FOILCORE_ABLATION_SUITE_HB_SEC", "30")))
    while not _SUITE_HB_STOP.wait(interval):
        with _SUITE_ACTIVE_LOCK:
            items = sorted((vid, ph, sl) for vid, (ph, sl) in _SUITE_ACTIVE.items())
        if not items:
            _suite_log(root, "[suite] heartbeat: no jobs in active set (workers idle or between phases).")
            continue
        parts = [f"{vid:02d}_{sl}({ph})" for vid, ph, sl in items[:40]]
        tail = f" …(+{len(items) - 40})" if len(items) > 40 else ""
        _suite_log(
            root,
            f"[suite] heartbeat: {len(items)} job(s) in flight: {', '.join(parts)}{tail}",
        )


def _log_contains_any(path: Path, needles: tuple[str, ...]) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(n in text for n in needles)


def _run(
    cmd: list[str],
    log_path: Path,
    cwd: Path,
    env_overrides: dict[str, str] | None = None,
    startup_gate: threading.Semaphore | None = None,
    startup_release_needles: tuple[str, ...] = (),
    startup_timeout_s: float = 600.0,
    suite_event_root: Path | None = None,
    suite_label: str | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if env_overrides:
        env.update(env_overrides)
    gate_acquired = False
    if startup_gate is not None:
        if suite_event_root is not None and suite_label:
            _suite_log(
                suite_event_root,
                f"WAIT startup_gate | {suite_label} | log={log_path.name}",
            )
        startup_gate.acquire()
        gate_acquired = True
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        if suite_event_root is not None and suite_label:
            _suite_log(
                suite_event_root,
                f"SPAWN pid={proc.pid} | {suite_label} | log={log_path.name}",
            )
        with _ACTIVE_PROCS_LOCK:
            _ACTIVE_PROCS.add(proc)
        try:
            if gate_acquired and startup_release_needles:
                deadline = time.monotonic() + max(1.0, float(startup_timeout_s))
                while proc.poll() is None and time.monotonic() < deadline:
                    if _log_contains_any(log_path, startup_release_needles):
                        startup_gate.release()
                        gate_acquired = False
                        break
                    time.sleep(1.0)
            return int(proc.wait())
        finally:
            if gate_acquired:
                startup_gate.release()
            with _ACTIVE_PROCS_LOCK:
                _ACTIVE_PROCS.discard(proc)


def _tail_file(path: Path, max_lines: int = 40) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return text.strip()
    return "\n".join(lines[-max_lines:])


_ERR_HINT = re.compile(
    r"(?i)(traceback|outofmemory|out of memory|tried to allocate|cuda error|"
    r"runtimeerror|killed|sigkill|signal 9|segmentation|nan |assertionerror|error:)"
)


def _train_exit_hint(rc: int) -> str | None:
    if rc in (-9, 137) or rc == -signal.SIGKILL:
        return (
            "Subprocess exit looks like SIGKILL (often OOM: shared GPU VRAM or host RAM). "
            "Try --gpu-slots 4 (or 2), --batch-size lower, --num-workers 0, and/or "
            "--startup-parallel 2. Each job loads the full cached train/val tables in that "
            "process; many concurrent jobs multiply RAM. FOILCORE_ABLATION_TRAIN_NUM_WORKERS "
            "pins per-job DataLoader workers; FOILCORE_ABLATION_DL_WORKERS_TOTAL_CAP budgets "
            "workers across concurrent jobs automatically."
        )
    return None


def _effective_train_num_workers(args, suite_parallel: int) -> int:
    """Cap DataLoader workers per job so suite_parallel * workers does not explode host RAM."""
    req = max(0, int(args.num_workers))
    env = os.environ.get("FOILCORE_ABLATION_TRAIN_NUM_WORKERS")
    if env is not None and str(env).strip() != "":
        return max(0, int(env))
    if suite_parallel <= 1:
        return req
    cap = int(os.environ.get("FOILCORE_ABLATION_DL_WORKERS_TOTAL_CAP", "16"))
    return max(0, min(req, cap // suite_parallel))


def _train_log_failure_snippet(path: Path, max_chars: int = 12000, tail_lines: int = 48) -> str | None:
    """Prefer the last traceback / OOM lines; tqdm tails alone hide the real failure."""
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(text) > 8_000_000:
        text = text[-8_000_000:]
    stripped = text.rstrip()
    if not stripped:
        return None
    tb = "Traceback (most recent call last)"
    idx = stripped.rfind("\n" + tb)
    if idx == -1:
        idx = stripped.rfind(tb)
    if idx != -1:
        block = stripped[idx:]
        if len(block) > max_chars:
            block = block[:max_chars] + "\n... [truncated]"
        return block
    bad_lines = [ln for ln in stripped.splitlines() if _ERR_HINT.search(ln)]
    if bad_lines:
        snippet = "\n".join(bad_lines[-80:])
        if len(snippet) > max_chars:
            snippet = "... [truncated from start]\n" + snippet[-max_chars:]
        return snippet
    return _tail_file(path, max_lines=tail_lines)


def _worker_env(args) -> dict[str, str]:
    # Many concurrent training processes can oversubscribe host CPU threads. Keep
    # BLAS/OpenMP modest unless the caller explicitly exports different values.
    threads = str(max(1, int(args.cpu_threads_per_job)))
    env: dict[str, str] = {}
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = os.environ.get(key, threads)
    return env


def _train_variant(spec: VariantSpec, args, root: Path, repo: Path) -> dict:
    vdir = _variant_dir(root, spec)
    ckpt_dir = vdir / "checkpoints"
    logs_dir = vdir / "logs"
    for d in (ckpt_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    label = f"[{spec.idx:02d}] {spec.slug}"
    _suite_active_set(spec.idx, "train", spec.slug)
    _suite_log(root, f"BEGIN train thread | {label} | dir={vdir.name}")
    try:
        return _train_variant_body(spec, args, root, repo, vdir, ckpt_dir, logs_dir, label)
    finally:
        _suite_active_clear(spec.idx)
        _suite_log(root, f"END train thread | {label}")


def _train_variant_body(
    spec: VariantSpec,
    args,
    root: Path,
    repo: Path,
    vdir: Path,
    ckpt_dir: Path,
    logs_dir: Path,
    label: str,
) -> dict:
    nw = int(getattr(args, "_ablation_train_num_workers", args.num_workers))
    train_cmd = [
        sys.executable,
        "-m",
        "src.train_model",
        "--variant-id",
        str(spec.idx),
        "--save-dir",
        str(ckpt_dir),
        "--train-csv",
        args.train_csv,
        "--val-csv",
        args.val_csv,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--num-workers",
        str(nw),
    ]
    if getattr(args, "no_dataset_cache", False):
        train_cmd.append("--no-dataset-cache")
    elif args.dataset_cache_dir:
        train_cmd.extend(["--dataset-cache-dir", args.dataset_cache_dir])
    env = _worker_env(args)
    rc = _run(
        train_cmd,
        logs_dir / "train.log",
        repo,
        env_overrides=env,
        startup_gate=_STARTUP_GATE,
        startup_release_needles=("Starting training from epoch",),
        startup_timeout_s=args.startup_timeout,
        suite_event_root=root,
        suite_label=label,
    )
    if rc != 0:
        tail = _train_log_failure_snippet(logs_dir / "train.log", tail_lines=48)
        _suite_log(root, f"RESULT train_failed rc={rc} | {label}")
        return {
            "variant_id": spec.idx,
            "status": "train_failed",
            "returncode": rc,
            "train_fail_hint": _train_exit_hint(rc),
            "dir": str(vdir),
            "train_log": str(logs_dir / "train.log"),
            "train_log_tail": tail,
        }

    best_ckpt = ckpt_dir / "foilcore_best.pt"
    if not best_ckpt.exists():
        _suite_log(root, f"RESULT missing_checkpoint | {label}")
        return {"variant_id": spec.idx, "status": "missing_checkpoint", "dir": str(vdir)}

    _suite_log(root, f"RESULT trained | {label} | checkpoint={best_ckpt.name}")
    return {
        "variant_id": spec.idx,
        "status": "trained",
        "dir": str(vdir),
        "checkpoint": str(best_ckpt),
    }


def _eval_variant(spec: VariantSpec, args, root: Path, repo: Path, train_result: dict) -> dict:
    vdir = _variant_dir(root, spec)
    ckpt_dir = vdir / "checkpoints"
    metrics_dir = vdir / "metrics"
    plots_dir = vdir / "plots"
    logs_dir = vdir / "logs"
    for d in (metrics_dir, plots_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    label = f"[{spec.idx:02d}] {spec.slug}"
    _suite_active_set(spec.idx, "eval", spec.slug)
    _suite_log(root, f"BEGIN eval thread | {label}")
    try:
        return _eval_variant_body(spec, args, root, repo, train_result, vdir, ckpt_dir, metrics_dir, plots_dir, logs_dir, label)
    finally:
        _suite_active_clear(spec.idx)
        _suite_log(root, f"END eval thread | {label}")


def _eval_variant_body(
    spec: VariantSpec,
    args,
    root: Path,
    repo: Path,
    train_result: dict,
    vdir: Path,
    ckpt_dir: Path,
    metrics_dir: Path,
    plots_dir: Path,
    logs_dir: Path,
    label: str,
) -> dict:
    best_ckpt = Path(train_result.get("checkpoint") or ckpt_dir / "foilcore_best.pt")
    if not best_ckpt.exists():
        _suite_log(root, f"RESULT eval skip missing_checkpoint | {label}")
        return {"variant_id": spec.idx, "status": "missing_checkpoint", "dir": str(vdir)}

    env = _worker_env(args)

    bench_out = metrics_dir / "benchmark.json"
    bench_cmd = [
        sys.executable,
        "-c",
        (
            "import json; from src.benchmark import benchmark; "
            f"r=benchmark({best_ckpt.as_posix()!r}, {args.val_csv!r}, "
            f"batch_size={args.eval_batch_size}, num_workers={args.num_workers}, "
            f"plot_dir={str(plots_dir / 'benchmark')!r}, plot_max={args.plot_max}, "
            f"ar_prefix_len={args.ar_prefix_len}); "
            f"open({bench_out.as_posix()!r},'w').write(json.dumps(r,indent=2))"
        ),
    ]
    bench_rc = _run(
        bench_cmd,
        logs_dir / "benchmark.log",
        repo,
        env_overrides=env,
        suite_event_root=root,
        suite_label=f"{label} [benchmark]",
    )

    evals_out = metrics_dir / "evals.json"
    evals_cmd = [
        sys.executable,
        "evals.py",
        "--ckpt",
        str(best_ckpt),
        "--csv",
        args.val_csv,
        "--out",
        str(evals_out),
        "--batch-size",
        str(args.eval_batch_size),
        "--nf-workers",
        str(args.nf_workers),
        "--nf-chunksize",
        str(args.nf_chunksize),
    ]
    eval_rc = _run(
        evals_cmd,
        logs_dir / "evals.log",
        repo,
        env_overrides=env,
        suite_event_root=root,
        suite_label=f"{label} [evals.py]",
    )

    # copy train outputs into neat metrics/plots folder
    for name in ("metrics_history.json", "training_curves.png"):
        src = ckpt_dir / name
        if src.exists():
            dst = metrics_dir / name if name.endswith(".json") else (plots_dir / name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

    st = "ok" if bench_rc == 0 and eval_rc == 0 else "eval_failed"
    _suite_log(root, f"RESULT {st} bench_rc={bench_rc} eval_rc={eval_rc} | {label}")
    return {
        "variant_id": spec.idx,
        "status": st,
        "benchmark_returncode": bench_rc,
        "evals_returncode": eval_rc,
        "dir": str(vdir),
        "checkpoint": str(best_ckpt),
        "benchmark_json": str(bench_out) if bench_out.exists() else None,
        "evals_json": str(evals_out) if evals_out.exists() else None,
    }


def _extract_eval_only_variants(root: Path, statuses: dict[int, dict]) -> None:
    base = statuses.get(0)
    if not base or not base.get("evals_json"):
        return
    base_evals = Path(base["evals_json"])
    if not base_evals.exists():
        return
    data = json.loads(base_evals.read_text())
    cond = data.get("conditions", {})

    mapping = {
        16: "foilcore_ar_gt_1",
        17: "foilcore_ar_rnd_1",
        20: "neuralfoil_xxsmall",
        21: "neuralfoil_xxxlarge",
    }
    for vid, key in mapping.items():
        try:
            spec = get_variant_spec(vid)
        except KeyError:
            continue
        vdir = _variant_dir(root, spec)
        md = vdir / "metrics"
        md.mkdir(parents=True, exist_ok=True)
        out = {
            "variant": asdict(spec),
            "source": str(base_evals),
            "condition_key": key,
            "metrics": cond.get(key),
            "win_rates_subset": {k: v for k, v in data.get("win_rates", {}).items() if key in k},
        }
        (md / "derived_metrics.json").write_text(json.dumps(out, indent=2))


def main() -> None:
    global _STARTUP_GATE

    ap = argparse.ArgumentParser(description="Run FoilCORE ablation suite (parallel workers).")
    ap.add_argument(
        "--variants",
        default="2,4,5,20,23",
        help=(
            "Variant ids, comma/range format. Default runs the full 2×2 ablation factorial "
            "(mixed vs all-normal block schedule × path-nonlin/GELU) plus two xxsmall-scale MLP baselines. "
            "Variant 0 (full FoilCORE) is pre-trained and excluded from training here. "
            "See src/ablations.py for the complete spec list."
        ),
    )
    ap.add_argument(
        "--parallel",
        default="all",
        help="Desired concurrent training jobs (integer, 'all', or 'auto'). On CUDA/ROCm this is capped by --gpu-slots unless --no-gpu-cap.",
    )
    ap.add_argument(
        "--gpu-slots",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max concurrent training subprocesses sharing the visible GPU(s). "
            "Default: 8 per visible CUDA device; on MPS-only hosts default is "
            "FOILCORE_ABLATION_MPS_SLOTS (default 1). Override with FOILCORE_ABLATION_GPU_SLOTS. "
            "Ignored with --no-gpu-cap."
        ),
    )
    ap.add_argument(
        "--no-gpu-cap",
        action="store_true",
        help="Do not cap --parallel against GPU slots (can OOM if many jobs share one GPU).",
    )
    ap.add_argument(
        "--eval-parallel",
        default="all",
        help="Concurrent benchmark/eval jobs after training. Use an integer, 'all', or 'auto' (default: all successful trainings).",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=900)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--startup-parallel",
        default="4",
        help=(
            "Max jobs simultaneously loading datasets/model before training starts. "
            "Use an integer, 'all', or 'auto' (default: 4)."
        ),
    )
    ap.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        help="Seconds before releasing a startup slot even if the training-start log marker was not seen.",
    )
    ap.add_argument("--plot-max", type=int, default=16)
    ap.add_argument("--ar-prefix-len", type=int, default=1)
    ap.add_argument("--val-csv", default="data/val.csv")
    ap.add_argument("--train-csv", default="data/train.csv")
    ap.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Pickled parsed DataFrames (default: data/.aero_cache relative to CSV dir unless AERO_CACHE_DIR).",
    )
    ap.add_argument(
        "--no-dataset-cache",
        action="store_true",
        help="Disable pickle cache; each worker parses CSV from scratch.",
    )
    ap.add_argument("--nf-workers", type=int, default=4)
    ap.add_argument("--nf-chunksize", type=int, default=64)
    ap.add_argument(
        "--cpu-threads-per-job",
        type=int,
        default=1,
        help="Default OMP/MKL/OpenBLAS/NumExpr threads for each subprocess unless already exported.",
    )
    ap.add_argument("--out-root", default="outputs/ablation_runs")
    args = ap.parse_args()

    selected = _parse_variant_ids(args.variants)
    specs = [get_variant_spec(i) for i in selected]
    train_specs = [s for s in specs if s.trainable]
    parallel_requested = _parse_parallel(args.parallel, len(train_specs))
    gpu_slots = (
        max(1, min(int(args.gpu_slots), len(train_specs)))
        if args.gpu_slots is not None
        else _default_gpu_slots(len(train_specs))
    )
    if args.no_gpu_cap:
        parallel = parallel_requested
    else:
        parallel = min(parallel_requested, gpu_slots)
    if parallel < parallel_requested:
        tqdm.write(
            f"Note: capping concurrent trainers {parallel_requested} → {parallel} "
            f"(GPU slots={gpu_slots}; raise with --gpu-slots N, FOILCORE_ABLATION_GPU_SLOTS, "
            f"or use --no-gpu-cap if you are sure VRAM can hold {parallel_requested} full jobs)."
        )
    args._ablation_train_num_workers = _effective_train_num_workers(args, parallel)
    if int(args._ablation_train_num_workers) != int(args.num_workers):
        tqdm.write(
            f"DataLoader num_workers per job: {args._ablation_train_num_workers} "
            f"(capped from {args.num_workers} for {parallel} concurrent trainer(s); "
            "override with FOILCORE_ABLATION_TRAIN_NUM_WORKERS or FOILCORE_ABLATION_DL_WORKERS_TOTAL_CAP)."
        )
    startup_parallel = min(parallel, _parse_parallel(args.startup_parallel, len(train_specs)))
    _STARTUP_GATE = threading.Semaphore(startup_parallel)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.out_root) / run_stamp
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent

    try:
        import torch

        if torch.cuda.is_available():
            hip = getattr(torch.version, "hip", None)
            kind = "ROCm/HIP" if hip else "CUDA"
            tqdm.write(
                f"{kind}: {torch.cuda.device_count()} visible device(s); "
                f"up to {parallel} concurrent training job(s) "
                f"(requested {parallel_requested}, gpu_slots cap={gpu_slots if not args.no_gpu_cap else 'off'})."
            )
        elif torch.backends.mps.is_available():
            tqdm.write(
                f"MPS (unified memory): up to {parallel} concurrent training job(s) "
                f"(requested {parallel_requested}, gpu_slots={gpu_slots if not args.no_gpu_cap else 'off'}). "
                "Tip: FOILCORE_DEVICE=mps, FOILCORE_ABLATION_MPS_SLOTS=1, --parallel 1."
            )
    except ImportError:
        pass

    if train_specs:
        if not args.no_dataset_cache:
            tqdm.write("Prebuilding parsed dataset cache (one parse; workers reload pickle)...")
            cd = Path(args.dataset_cache_dir) if args.dataset_cache_dir else None
            warmup_aero_dataset_caches(args.train_csv, args.val_csv, cache_dir=cd)
        tqdm.write(
            f"Run directory: {root}\n"
            f"Trainable variants: {len(train_specs)}  |  workers: {parallel}  |  "
            f"startup loaders: {startup_parallel}  |  epochs: {args.epochs}\n"
            "Phase 1 trains all variants first; phase 2 benchmarks/evaluates successful checkpoints."
        )

    statuses: dict[int, dict] = {}
    hb_thread: threading.Thread | None = None
    if train_specs:
        _SUITE_HB_STOP.clear()
        hb_thread = threading.Thread(
            target=_suite_heartbeat_loop,
            args=(root,),
            daemon=True,
            name="suite-heartbeat",
        )
        hb_thread.start()
        preview = ", ".join(f"{s.idx:02d}_{s.slug}" for s in train_specs[:14])
        if len(train_specs) > 14:
            preview += f", …(+{len(train_specs) - 14} more)"
        iv = max(5.0, float(os.environ.get("FOILCORE_ABLATION_SUITE_HB_SEC", "30")))
        _suite_log(
            root,
            f"Phase 1: submitted {len(train_specs)} variant(s), max_workers={parallel} | "
            f"heartbeat every {iv:.0f}s | live log: {root / 'suite_events.log'} | queue: {preview}",
        )

    ex = ThreadPoolExecutor(max_workers=parallel)
    try:
        fut_map = {ex.submit(_train_variant, s, args, root, repo): s for s in train_specs}
        with tqdm(
            total=len(fut_map),
            desc="Ablation suite",
            unit="variant",
            miniters=1,
            mininterval=0.2,
            dynamic_ncols=True,
        ) as pbar:
            for f in as_completed(fut_map):
                s = fut_map[f]
                try:
                    res = f.result()
                except Exception as exc:
                    res = {
                        "variant_id": s.idx,
                        "status": f"exception: {exc}",
                        "dir": str(_variant_dir(root, s)),
                    }
                statuses[s.idx] = res
                tqdm.write(f"[{s.idx:02d}] {s.slug}: {res['status']}")
                if res.get("status") == "train_failed":
                    if res.get("train_fail_hint"):
                        tqdm.write("Hint: " + res["train_fail_hint"])
                    if res.get("train_log_tail"):
                        tqdm.write("--- train.log (tail) ---\n" + res["train_log_tail"])
                pbar.set_postfix_str(f"last={s.slug}", refresh=False)
                pbar.update(1)
    except KeyboardInterrupt:
        tqdm.write("\nInterrupted — terminating active training/eval subprocesses...")
        _terminate_active_processes()
        raise
    finally:
        _shutdown_executor(ex)
        if hb_thread is not None:
            _SUITE_HB_STOP.set()
            hb_thread.join(timeout=3.0)

    eval_specs = [s for s in train_specs if statuses.get(s.idx, {}).get("status") == "trained"]
    if eval_specs:
        eval_parallel = _parse_parallel(args.eval_parallel, len(eval_specs))
        tqdm.write(f"\nStarting eval phase: {len(eval_specs)} checkpoint(s)  |  workers: {eval_parallel}")
        _SUITE_HB_STOP.clear()
        hb_ev = threading.Thread(
            target=_suite_heartbeat_loop,
            args=(root,),
            daemon=True,
            name="suite-heartbeat-eval",
        )
        hb_ev.start()
        _suite_log(
            root,
            f"Phase 2: eval/benchmark {len(eval_specs)} checkpoint(s), max_workers={eval_parallel}",
        )
        ex = ThreadPoolExecutor(max_workers=eval_parallel)
        try:
            fut_map = {
                ex.submit(_eval_variant, s, args, root, repo, statuses[s.idx]): s
                for s in eval_specs
            }
            with tqdm(
                total=len(fut_map),
                desc="Ablation eval",
                unit="variant",
                miniters=1,
                mininterval=0.2,
                dynamic_ncols=True,
            ) as pbar:
                for f in as_completed(fut_map):
                    s = fut_map[f]
                    try:
                        res = f.result()
                    except Exception as exc:
                        res = {
                            "variant_id": s.idx,
                            "status": f"exception: {exc}",
                            "dir": str(_variant_dir(root, s)),
                        }
                    statuses[s.idx] = res
                    tqdm.write(f"[{s.idx:02d}] {s.slug}: {res['status']}")
                    pbar.set_postfix_str(f"last={s.slug}", refresh=False)
                    pbar.update(1)
        except KeyboardInterrupt:
            tqdm.write("\nInterrupted — terminating active training/eval subprocesses...")
            _terminate_active_processes()
            raise
        finally:
            _shutdown_executor(ex)
            _SUITE_HB_STOP.set()
            hb_ev.join(timeout=3.0)

    # eval-only variants (derived from baseline eval JSON)
    _extract_eval_only_variants(root, statuses)

    for s in specs:
        statuses.setdefault(s.idx, {"variant_id": s.idx, "status": "not_run", "dir": str(_variant_dir(root, s))})

    summary = {
        "timestamp": run_stamp,
        "root": str(root),
        "requested_variants": selected,
        "variants": [asdict(get_variant_spec(i)) for i in selected],
        "results": [statuses[i] for i in selected],
    }
    (root / "suite_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSuite complete. Summary: {root / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
