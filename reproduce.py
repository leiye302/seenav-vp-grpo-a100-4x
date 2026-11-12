"""Run SeeNav-Agent VP + GRPO on four A100 80GB GPUs."""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
DEFAULT_SOURCE = ROOT / ".deps" / "SeeNav-Agent"
GPU_COUNT = 4
OLD_GPU = "trainer.n_gpus_per_node=8"
NEW_GPU = "trainer.n_gpus_per_node=4"


def git(source, *args, direct=False):
    command = ["git", "-c", "core.autocrlf=false", "-c", "credential.helper="]
    if direct:
        command += ["-c", "http.proxy=", "-c", "http.version=HTTP/1.1"]
    return subprocess.check_output(
        [*command, "-C", str(source), *args], text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    ).strip()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source):
    if not (source / ".git").exists() or git(source, "rev-parse", "HEAD") != LOCK["commit"]:
        raise ValueError("Use the pinned official checkout (python reproduce.py fetch).")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("The official checkout contains tracked changes; use a clean pinned checkout.")
    data = source / LOCK["navigation_data"] / "base.json"
    if (sha(data) != LOCK["evaluation_task_sha256"]
            or len(json.loads(data.read_text(encoding="utf-8"))["tasks"]) != LOCK["evaluation_tasks"]
            or sha(source / "SeeNav" / "envs" / "datasets" / "base.json") != sha(data)):
        raise ValueError("The official 60-task evaluation file does not match its lock.")
    return source


def fetch_source(source, direct=False):
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="seenav-fetch-", dir=source.parent) as temporary:
            staging = Path(temporary) / "checkout"
            staging.mkdir()
            git(staging, "init")
            git(staging, "config", "core.autocrlf", "false")
            git(staging, "remote", "add", "origin", LOCK["repository"])
            git(staging, "fetch", "--depth", "1", "origin", LOCK["commit"], direct=direct)
            git(staging, "checkout", "--detach", LOCK["commit"])
            verify_source(staging)
            staging.rename(source)
    verify_source(source)


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Upstream launcher changed: expected exactly one {old!r}")
    return text.replace(old, new, 1)


def four_gpu_launcher(native):
    """The single training recipe difference."""
    return replace_one(native, OLD_GPU, NEW_GPU)


def isolated_paths(script):
    """Redirect only file locations; keep the native program and its options."""
    line_continuation = " " + chr(92) + "\n"
    data_root = '${SEENAV_RUN_DIR}/data'
    # Preserve the official preprocessor and 8/8 rows; only its output path changes.
    script = replace_one(script, "    --val_data_size $val_data_size\n",
                         '    --val_data_size $val_data_size' + line_continuation
                         + f'    --local_dir "{data_root}"\n')
    for key, filename in (("train", "train.parquet"), ("val", "test.parquet")):
        arg = f"data.{key}_files"
        script = replace_one(script, f"{arg}=$HOME/data/verl-agent/visual/{filename}",
                             f'{arg}="{data_root}/visual/{filename}"')
    script = replace_one(script, "data.metrics_save_dir=your_metrics_save_path",
                         'data.metrics_save_dir="${SEENAV_RUN_DIR}/metrics"')
    return replace_one(script, "    trainer.val_before_train=True $@\n",
                       '    trainer.default_local_dir="${SEENAV_RUN_DIR}/checkpoints"' + line_continuation
                       + "    trainer.val_before_train=True $@\n")


def gpu_ids(requested, visible):
    actual = requested if requested is not None else visible
    if not actual:
        raise ValueError("Specify four allocated devices with --gpus or CUDA_VISIBLE_DEVICES.")
    ids = [value.strip() for value in actual.split(",")]
    if (len(ids) != GPU_COUNT or len(set(ids)) != GPU_COUNT
            or any(not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", value) for value in ids)):
        raise ValueError("Four distinct full-GPU indices or CUDA UUIDs are required.")
    if visible is not None and not set(ids).issubset({part.strip() for part in visible.split(",")}):
        raise ValueError("--gpus cannot select devices outside CUDA_VISIBLE_DEVICES.")
    return ids


def hardware_check(env):
    code = """
import json
import torch
devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
if len(devices) != 4 or any("A100" not in d.name or d.total_memory < 70 * 1024**3 for d in devices):
    raise RuntimeError("Expected four visible A100 80GB GPUs.")
print(json.dumps([{"name": d.name, "total_bytes": d.total_memory} for d in devices]))
"""
    return json.loads(subprocess.check_output([sys.executable, "-c", code], env=env, text=True))


def write_new(path, content):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def package_versions():
    packages = {}
    for name in ("torch", "torchvision", "vllm", "transformers", "tensordict",
                 "datasets", "ray", "numpy", "ai2thor", "flash-attn"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def launch(source, out, gpus):
    if not sys.platform.startswith("linux"):
        raise ValueError("Native SeeNav training requires the documented Linux/CUDA environment.")
    if out.exists() or out.is_relative_to(source):
        raise ValueError("Choose a new output directory outside the official checkout.")
    # The upstream launcher passes paths through Hydra, whose CLI has special
    # meanings for these characters. Keep the path explicit instead of guessing.
    if any(c in str(out) for c in (" ", "\t", "\r", "\n", ",", "=", "[", "]", "{", "}", '"', "'", "\\")):
        raise ValueError("Choose an output path without whitespace or Hydra metacharacters.")
    ids = gpu_ids(gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(ids),
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": str(source / "verl-agent"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "RAY_ADDRESS": "local",
        "SEENAV_RUN_DIR": str(out),
    }
    devices = hardware_check(env)
    native_path = source / LOCK["train_launcher"]
    native = native_path.read_text(encoding="utf-8")
    card_only = four_gpu_launcher(native)
    runner = isolated_paths(card_only)
    out.mkdir(parents=True, exist_ok=False)
    (out / "data" / "visual").mkdir(parents=True)
    (out / "metrics").mkdir()
    write_new(out / "official-4gpu.sh", card_only)
    write_new(out / "launch.sh", runner)
    write_new(out / "binding.json", json.dumps({
        "official_commit": LOCK["commit"],
        "method": LOCK["method"],
        "official_launcher_sha256": sha(native_path),
        "four_gpu_launcher_sha256": sha(out / "official-4gpu.sh"),
        "effective_launcher_sha256": sha(out / "launch.sh"),
        "card_override": {"trainer.n_gpus_per_node": GPU_COUNT},
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "gpus": devices,
        "model_id": LOCK["base_model"],
        "base_tasks_sha256": LOCK["evaluation_task_sha256"],
    }, indent=2) + "\n")
    print(f"Running official SeeNav {LOCK['method']}; follow {out / 'train.log'}", flush=True)
    # bash -e reports a failed native preparation command before starting training.
    with (out / "train.log").open("xb") as log:
        result = subprocess.run(["bash", "-e", str(out / "launch.sh")],
                                cwd=source / "verl-agent", env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    write_new(out / "exit.json", json.dumps({"returncode": result.returncode}) + "\n")
    verify_source(source)
    if result.returncode:
        raise SystemExit(f"Official launcher exited with code {result.returncode}; inspect {out / 'train.log'}")
    print(f"Native launcher exited successfully; see {out / 'metrics'} and {out / 'checkpoints'}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "check", "plan", "train"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Optional already-cloned, clean pinned public source.")
    parser.add_argument("--direct", action="store_true", help="Use a direct HTTPS GitHub fetch (fetch only).")
    parser.add_argument("--gpus", help="Four allocated GPU IDs; default: scheduler's CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--output", type=Path, help="New experiment output directory for train.")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.command == "fetch":
        fetch_source(source, args.direct)
        print(f"Verified SeeNav-Agent {LOCK['commit']}")
        return
    verify_source(source)
    native = (source / LOCK["train_launcher"]).read_text(encoding="utf-8")
    if args.command == "check":
        print(json.dumps({"official_commit": LOCK["commit"], "method": LOCK["method"], "gpu_count": GPU_COUNT,
            "base_eval_tasks": LOCK["evaluation_tasks"], "base_eval_sha256": LOCK["evaluation_task_sha256"],
            "check": "source/config only; no simulator or training run"}, indent=2))
    elif args.command == "plan":
        four_gpu = four_gpu_launcher(native)
        print("".join(difflib.unified_diff(native.splitlines(keepends=True), four_gpu.splitlines(keepends=True),
                   fromfile="official", tofile="four_gpu")), end="")
        print("\n# Executed launcher (absolute data/output paths supplied by SEENAV_RUN_DIR)")
        print(isolated_paths(four_gpu), end="")
    else:
        output = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime("seenav-%Y%m%dT%H%M%SZ")
        launch(source, output.resolve(), args.gpus)


if __name__ == "__main__":
    main()
