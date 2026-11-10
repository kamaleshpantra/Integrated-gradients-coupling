"""
run_all_experiments.py  -  Run all remaining experiments sequentially, then analyze.

Run this AFTER sst2_main_seed42.json is confirmed complete.

Usage:
    python project/src/run_all_experiments.py
"""

import os
import sys
import json
import subprocess
import csv
from datetime import datetime

BASE = "c:/Users/91798/OneDrive/Desktop/Research"
SRC  = os.path.join(BASE, "project", "src")
DATA = os.path.join(BASE, "project", "data")
RES  = os.path.join(BASE, "project", "results")
RUN_LOG = os.path.join(BASE, "project", "run_log.csv")

os.makedirs(RES, exist_ok=True)

def log_run(config, seed, output_file, status, elapsed=None, notes=""):
    row = {
        "date": datetime.now().isoformat(),
        "config": config,
        "seed": seed,
        "output_file": output_file,
        "status": status,
        "elapsed_seconds": elapsed or "",
        "notes": notes,
    }
    write_header = not os.path.exists(RUN_LOG)
    with open(RUN_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"  [run_log] {status}: {output_file}")


def run_experiment(args_list, output_file, config_desc, seed):
    """Run a subprocess experiment, log result."""
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                d = json.load(f)
            n = len(d.get("records", []))
            if n > 0:
                print(f"  SKIP (already exists, {n} records): {output_file}")
                log_run(config_desc, seed, output_file, "SKIPPED", notes=f"already_exists_n={n}")
                return True
        except Exception:
            pass

    print(f"\n>>> Running: {config_desc} (seed={seed})")
    cmd = [sys.executable] + args_list
    import time
    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE)
    elapsed = time.time() - t0
    status = "SUCCESS" if result.returncode == 0 else f"FAILED(rc={result.returncode})"
    log_run(config_desc, seed, output_file, status, elapsed)
    if result.returncode != 0:
        print(f"  ERROR: {config_desc} exited with code {result.returncode}")
        return False
    return True


def main():
    eval_script = os.path.join(SRC, "evaluate_dataset.py")
    sst2_data   = os.path.join(DATA, "sst2_val.json")
    imdb_data   = os.path.join(DATA, "imdb_500.json")

    experiments = [
        # (description, args_list, output_file, seed)
        # SST-2 seed 42 (should already exist)
        ("sst2_150_steps20_seed42",
         [eval_script, "--dataset_path", sst2_data, "--num_samples", "150",
          "--steps", "20", "--temperature", "0.1", "--seed", "42",
          "--output_file", os.path.join(RES, "sst2_main_seed42.json")],
         os.path.join(RES, "sst2_main_seed42.json"), 42),

        # SST-2 seed 123
        ("sst2_150_steps20_seed123",
         [eval_script, "--dataset_path", sst2_data, "--num_samples", "150",
          "--steps", "20", "--temperature", "0.1", "--seed", "123",
          "--output_file", os.path.join(RES, "sst2_main_seed123.json")],
         os.path.join(RES, "sst2_main_seed123.json"), 123),

        # SST-2 seed 999
        ("sst2_150_steps20_seed999",
         [eval_script, "--dataset_path", sst2_data, "--num_samples", "150",
          "--steps", "20", "--temperature", "0.1", "--seed", "999",
          "--output_file", os.path.join(RES, "sst2_main_seed999.json")],
         os.path.join(RES, "sst2_main_seed999.json"), 999),

        # IMDB 100 samples seed 42
        ("imdb_100_steps20_seed42",
         [eval_script, "--dataset_path", imdb_data, "--num_samples", "100",
          "--steps", "20", "--temperature", "0.1", "--seed", "42",
          "--output_file", os.path.join(RES, "imdb_main_seed42.json")],
         os.path.join(RES, "imdb_main_seed42.json"), 42),

        # Ablation: SST-2 steps=50 (50 samples, seed 42)
        ("sst2_50samples_steps50_seed42",
         [eval_script, "--dataset_path", sst2_data, "--num_samples", "50",
          "--steps", "50", "--temperature", "0.1", "--seed", "42",
          "--output_file", os.path.join(RES, "ablation_steps50_seed42.json")],
         os.path.join(RES, "ablation_steps50_seed42.json"), 42),

        # Ablation: SST-2 steps=100 (50 samples, seed 42)
        ("sst2_50samples_steps100_seed42",
         [eval_script, "--dataset_path", sst2_data, "--num_samples", "50",
          "--steps", "100", "--temperature", "0.1", "--seed", "42",
          "--output_file", os.path.join(RES, "ablation_steps100_seed42.json")],
         os.path.join(RES, "ablation_steps100_seed42.json"), 42),
    ]

    failed = []
    for desc, args, out_file, seed in experiments:
        ok = run_experiment(args, out_file, desc, seed)
        if not ok:
            failed.append(desc)

    print("\n\n" + "="*60)
    if failed:
        print(f"FAILED experiments: {failed}")
        print("Fix failures before running analysis.")
    else:
        print("All experiments complete. Running analysis...")
        _run_analysis()

    print("Done.")


def _run_analysis():
    analyze_script = os.path.join(
        "c:/Users/91798/OneDrive/Desktop/Research", "project", "src", "analyze_results.py")

    sst2_files = [
        os.path.join(RES, "sst2_main_seed42.json"),
        os.path.join(RES, "sst2_main_seed123.json"),
        os.path.join(RES, "sst2_main_seed999.json"),
    ]
    # Filter to only existing files
    sst2_files = [f for f in sst2_files if os.path.exists(f)]

    if not sst2_files:
        print("No SST-2 result files found. Skipping analysis.")
        return

    cmd = [sys.executable, analyze_script,
           "--result_files"] + sst2_files + [
           "--output_prefix",
           os.path.join(RES, "analysis_sst2")]
    print(f"\n>>> Analyzing SST-2 results ({len(sst2_files)} seeds)...")
    result = subprocess.run(cmd, cwd="c:/Users/91798/OneDrive/Desktop/Research")
    if result.returncode != 0:
        print(f"  Analysis FAILED with code {result.returncode}")
        return

    print("\n>>> Generating figures...")
    fig_script = os.path.join(
        "c:/Users/91798/OneDrive/Desktop/Research", "project", "src", "generate_figures.py")
    cmd2 = [sys.executable, fig_script,
            "--analysis_prefix", os.path.join(RES, "analysis_sst2"),
            "--output_dir", "c:/Users/91798/OneDrive/Desktop/Research/project/figures"]
    result2 = subprocess.run(cmd2, cwd="c:/Users/91798/OneDrive/Desktop/Research")
    if result2.returncode != 0:
        print(f"  Figure generation FAILED with code {result2.returncode}")
        return

    print("\n>>> Writing LaTeX + Markdown paper draft...")
    paper_script = os.path.join(
        "c:/Users/91798/OneDrive/Desktop/Research", "project", "src", "write_paper.py")
    cmd3 = [sys.executable, paper_script,
            "--analysis_prefix", os.path.join(RES, "analysis_sst2"),
            "--output_dir", "c:/Users/91798/OneDrive/Desktop/Research/project/paper",
            "--draft_file", "c:/Users/91798/OneDrive/Desktop/Research/project/paper_draft.md"]
    result3 = subprocess.run(cmd3, cwd="c:/Users/91798/OneDrive/Desktop/Research")
    if result3.returncode != 0:
        print(f"  Paper writing FAILED with code {result3.returncode}")


if __name__ == "__main__":
    main()
