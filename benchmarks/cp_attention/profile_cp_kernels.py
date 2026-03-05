#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
CP Kernel Profiler — nsys-based kernel breakdown for CP attention benchmarks.

Launches bench_cp_pytorch.py under nsys, exports GPU trace CSV, and categorizes
kernels into cuDNN attention, CP support (thd_*), NCCL communication, and other.

Usage:
  python profile_cp_kernels.py --config llama3_8b --seq_len 131072 --ngpus 8 --qkv_format bshd
  python profile_cp_kernels.py --config llama3_8b --seq_len 131072 --ngpus 8 --qkv_format thd
  python profile_cp_kernels.py --config llama3_8b --seq_len 131072 --ngpus 8 --compare
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


BENCH_DIR = Path(__file__).parent
BENCH_SCRIPT = BENCH_DIR / "bench_cp_pytorch.py"

KERNEL_CATEGORIES = {
    "cudnn": "cuDNN Attention",
    "thd_": "CP Support (THD)",
    "nccl": "NCCL Communication",
    "flash": "Flash Attention",
}


def parse_args():
    parser = argparse.ArgumentParser(description="CP Kernel Profiler (nsys)")
    parser.add_argument("--config", type=str, default="llama3_8b")
    parser.add_argument("--seq_len", type=int, default=131072)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--qkv_format", type=str, default="bshd",
                        choices=["bshd", "thd"])
    parser.add_argument("--compare", action="store_true",
                        help="Run both BSHD and THD and compare kernel breakdown")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--master_port", type=int, default=29700)
    parser.add_argument("--baseline", action="store_true",
                        help="Also profile no-CP baseline")
    return parser.parse_args()


def run_nsys_profile(config, seq_len, batch_size, ngpus, qkv_format,
                     output_prefix, master_port, baseline=False):
    """Launch benchmark under nsys profile, return path to .nsys-rep file."""
    nsys_rep = f"{output_prefix}.nsys-rep"

    cmd = [
        "nsys", "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        f"--output={output_prefix}",
        "--force-overwrite=true",
        "torchrun",
        f"--nproc-per-node={ngpus}",
        f"--master-port={master_port}",
        str(BENCH_SCRIPT),
        f"--config={config}",
        f"--seq_len={seq_len}",
        f"--batch_size={batch_size}",
        f"--qkv_format={qkv_format}",
        "--profile",
    ]
    if baseline:
        cmd.append("--baseline")

    print(f"\n>>> Running nsys profile: {qkv_format}, {ngpus} GPUs, S={seq_len}")
    print(f"    Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd="/tmp", capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: nsys profile failed with exit code {result.returncode}")
        return None

    return nsys_rep


def export_gpu_trace(nsys_rep, output_prefix):
    """Export GPU kernel trace from .nsys-rep to CSV."""
    csv_path = f"{output_prefix}_gpu_trace.csv"

    cmd = [
        "nsys", "stats",
        "-r", "cuda_gpu_trace",
        "--format", "csv",
        f"--output={output_prefix}",
        nsys_rep,
    ]

    print(f"    Exporting GPU trace to CSV...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: nsys stats failed: {result.stderr[:500]}")
        return None

    # nsys stats appends _cuda_gpu_trace.csv
    expected = f"{output_prefix}_cuda_gpu_trace.csv"
    if os.path.exists(expected):
        return expected
    # Fallback: look for the file
    for f in Path(output_prefix).parent.glob("*gpu_trace*"):
        if f.suffix == ".csv":
            return str(f)

    print(f"ERROR: Could not find GPU trace CSV")
    return None


def categorize_kernel(name):
    """Categorize a kernel name into a group."""
    name_lower = name.lower()
    for pattern, category in KERNEL_CATEGORIES.items():
        if pattern in name_lower:
            return category
    return "Other"


def parse_gpu_trace(csv_path, device_filter=0):
    """Parse GPU trace CSV and return kernel breakdown by category.

    Filters to device_filter (default: GPU 0 / rank 0) to avoid
    double-counting in multi-GPU runs.
    """
    categories = defaultdict(lambda: {"count": 0, "total_ns": 0, "kernels": defaultdict(int)})

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter to one device
            device = row.get("Device", "")
            if device_filter is not None and str(device_filter) not in str(device):
                # Try matching on device ID field
                dev_id = row.get("DeviceId", row.get("Device Id", ""))
                if str(device_filter) not in str(dev_id):
                    continue

            name = row.get("Name", row.get("Kernel Name", ""))
            duration_ns = int(row.get("Duration (ns)", row.get("Duration", 0)))

            cat = categorize_kernel(name)
            categories[cat]["count"] += 1
            categories[cat]["total_ns"] += duration_ns
            # Store shortened kernel name
            short_name = name.split("(")[0].strip()[:80]
            categories[cat]["kernels"][short_name] += 1

    return dict(categories)


def print_breakdown(categories, label):
    """Print kernel time breakdown table."""
    total_ns = sum(c["total_ns"] for c in categories.values())
    if total_ns == 0:
        print(f"\n  No kernels captured for {label}")
        return

    total_ms = total_ns / 1e6

    print(f"\n{'='*70}")
    print(f"Kernel Breakdown: {label}")
    print(f"{'='*70}")
    print(f"{'Category':<25} {'Count':>6} {'Time (ms)':>10} {'%':>7}")
    print(f"{'-'*70}")

    for cat in sorted(categories.keys(), key=lambda c: categories[c]["total_ns"], reverse=True):
        c = categories[cat]
        ms = c["total_ns"] / 1e6
        pct = 100.0 * c["total_ns"] / total_ns
        print(f"{cat:<25} {c['count']:>6} {ms:>10.2f} {pct:>6.1f}%")

    print(f"{'-'*70}")
    print(f"{'TOTAL':<25} {sum(c['count'] for c in categories.values()):>6} {total_ms:>10.2f} {'100.0':>6}%")
    print(f"{'='*70}")

    # Top kernels per category
    for cat in sorted(categories.keys(), key=lambda c: categories[c]["total_ns"], reverse=True):
        c = categories[cat]
        if c["count"] == 0:
            continue
        top = sorted(c["kernels"].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  {cat} — top kernels:")
        for name, count in top:
            print(f"    [{count:>3}x] {name}")


def print_comparison(bshd_cats, thd_cats):
    """Print side-by-side BSHD vs THD kernel breakdown."""
    all_cats = sorted(set(list(bshd_cats.keys()) + list(thd_cats.keys())))

    bshd_total = sum(c["total_ns"] for c in bshd_cats.values()) or 1
    thd_total = sum(c["total_ns"] for c in thd_cats.values()) or 1

    print(f"\n{'='*80}")
    print(f"BSHD vs THD Kernel Comparison")
    print(f"{'='*80}")
    print(f"{'Category':<25} {'BSHD (ms)':>10} {'BSHD %':>7} {'THD (ms)':>10} {'THD %':>7} {'Delta':>8}")
    print(f"{'-'*80}")

    for cat in all_cats:
        bshd_ns = bshd_cats.get(cat, {}).get("total_ns", 0)
        thd_ns = thd_cats.get(cat, {}).get("total_ns", 0)
        bshd_ms = bshd_ns / 1e6
        thd_ms = thd_ns / 1e6
        bshd_pct = 100.0 * bshd_ns / bshd_total
        thd_pct = 100.0 * thd_ns / thd_total
        delta_ms = thd_ms - bshd_ms
        print(f"{cat:<25} {bshd_ms:>10.2f} {bshd_pct:>6.1f}% {thd_ms:>10.2f} {thd_pct:>6.1f}% {delta_ms:>+7.2f}")

    bshd_total_ms = bshd_total / 1e6
    thd_total_ms = thd_total / 1e6
    print(f"{'-'*80}")
    print(f"{'TOTAL':<25} {bshd_total_ms:>10.2f} {'100.0':>6}% {thd_total_ms:>10.2f} {'100.0':>6}% {thd_total_ms - bshd_total_ms:>+7.2f}")
    print(f"{'='*80}")

    # Highlight THD-specific overhead
    thd_support = thd_cats.get("CP Support (THD)", {}).get("total_ns", 0) / 1e6
    if thd_support > 0:
        print(f"\n  THD-specific overhead (thd_* kernels): {thd_support:.2f} ms")
        print(f"  This accounts for {100.0 * thd_support / (thd_total/1e6):.1f}% of THD total time")


def main():
    args = parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BENCH_DIR / f"profiles_{ts}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    formats = ["bshd", "thd"] if args.compare else [args.qkv_format]
    all_categories = {}

    for fmt in formats:
        prefix = output_dir / f"prof_{args.config}_{fmt}_{args.ngpus}gpu"

        nsys_rep = run_nsys_profile(
            args.config, args.seq_len, args.batch_size, args.ngpus, fmt,
            str(prefix), args.master_port, baseline=args.baseline,
        )
        args.master_port += 1  # avoid port conflict

        if nsys_rep is None:
            print(f"Skipping {fmt} — nsys profile failed")
            continue

        csv_path = export_gpu_trace(nsys_rep, str(prefix))
        if csv_path is None:
            print(f"Skipping {fmt} — GPU trace export failed")
            continue

        categories = parse_gpu_trace(csv_path)
        all_categories[fmt] = categories
        label = f"{fmt.upper()} | {args.config} | S={args.seq_len} | {args.ngpus} GPUs"
        print_breakdown(categories, label)

    if args.compare and "bshd" in all_categories and "thd" in all_categories:
        print_comparison(all_categories["bshd"], all_categories["thd"])

    print(f"\nProfiles saved in: {output_dir}")


if __name__ == "__main__":
    main()
