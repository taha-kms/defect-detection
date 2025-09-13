from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math

import matplotlib.pyplot as plt

from src.utils import env


MetricDict = Dict[str, float]


def _parse_metrics_file(path: Path) -> MetricDict | None:
    if not path.exists():
        return None
    result: MetricDict = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                result[k.strip()] = float(v.strip())
            except ValueError:
                pass
    # sanity: required keys
    required = {"image_auroc", "pixel_auroc", "auprc", "pro"}
    if not required.issubset(result.keys()):
        return None
    return result


def collect_metrics(models: List[str], classes: List[str]) -> List[Dict[str, str | float]]:
    """
    Walk runs directory and gather metrics rows.
    Each row: {model, class, image_auroc, pixel_auroc, auprc, pro}
    """
    rows: List[Dict[str, str | float]] = []
    for model in models:
        for cls in classes:
            mpath = env.RUNS_DIR / model / cls / "eval" / "metrics.txt"
            mdict = _parse_metrics_file(mpath)
            if mdict is None:
                print(f"Missing or invalid metrics: {mpath}")
                continue
            row = {"model": model, "class": cls}
            row.update(mdict)
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, str | float]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "class", "image_auroc", "pixel_auroc", "auprc", "pro"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote CSV: {out_csv}")


def _avg(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def summarize(rows: List[Dict[str, str | float]]) -> Dict[str, Dict[str, float]]:
    """
    Returns per-model averages for each metric.
    {
      model: {image_auroc: x, pixel_auroc: y, auprc: z, pro: w}
    }
    """
    per_model: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        m = str(r["model"])
        per_model.setdefault(m, {"image_auroc": [], "pixel_auroc": [], "auprc": [], "pro": []})
        for k in ["image_auroc", "pixel_auroc", "auprc", "pro"]:
            per_model[m][k].append(float(r[k]))
    # average
    out: Dict[str, Dict[str, float]] = {}
    for m, md in per_model.items():
        out[m] = {k: _avg(v) for k, v in md.items()}
    return out


def write_markdown(rows: List[Dict[str, str | float]], summary: Dict[str, Dict[str, float]], out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    models = sorted(summary.keys())
    metrics = ["image_auroc", "pixel_auroc", "auprc", "pro"]

    # find best per metric
    best: Dict[str, Tuple[str, float]] = {}
    for met in metrics:
        best_model, best_val = None, -math.inf
        for m in models:
            v = summary[m][met]
            if v > best_val:
                best_model, best_val = m, v
        best[met] = (best_model, best_val)

    with open(out_md, "w") as f:
        f.write("# Experiment Summary\n\n")
        f.write(f"- Data root: `{env.DATA_DIR}`\n")
        f.write(f"- Runs root: `{env.RUNS_DIR}`\n\n")

        # Per-row table
        f.write("## Per-class results\n\n")
        f.write("| Model | Class | Image AUROC | Pixel AUROC | AUPRC | PRO |\n")
        f.write("|------:|:------|------------:|------------:|------:|----:|\n")
        for r in sorted(rows, key=lambda x: (x["model"], x["class"])):
            f.write(f"| {r['model']} | {r['class']} | {float(r['image_auroc']):.4f} | {float(r['pixel_auroc']):.4f} | {float(r['auprc']):.4f} | {float(r['pro']):.4f} |\n")

        # Summary
        f.write("\n## Per-model averages (across classes)\n\n")
        f.write("| Model | Image AUROC | Pixel AUROC | AUPRC | PRO |\n")
        f.write("|------:|------------:|------------:|------:|----:|\n")
        for m in models:
            s = summary[m]
            f.write(f"| {m} | {s['image_auroc']:.4f} | {s['pixel_auroc']:.4f} | {s['auprc']:.4f} | {s['pro']:.4f} |\n")

        f.write("\n**Best models by metric**\n\n")
        for met, (bm, bv) in best.items():
            f.write(f"- {met}: **{bm}** ({bv:.4f})\n")

    print(f"Wrote Markdown: {out_md}")


def plot_averages(summary: Dict[str, Dict[str, float]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(summary.keys())
    metrics = ["image_auroc", "pixel_auroc", "auprc", "pro"]

    for met in metrics:
        vals = [summary[m][met] for m in models]
        plt.figure()
        plt.bar(models, vals)
        plt.ylabel(met.replace("_", " ").title())
        plt.title(f"Average {met.replace('_', ' ').title()} by model")
        plt.xticks(rotation=15)
        plt.tight_layout()
        fig_path = out_dir / f"{met}.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved plot: {fig_path}")