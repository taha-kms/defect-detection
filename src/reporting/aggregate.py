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
            v = v.strip()
            try:
                # try float first, otherwise keep raw string
                result[k.strip()] = float(v)
            except ValueError:
                result[k.strip()] = v

    required = {"image_auroc", "pixel_auroc", "auprc", "pro"}
    if not required.issubset(result.keys()):
        return None
    return result



def collect_metrics(models: List[str], classes: List[str]) -> List[Dict[str, str | float]]:
    rows: List[Dict[str, str | float]] = []
    for model in models:
        for cls in classes:
            candidates = [
                env.RUNS_DIR / model / cls / "eval" / "metrics.txt",
                env.RUNS_DIR / model / cls / "latest" / "eval" / "metrics.txt",
            ]
            # also scan run directories
            runs_dir = env.RUNS_DIR / model / cls / "runs"
            if runs_dir.exists():
                for run in runs_dir.iterdir():
                    candidates.append(run / "eval" / "metrics.txt")

            for path in candidates:
                mdict = _parse_metrics_file(path)
                if mdict is not None:
                    row = {"model": model, "class": cls}
                    row.update(mdict)
                    rows.append(row)
                    break
    return rows



def global_summary(rows: List[Dict[str, str | float]]) -> Dict[str, float]:
    metrics = ["image_auroc", "pixel_auroc", "auprc", "pro"]
    out = {}
    for m in metrics:
        vals = [float(r[m]) for r in rows if m in r]
        out[m] = sum(vals) / max(1, len(vals))
    return out




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


def write_markdown(rows: List[Dict[str, str | float]],
                   summary: Dict[str, Dict[str, float]],
                   out_md: Path):
    """
    Generate a Markdown report with:
      - Per-class results (all metrics found)
      - Per-model averages
      - Global averages (across all rows)
      - Best models per metric
      - Links to plots/qualitative images if available
    """
    out_md.parent.mkdir(parents=True, exist_ok=True)
    models = sorted(summary.keys())
    metrics = ["image_auroc", "pixel_auroc", "auprc", "pro"]

    # --------- Best per metric ---------
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

        # --------- Per-class table ---------
        f.write("## Per-class results\n\n")

        # gather all possible columns
        all_fields = sorted({k for r in rows for k in r.keys()})
        # put model, class first
        field_order = ["model", "class"] + [k for k in all_fields if k not in ("model", "class")]

        # header
        f.write("| " + " | ".join(field_order) + " |\n")
        f.write("|" + "|".join([":---:" for _ in field_order]) + "|\n")

        for r in sorted(rows, key=lambda x: (x["model"], x["class"])):
            vals = []
            for k in field_order:
                v = r.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")

        # --------- Per-model averages ---------
        f.write("\n## Per-model averages (across classes)\n\n")
        f.write("| Model | " + " | ".join(m.title().replace("_", " ") for m in metrics) + " |\n")
        f.write("|" + "|".join([":---:" for _ in range(len(metrics)+1)]) + "|\n")
        for m in models:
            s = summary[m]
            f.write(f"| {m} | " + " | ".join(f"{s[k]:.4f}" for k in metrics) + " |\n")

        # --------- Global averages ---------
        f.write("\n## Global averages (all models & classes)\n\n")
        global_vals = {}
        for met in metrics:
            vals = [float(r[met]) for r in rows if met in r]
            global_vals[met] = sum(vals) / max(1, len(vals))
        f.write("| " + " | ".join(m.title().replace("_", " ") for m in metrics) + " |\n")
        f.write("|" + "|".join([":---:" for _ in metrics]) + "|\n")
        f.write("| " + " | ".join(f"{global_vals[m]:.4f}" for m in metrics) + " |\n")

        # --------- Best models ---------
        f.write("\n**Best models by metric**\n\n")
        for met, (bm, bv) in best.items():
            f.write(f"- {met}: **{bm}** ({bv:.4f})\n")

        # --------- Optional visual references ---------
        f.write("\n## Visual Diagnostics\n\n")
        for r in sorted(rows, key=lambda x: (x["model"], x["class"])):
            model, cls = r["model"], r["class"]
            run_dir = env.RUNS_DIR / model / cls / "latest" / "eval"
            roc_path = run_dir / "roc_curve.png"
            pr_path = run_dir / "pr_curve.png"
            teaser_path = run_dir / "qualitative" / "teaser.png"

            f.write(f"### {model} – {cls}\n\n")
            if roc_path.exists():
                f.write(f"![ROC]({roc_path})\n\n")
            if pr_path.exists():
                f.write(f"![PR]({pr_path})\n\n")
            if teaser_path.exists():
                f.write(f"![Teaser]({teaser_path})\n\n")

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