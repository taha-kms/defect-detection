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
    """
    Collect all metrics from runs/<model>/<class>/runs/<runid>/eval/metrics.txt
    (and also handle legacy latest/eval/metrics.txt if present).
    """
    rows: List[Dict[str, str | float]] = []
    for model in models:
        for cls in classes:
            candidates: List[Path] = []

            # legacy style
            candidates.append(env.RUNS_DIR / model / cls / "eval" / "metrics.txt")
            candidates.append(env.RUNS_DIR / model / cls / "latest" / "eval" / "metrics.txt")

            # all run directories
            runs_dir = env.RUNS_DIR / model / cls / "runs"
            if runs_dir.exists():
                for run in sorted(runs_dir.iterdir()):
                    mfile = run / "eval" / "metrics.txt"
                    if mfile.exists():
                        candidates.append(mfile)

            for path in candidates:
                mdict = _parse_metrics_file(path)
                if mdict is not None:
                    row = {"model": model, "class": cls}
                    row.update(mdict)
                    rows.append(row)
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


def write_markdown(rows: List[Dict[str, str | float]], out_md: Path):
    """
    Generate a Markdown report with:
      1. Best overall across all classes/models
      2. Per-class metrics (sorted by image_auroc)
      3. Per-model metrics (sorted by image_auroc)
      4. Visual diagnostics (ROC, PR, qualitative)
    """
    out_md.parent.mkdir(parents=True, exist_ok=True)
    core_metrics = ["image_auroc", "pixel_auroc", "auprc", "pro"]

    # -------- Best overall (single row) --------
    best_overall = None
    best_score = -math.inf
    for r in rows:
        score = sum(float(r[m]) for m in core_metrics if m in r) / len(core_metrics)
        if score > best_score:
            best_overall = r
            best_score = score

    with open(out_md, "w") as f:
        f.write("# Experiment Summary\n\n")
        f.write(f"- Data root: `{env.DATA_DIR}`\n")
        f.write(f"- Runs root: `{env.RUNS_DIR}`\n\n")

        # ---------- 1. Best overall ----------
        f.write("## 1. Best Overall\n\n")
        if best_overall:
            f.write(f"Best overall: **{best_overall['model']}** on **{best_overall['class']}**\n\n")
            for m in core_metrics:
                if m in best_overall:
                    f.write(f"- {m}: {best_overall[m]:.4f}\n")
            # also dump extra fields
            for k, v in best_overall.items():
                if k not in ("model", "class") and k not in core_metrics:
                    f.write(f"- {k}: {v}\n")
            f.write("\n")

        # ---------- 2. Per-class info ----------
        f.write("## 2. Per-Class Information (sorted)\n\n")
        classes = sorted(set(r["class"] for r in rows))
        for cls in classes:
            f.write(f"### Class: {cls}\n\n")
            cls_rows = [r for r in rows if r["class"] == cls]
            cls_rows.sort(key=lambda r: float(r.get("image_auroc", 0.0)), reverse=True)

            all_fields = sorted({k for r in cls_rows for k in r.keys()})
            field_order = ["model", "class"] + [k for k in all_fields if k not in ("model", "class")]

            f.write("| " + " | ".join(field_order) + " |\n")
            f.write("|" + "|".join([":---:" for _ in field_order]) + "|\n")
            for r in cls_rows:
                vals = []
                for k in field_order:
                    v = r.get(k, "")
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                f.write("| " + " | ".join(vals) + " |\n")
            f.write("\n")

        # ---------- 3. Per-model info ----------
        f.write("## 3. Per-Model Information (sorted)\n\n")
        models = sorted(set(r["model"] for r in rows))
        for m in models:
            f.write(f"### Model: {m}\n\n")
            model_rows = [r for r in rows if r["model"] == m]
            model_rows.sort(key=lambda r: float(r.get("image_auroc", 0.0)), reverse=True)

            all_fields = sorted({k for r in model_rows for k in r.keys()})
            field_order = ["class"] + [k for k in all_fields if k not in ("model", "class")]

            f.write("| " + " | ".join(field_order) + " |\n")
            f.write("|" + "|".join([":---:" for _ in field_order]) + "|\n")
            for r in model_rows:
                vals = []
                for k in field_order:
                    v = r.get(k, "")
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                f.write("| " + " | ".join(vals) + " |\n")
            f.write("\n")


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