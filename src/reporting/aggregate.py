from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math
import json

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
            candidates: List[Path] = []

            # legacy paths
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
                if mdict is None:
                    continue

                # Derive clean run_id and run_path
                # path = .../<model>/<class>/(runs/<runid>|latest|eval)/eval/metrics.txt
                eval_dir = path.parent                    # .../eval
                parent = eval_dir.parent                  # .../runs/<runid>  OR .../latest OR .../<class>
                run_id = "unknown"
                run_path = str(parent)
                try:
                    if parent.name == "latest":
                        run_id = "latest"
                        run_json = parent / "run.json"
                        try:
                            if run_json.exists():
                                with open(run_json) as f:
                                    meta = json.load(f)
                            if "run_id" in meta:
                                        run_id = meta["run_id"]
                        except Exception: pass

                    elif parent.name == "eval":
                        # legacy: .../<class>/eval/metrics.txt
                        run_id = "legacy"
                    else:
                        # likely .../runs/<runid>
                        if parent.parent.name == "runs":
                            run_id = parent.name          # <runid>
                except Exception:
                    pass

                row: Dict[str, str | float] = {"model": model, "class": cls}
                row.update(mdict)          # merge parsed metrics first
                row["run_id"] = run_id     # then force the clean run_id
                row["run_path"] = run_path # keep full path for CSV traceability

                rows.append(row)
    return rows



def global_summary(rows: List[Dict[str, str | float]]) -> Dict[str, float]:
    # average over known metrics if present
    metrics = ["image_auroc", "pixel_auroc", "auprc", "pro", "latency_sec"]
    out = {}
    for m in metrics:
        vals = [float(r[m]) for r in rows if m in r]
        if vals:
            out[m] = sum(vals) / len(vals)
    return out



def write_csv(rows: List[Dict[str, str | float]], out_csv: Path):
    """
    Write all rows into a CSV file.
    Dynamically determines fieldnames so extra fields (run_id, threshold, etc.)
    are always included.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # gather all keys across all rows
    fieldnames = sorted({k for r in rows for k in r.keys()})

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote CSV: {out_csv}")


def summarize(rows: List[Dict[str, str | float]]) -> Dict[str, Dict[str, float]]:
    """
    Returns per-model averages for each metric (including latency_sec when present).
    """
    per_model: Dict[str, Dict[str, List[float]]] = {}
    wanted = ["image_auroc", "pixel_auroc", "auprc", "pro", "latency_sec"]

    for r in rows:
        m = str(r["model"])
        per_model.setdefault(m, {k: [] for k in wanted})
        for k in wanted:
            if k in r:  # <-- guard missing keys
                per_model[m][k].append(float(r[k]))

    # average
    out: Dict[str, Dict[str, float]] = {}
    for m, md in per_model.items():
        out[m] = {k: (sum(v) / len(v) if v else float("nan")) for k, v in md.items()}
    return out



def write_markdown(rows: List[Dict[str, str | float]], out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    core_metrics_accuracy = ["image_auroc", "pixel_auroc", "auprc", "pro"]
    extra_metrics = ["latency_sec"]

    # -------- Best overall (single row) --------
    best_overall = None
    best_score = -math.inf
    for r in rows:
        vals = [float(r[m]) for m in core_metrics_accuracy if m in r]
        if not vals:
            continue
        score = sum(vals) / len(vals)
        if score > best_score:
            best_overall = r
            best_score = score

    with open(out_md, "w") as f:
        f.write("# Experiment Summary\n\n")
        f.write(f"- Data root: `{env.DATA_DIR}`\n")
        f.write(f"- Runs root: `{env.RUNS_DIR}`\n\n")

        f.write("## 1. Best Overall\n\n")
        if best_overall:
            f.write(f"Best overall: **{best_overall['model']}** on **{best_overall['class']}**\n run ID: **{best_overall['run_id']}**\n\n")
            # print accuracy metrics
            for m in core_metrics_accuracy:
                if m in best_overall:
                    f.write(f"- {m}: {best_overall[m]:.4f}\n")
            # print latency if present
            for m in extra_metrics:
                if m in best_overall:
                    f.write(f"- {m}: {best_overall[m]:.4f}\n")
            # also dump any other extra fields
            for k, v in best_overall.items():
                if k not in ("model", "class", "run_id") and k not in (core_metrics_accuracy + extra_metrics):
                    f.write(f"- {k}: {v}\n")
            f.write("\n")

        # ---------- 2. Per-class info ----------
        f.write("## 2. Per-Class Information \n\n")
        classes = sorted(set(r["class"] for r in rows))
        for cls in classes:
            f.write(f"### Class: {cls}\n\n")
            cls_rows = [r for r in rows if r["class"] == cls]
            cls_rows.sort(key=lambda r: float(r.get("image_auroc", 0.0)), reverse=True)

            all_fields = sorted({k for r in cls_rows for k in r.keys()})
            field_order = ["run_id", "model", "class", "latency_sec"] + [k for k in all_fields if k not in ("model", "class", "run_id", "latency_sec")]

            f.write("| " + " | ".join(field_order) + " |\n")
            f.write("|" + "|".join([":---:" for _ in field_order]) + "|\n")
            for r in cls_rows:
                vals = []
                for k in field_order:
                    v = r.get(k, "")
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        # try numeric rendering if it's a numeric string
                        try:
                            vals.append(f"{float(v):.4f}")
                        except Exception:
                            vals.append(str(v))
                f.write("| " + " | ".join(vals) + " |\n")
            f.write("\n")

        # ---------- 3. Per-model info ----------
        f.write("## 3. Per-Model Information \n\n")
        models = sorted(set(r["model"] for r in rows))
        for m in models:
            f.write(f"### Model: {m}\n\n")
            model_rows = [r for r in rows if r["model"] == m]
            model_rows.sort(key=lambda r: float(r.get("image_auroc", 0.0)), reverse=True)

            all_fields = sorted({k for r in model_rows for k in r.keys()})
            field_order = ["run_id", "class", "latency_sec"] + [k for k in all_fields if k not in ("model", "class", "run_id", "latency_sec")]

            f.write("| " + " | ".join(field_order) + " |\n")
            f.write("|" + "|".join([":---:" for _ in field_order]) + "|\n")
            for r in model_rows:
                vals = []
                for k in field_order:
                    v = r.get(k, "")
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        try:
                            vals.append(f"{float(v):.4f}")
                        except Exception:
                            vals.append(str(v))
                f.write("| " + " | ".join(vals) + " |\n")
            f.write("\n")

    print(f"Wrote Markdown: {out_md}")


def plot_averages(summary: Dict[str, Dict[str, float]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(summary.keys())
    metrics = ["image_auroc", "pixel_auroc", "auprc", "pro", "latency_sec"]

    for met in metrics:
        vals = [summary.get(m, {}).get(met, float("nan")) for m in models]  # <-- safer
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