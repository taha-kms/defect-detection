import argparse
from pathlib import Path
from src.reporting.aggregate import collect_metrics, write_csv, summarize, write_markdown, plot_averages


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics and generate a report")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--out", type=str, default="runs/summary")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_metrics(args.models, args.classes)
    if not rows:
        print("No metrics found to aggregate. Did you run evaluation?")
        return 1

    write_csv(rows, out_dir / "summary.csv")
    summ = summarize(rows)
    write_markdown(rows, summ, out_dir / "summary.md")
    plot_averages(summ, out_dir / "plots")

    return 0 