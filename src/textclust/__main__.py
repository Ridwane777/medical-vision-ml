"""CLI: python -m textclust path/to/file.csv [--text-column C] [--label-column C] [--output DIR]."""
import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")  # the CLI only writes PNG files

from .pipeline import Config, run  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(prog="textclust", description="Cluster the texts of any CSV file.")
    p.add_argument("csv")
    p.add_argument("--text-column", help="name or 0-based index (default: auto-detected)")
    p.add_argument("--label-column", help="name or index of reference labels (default: auto-detected)")
    p.add_argument("--no-labels", action="store_true", help="disable label-column detection")
    p.add_argument("--output", default="runs", help="parent folder of the run directory")
    p.add_argument("--k", type=int, help="force the number of K-Means clusters")
    p.add_argument("--language", choices=["en", "fr", "none"], help="force the stop-word list")
    p.add_argument("--ngram-max", type=int, default=1, help="use n-grams up to this length")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    def idx(v):
        return int(v) if v is not None and v.isdigit() else v

    cfg = Config(fixed_k=a.k, language=a.language, ngram_range=(1, a.ngram_max))
    res = run(a.csv, text_column=idx(a.text_column), label_column=False if a.no_labels else idx(a.label_column),
              config=cfg, output_dir=a.output)
    ch = res.metrics["chosen"]
    print(json.dumps({"run_dir": str(res.run_dir), "algorithm": ch["algorithm"], "n_clusters": ch["n_clusters"],
                      "silhouette": round(ch["silhouette"], 4), "noise_ratio": ch["noise_ratio"],
                      "external_labels": res.metrics["external_labels"]}, indent=2))
    print(res.clusters.drop(columns="example").to_string(index=False))


if __name__ == "__main__":
    main()
