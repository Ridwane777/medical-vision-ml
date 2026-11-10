"""Thin wrapper kept for backward compatibility: runs the adaptive `textclust` pipeline.

Usage (from the repository root):
    python unsupervised_learning/unsupervised_learning.py [path/to/file.csv] [textclust options]
Equivalent to: PYTHONPATH=src python -m textclust path/to/file.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from textclust.__main__ import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if not any(a.lower().endswith(".csv") for a in args):  # default dataset and output folder
        args = [str(Path(__file__).with_name("Student_Dataset.csv")), "--output", str(ROOT / "runs")] + args
    main(args)
