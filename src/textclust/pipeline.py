"""End-to-end pipeline: load -> clean -> features -> model selection -> report."""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import clean, features, io, model, report

log = logging.getLogger(__name__)


@dataclass
class Config:
    """All settings are optional; None means "derive from the data"."""

    min_tokens: int = 2
    language: str | None = None          # "en", "fr", "none" or None (auto-detect)
    ngram_range: tuple = (1, 1)
    max_components: int = 100
    target_variance: float | None = None  # e.g. 0.5 keeps the fewest SVD components reaching 50%
    k_max_cap: int = 20
    fixed_k: int | None = None
    algorithms: tuple = ("kmeans", "hdbscan")
    max_noise: float = 0.3
    n_terms: int = 8
    n_examples: int = 3
    projection: bool = True
    random_state: int = 42


@dataclass
class Result:
    data: pd.DataFrame               # cleaned rows with a `cluster` column
    text_column: str
    label_column: str | None
    algorithm: str
    n_clusters: int
    labels: np.ndarray
    metrics: dict
    clusters: pd.DataFrame
    run_dir: Path | None
    profile: dict = field(default_factory=dict)


def run(csv_path, text_column=None, label_column=None, config: Config | None = None,
        output_dir="runs", save=True) -> Result:
    cfg = config or Config()
    loaded = io.load_csv(csv_path, text_column=text_column, label_column=label_column)
    data, clean_report = clean.clean_corpus(loaded.frame, loaded.text_column, cfg.min_tokens)
    if len(data) < 3:
        raise ValueError(f"only {len(data)} usable texts after cleaning; at least 3 are needed")
    lang = cfg.language or clean_report["language"]
    stop = None if lang == "none" else clean.stop_words_for(lang)
    feats = features.build_features(data["clean_text"], stop_words=stop, ngram_range=tuple(cfg.ngram_range),
                                    max_components=cfg.max_components, target_variance=cfg.target_variance,
                                    random_state=cfg.random_state)
    choice = model.select_model(feats.embedding, random_state=cfg.random_state, k_max_cap=cfg.k_max_cap,
                                max_noise=cfg.max_noise, algorithms=cfg.algorithms, fixed_k=cfg.fixed_k)
    labels = choice.labels
    data = data.assign(cluster=labels)

    external = None
    if loaded.label_column is not None:
        truth = data[loaded.label_column].astype(str).to_numpy()
        external = model.external_scores(truth, labels)
        log.info("external agreement with %r: %s", loaded.label_column, external)

    terms = report.top_terms(feats.tfidf, labels, feats.vectorizer, cfg.n_terms)
    examples = report.representatives(feats.embedding, labels, data[loaded.text_column], cfg.n_examples)
    table = report.cluster_table(labels, terms, examples)

    metrics = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "config": dataclasses.asdict(cfg),
        "data_profile": loaded.profile,
        "cleaning": clean_report,
        "language_used": lang,
        "features": feats.params,
        "chosen": {"algorithm": choice.algorithm, "n_clusters": choice.n_clusters,
                   "noise_ratio": round(choice.noise_ratio, 4), **choice.scores, **choice.params},
        "candidates": choice.candidates,
        "k_curve": choice.k_curve,
        "external_labels": external,
        "cluster_sizes": {int(r.cluster): int(r.size) for r in table.itertuples()},
        "top_terms": terms,
        "examples": examples,
    }

    run_dir = None
    if save:
        run_dir = Path(output_dir) / dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir.mkdir(parents=True, exist_ok=True)
        figs = [report.plot_k_curve(choice.k_curve, run_dir / "k_selection.png"),
                report.plot_sizes(table, run_dir / "cluster_sizes.png")]
        if cfg.projection:
            xy, info = report.project_2d(feats.embedding, cfg.random_state)
            metrics["projection"] = info
            figs.append(report.plot_projection(xy, labels, info, run_dir / "projection_2d.png"))
        for f in figs:
            if f is not None:
                plt.close(f)
        data.to_csv(run_dir / "assignments.csv", index=False)
        report.save_json(metrics, run_dir / "metrics.json")

    return Result(data, loaded.text_column, loaded.label_column, choice.algorithm, choice.n_clusters,
                  labels, metrics, table, run_dir, loaded.profile)
