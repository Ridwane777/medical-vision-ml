"""Adaptivity tests for textclust, built on generated temporary CSV files (unittest only)."""
import csv
import logging
import random
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textclust import Config, run  # noqa: E402
from textclust.clean import FRENCH_STOP_WORDS, clean_corpus, normalize  # noqa: E402
from textclust.features import tfidf_bounds  # noqa: E402
from textclust.io import load_csv  # noqa: E402

logging.disable(logging.CRITICAL)

EN_TOPICS = {
    "knee": "knee ache joint stiff kneecap swelling climbing stairs squat cartilage".split(),
    "skin": "rash itchy skin hives red spots welts blister eczema patches".split(),
    "breath": "breath cough wheeze lungs chest breathing winded asthma inhaler phlegm".split(),
    "head": "headache migraine throbbing temples dizzy vision nausea light forehead pounding".split(),
}
EN_FILLER = "i have been feeling the my and it is with when after for a this since".split()
FR_SENTENCES = [
    "je ressens une douleur dans le genou quand je monte les escaliers",
    "ma peau est rouge et elle me gratte depuis trois jours",
    "j'ai du mal à respirer et je tousse beaucoup la nuit",
    "le mal de tête est très fort et je vois des taches",
    "mon épaule est raide et je ne peux pas lever le bras",
]


def make_texts(n, seed=0, topics=EN_TOPICS, words=8):
    rng = random.Random(seed)
    names = list(topics)
    rows = []
    for i in range(n):
        t = names[i % len(names)]
        toks = rng.sample(topics[t], 5) + rng.sample(EN_FILLER, 4) + [str(rng.randint(1, 9))]
        rng.shuffle(toks)
        rows.append((t, " ".join(toks[:words + 2]).capitalize() + "."))
    return rows


class Tmp(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name)

    def tearDown(self):
        self._d.cleanup()

    def write(self, rows, name="data.csv", delimiter=",", encoding="utf-8", header=None):
        p = self.dir / name
        with open(p, "w", newline="", encoding=encoding) as f:
            w = csv.writer(f, delimiter=delimiter)
            if header:
                w.writerow(header)
            w.writerows(rows)
        return p


class TestLoading(Tmp):
    def rows(self, n=40):
        return [(i, t, txt) for i, (t, txt) in enumerate(make_texts(n), 1)]

    def test_header_and_no_header(self):
        with_h = load_csv(self.write(self.rows(), "h.csv", header=["id", "topic", "text"]))
        self.assertTrue(with_h.profile["header"])
        self.assertEqual(with_h.text_column, "text")
        self.assertEqual(len(with_h.frame), 40)
        no_h = load_csv(self.write(self.rows(), "n.csv"))
        self.assertFalse(no_h.profile["header"])
        self.assertEqual(no_h.text_column, "col2")
        self.assertEqual(len(no_h.frame), 40)

    def test_delimiters(self):
        for d in [",", ";", "\t"]:
            with self.subTest(delimiter=d):
                ld = load_csv(self.write(self.rows(), f"d{ord(d)}.csv", delimiter=d, header=["id", "topic", "text"]))
                self.assertEqual(ld.profile["delimiter"], d)
                self.assertEqual(ld.text_column, "text")

    def test_latin1_encoding(self):
        rows = [(i, s + " é à ç") for i, s in enumerate(FR_SENTENCES * 4, 1)]
        ld = load_csv(self.write(rows, encoding="latin-1"))
        self.assertEqual(ld.profile["encoding"], "latin-1")
        self.assertIn("é", ld.frame[ld.text_column].iloc[0])

    def test_ragged_rows(self):
        rows = self.rows(30)
        ragged = [r if i % 3 else (r[0], r[2]) for i, r in enumerate(rows)]  # every third row lost a field
        ld = load_csv(self.write(ragged))
        self.assertEqual(ld.profile["ragged_rows_repaired"]["short"], 10)
        self.assertEqual(list(ld.frame[ld.text_column]), [r[2] for r in rows])

    def test_constant_column_is_not_a_label(self):
        rows = [(i, "0x000000", txt) for i, (_, txt) in enumerate(make_texts(60), 1)]
        ld = load_csv(self.write(rows))
        self.assertIsNone(ld.label_column)
        self.assertIn("col1", ld.profile["constant_columns"])

    def test_text_column_override(self):
        rows = [(txt, f"note {i % 3}") for i, (_, txt) in enumerate(make_texts(30))]
        path = self.write(rows, header=["body", "note"])
        self.assertEqual(load_csv(path).text_column, "body")
        self.assertEqual(load_csv(path, text_column="note").text_column, "note")
        self.assertEqual(load_csv(path, text_column=1).text_column, "note")


class TestCleaning(unittest.TestCase):
    def test_normalize_keeps_accents(self):
        self.assertEqual(normalize("J'ai 3 DOULEURS à l'épaule! https://x.org"), "j ai douleurs à l épaule")

    def test_language_detection(self):
        import pandas as pd
        en = pd.DataFrame({"t": [txt for _, txt in make_texts(20)]})
        self.assertEqual(clean_corpus(en, "t")[1]["language"], "en")
        fr = pd.DataFrame({"t": FR_SENTENCES * 3})
        _, rep = clean_corpus(fr, "t")
        self.assertEqual(rep["language"], "fr")
        self.assertEqual(rep["dropped_duplicates"], 10)

    def test_drops_empty_and_short(self):
        import pandas as pd
        df = pd.DataFrame({"t": ["", None, "ok", "a real sentence here", "a real sentence here!"]})
        out, rep = clean_corpus(df, "t", min_tokens=2)
        self.assertEqual(len(out), 1)
        self.assertEqual((rep["dropped_empty"], rep["dropped_too_short"], rep["dropped_duplicates"]), (2, 1, 1))


class TestPipeline(Tmp):
    def test_french_corpus_uses_french_stop_words(self):
        rows = [(i, f"{s} jour {i}") for i, s in enumerate(FR_SENTENCES * 8)]
        res = run(self.write(rows), save=False)
        self.assertEqual(res.metrics["language_used"], "fr")
        vocab = set(res.metrics["top_terms"][0]) | {w for t in res.metrics["top_terms"].values() for w in t}
        self.assertFalse(vocab & FRENCH_STOP_WORDS)

    def test_tiny_corpus_shrinks_parameters(self):
        rows = [(i, txt) for i, (_, txt) in enumerate(make_texts(15), 1)]
        res = run(self.write(rows), output_dir=self.dir / "runs")
        self.assertEqual(res.metrics["features"]["min_df"], 1)
        self.assertLessEqual(res.metrics["features"]["svd_components"], 14)
        self.assertGreaterEqual(res.n_clusters, 2)
        self.assertEqual(res.metrics["projection"]["method"], "PCA")
        self.assertTrue((res.run_dir / "metrics.json").exists())
        self.assertEqual(tfidf_bounds(15), (1, 0.95))

    def test_larger_corpus_recovers_topics(self):
        rows = [(i, "0x000000", t, txt) for i, (t, txt) in enumerate(make_texts(600, seed=1), 1)]
        path = self.write(rows, header=["id", "code", "topic", "testimony"])
        res = run(path, config=Config(projection=False), output_dir=self.dir / "runs")
        self.assertEqual(res.text_column, "testimony")
        self.assertEqual(res.label_column, "topic")
        self.assertGreater(res.metrics["external_labels"]["ari"], 0.5)
        for name in ("metrics.json", "assignments.csv", "k_selection.png", "cluster_sizes.png"):
            self.assertTrue((res.run_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
