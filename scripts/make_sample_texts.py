"""Generate app/examples/sample_texts.csv: 150 SYNTHETIC short symptom descriptions over 5 topics.

No real patient text is used. Sentences are assembled from templates with a fixed seed, so the file is reproducible:
    python scripts/make_sample_texts.py
"""
import csv
import random
from pathlib import Path

TOPICS = {
    "skin": (["an itchy red rash", "raised hives", "dry flaky skin", "a burning rash", "small itchy bumps"],
             ["on my arms", "on my neck", "across my chest", "on both legs", "around my wrists"],
             ["after using a new soap", "since yesterday", "that gets worse at night", "after gardening",
              "that keeps spreading"]),
    "joints": (["a stiff aching knee", "a swollen ankle", "sharp pain in my hip", "a sore shoulder joint",
                "painful stiff fingers"],
               ["when I climb stairs", "every morning", "after a long walk", "when the weather is cold",
                "after kneeling"],
               ["and it clicks when I bend it", "and the joint feels warm", "and I limp a little",
                "and it eases after moving", "and it is hard to grip things"]),
    "breathing": (["a tight chest", "shortness of breath", "a dry cough", "wheezing", "a heavy feeling in my chest"],
                  ["when I run", "at night", "after climbing stairs", "in cold air", "when I lie down"],
                  ["and I cannot take a deep breath", "and my breathing sounds noisy", "and I feel out of breath",
                   "and the cough keeps me awake", "and my inhaler helps a bit"]),
    "digestion": (["stomach cramps", "bloating", "nausea", "heartburn", "an upset stomach"],
                  ["after eating", "after dairy", "in the morning", "after spicy food", "after big meals"],
                  ["with loose stools", "and I feel full quickly", "and a sour taste in my mouth",
                   "and I lose my appetite", "with gas and discomfort"]),
    "head": (["a throbbing headache", "a pounding migraine", "dizziness", "pressure behind my eyes",
              "a dull headache"],
             ["in the afternoon", "after screen work", "when I stand up quickly", "most mornings",
              "after poor sleep"],
             ["and bright light bothers me", "and I feel a bit sick", "and the room seems to spin",
              "and my neck feels tense", "and painkillers barely help"]),
}
OPENERS = ["I have", "I keep getting", "For a week I have had", "Lately I notice", "My main problem is"]


def main(n_per_topic=30, seed=7, out=Path(__file__).resolve().parents[1] / "app" / "examples" / "sample_texts.csv"):
    rng = random.Random(seed)
    rows, seen = [], set()
    for topic, (what, when, extra) in TOPICS.items():
        while sum(r[1] == topic for r in rows) < n_per_topic:
            text = f"{rng.choice(OPENERS)} {rng.choice(what)} {rng.choice(when)} {rng.choice(extra)}."
            if text not in seen:
                seen.add(text)
                rows.append((text, topic))
    rng.shuffle(rows)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["description", "topic"])
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
