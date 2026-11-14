#!/usr/bin/env bash
# Assemble the self-contained Hugging Face Space folder dist/space/ from the repository sources.
#   scripts/build_space.sh [MODEL_FILE]      (default: dist/kneeclf-hog-logreg.pkl)
# The model must exist first: `python -m kneeclf export RUN_DIR` or `python -m kneeclf download-model --dest dist`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/dist/kneeclf-hog-logreg.pkl}"
OUT="$ROOT/dist/space"

[ -f "$MODEL" ] || { echo "missing model file: $MODEL" >&2; exit 1; }
[ -f "$MODEL.sha256" ] || { echo "missing checksum file: $MODEL.sha256" >&2; exit 1; }
( cd "$(dirname "$MODEL")" && sha256sum -c "$(basename "$MODEL").sha256" >/dev/null ) \
  || { echo "checksum mismatch for $MODEL" >&2; exit 1; }

# Start from an empty generated folder (dist/ is git-ignored; never edit dist/space by hand).
if [ -d "$OUT" ]; then
  find "$OUT" -mindepth 1 -delete
fi
mkdir -p "$OUT/examples"
cp "$ROOT/app/app.py" "$ROOT/app/requirements.txt" "$ROOT/app/README.md" "$OUT/"
cp "$ROOT"/app/examples/*.png "$ROOT/app/examples/sample_texts.csv" "$ROOT/app/examples/ATTRIBUTION.md" "$OUT/examples/"
for pkg in kneeclf textclust; do
  mkdir -p "$OUT/$pkg"
  cp "$ROOT"/src/$pkg/*.py "$OUT/$pkg/"
done
cp "$MODEL" "$OUT/kneeclf-hog-logreg.pkl"
cp "$MODEL.sha256" "$OUT/kneeclf-hog-logreg.pkl.sha256"
cat > "$OUT/.gitattributes" <<'GA'
*.pkl filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
GA
echo "Space folder ready: $OUT"
du -sh "$OUT"
