#!/usr/bin/env bash
# Fill demo/assets/ (git-ignored) so demo/ can be served as a static site (GitHub Pages or any web server).
#   scripts/build_demo.sh
# Copies, so that src/ and app/examples/ stay the single source of truth:
#   - the portable JSON model: dist/kneeclf-hog-logreg.json if present, else downloaded from the latest GitHub
#     release; the SHA-256 is checked in both cases
#   - the example X-rays and the synthetic text sample from app/examples/
#   - the textclust Python modules, loaded by Gradio-Lite in the browser for the clustering tab
# Preview locally: python -m http.server -d demo 8000, then open http://localhost:8000/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/demo/assets"
NAME="kneeclf-hog-logreg.json"
RELEASE="https://github.com/Ridwane777/medical-vision-ml/releases/latest/download"

rm -rf "$OUT"
mkdir -p "$OUT/examples" "$OUT/py/textclust"

if [ -f "$ROOT/dist/$NAME" ]; then
  cp "$ROOT/dist/$NAME" "$ROOT/dist/$NAME.sha256" "$OUT/"
else
  echo "dist/$NAME not found: downloading it from the latest release"
  curl -fsSL -o "$OUT/$NAME" "$RELEASE/$NAME"
  curl -fsSL -o "$OUT/$NAME.sha256" "$RELEASE/$NAME.sha256"
fi
( cd "$OUT" && sha256sum -c "$NAME.sha256" >/dev/null ) || { echo "checksum mismatch for $NAME" >&2; exit 1; }
rm "$OUT/$NAME.sha256"

cp "$ROOT"/app/examples/kl*.png "$ROOT/app/examples/sample_texts.csv" "$ROOT/app/examples/ATTRIBUTION.md" "$OUT/examples/"
# Modules listed in the <gradio-file> tags of demo/index.html (the CLI entry point is not needed).
SHIPPED="__init__ clean features io model pipeline report"
for m in $SHIPPED; do
  cp "$ROOT/src/textclust/$m.py" "$OUT/py/textclust/"
done
for f in "$ROOT"/src/textclust/*.py; do
  m="$(basename "$f" .py)"
  case " $SHIPPED __main__ " in
    *" $m "*) ;;
    *) echo "textclust/$m.py is not shipped to the demo: add it to SHIPPED and to demo/index.html" >&2; exit 1 ;;
  esac
done
echo "Static demo ready: $ROOT/demo"
du -sh "$ROOT/demo"
