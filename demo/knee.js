// Knee X-ray demo model in plain JavaScript: HOG -> standardization -> logistic regression.
// Port of src/kneeclf/portable.py (the Python reference); tests/test_demo_js.py checks both give the same score.
// Works in the browser (window.Knee) and in Node (module.exports) for the test.
(function (root) {
  "use strict";

  // PIL "L" conversion of RGBA pixels (ITU-R 601-2 luma, same integer rounding as Pillow).
  function rgbaToGray(rgba, n) {
    const g = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const r = rgba[4 * i], gr = rgba[4 * i + 1], b = rgba[4 * i + 2];
      g[i] = (r * 19595 + gr * 38470 + b * 7471 + 0x8000) >> 16;
    }
    return g;
  }

  // Area-average (box) resize of a uint8 gray image to side x side. Exact Pillow BOX output is not
  // guaranteed for every size (rounding of fractional box edges), the result differs by at most 1 gray level.
  function resizeBox(gray, w, h, side) {
    if (w === side && h === side) return gray;
    const out = new Uint8Array(side * side);
    const sx = w / side, sy = h / side;
    for (let oy = 0; oy < side; oy++) {
      const y0 = oy * sy, y1 = y0 + sy;
      for (let ox = 0; ox < side; ox++) {
        const x0 = ox * sx, x1 = x0 + sx;
        let acc = 0, wsum = 0;
        for (let y = Math.floor(y0); y < Math.min(h, Math.ceil(y1)); y++) {
          const wy = Math.min(y + 1, y1) - Math.max(y, y0);
          for (let x = Math.floor(x0); x < Math.min(w, Math.ceil(x1)); x++) {
            const wx = Math.min(x + 1, x1) - Math.max(x, x0);
            acc += gray[y * w + x] * wx * wy;
            wsum += wx * wy;
          }
        }
        out[oy * side + ox] = Math.min(255, Math.max(0, Math.round(acc / wsum)));
      }
    }
    return out;
  }

  // uint8 gray (inputSide^2) -> Float64Array (size^2) in [0, 1], block mean (inputSide must be a multiple).
  function preprocess(gray, w, h, model) {
    const sIn = model.input_side, s = model.image_size;
    const g = resizeBox(gray, w, h, sIn);
    const f = sIn / s;
    if (!Number.isInteger(f)) throw new Error("input_side must be a multiple of image_size");
    const out = new Float64Array(s * s);
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        let acc = 0;
        for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) acc += g[(y * f + dy) * sIn + x * f + dx] / 255;
        out[y * s + x] = acc / (f * f);
      }
    }
    return out;
  }

  function mod(a, m) { return ((a % m) + m) % m; }

  // HOG with linear bin interpolation and L2-Hys block normalization (same layout as numpy hog_batch).
  function hog(img, side, p) {
    const cell = p.cell, block = p.block, bins = p.bins, clip = p.clip === undefined ? 0.2 : p.clip;
    const nc = Math.floor(side / cell);
    const hist = new Float64Array(nc * nc * bins);
    for (let y = 0; y < nc * cell; y++) {
      for (let x = 0; x < nc * cell; x++) {
        const gx = (x > 0 && x < side - 1) ? img[y * side + x + 1] - img[y * side + x - 1] : 0;
        const gy = (y > 0 && y < side - 1) ? img[(y + 1) * side + x] - img[(y - 1) * side + x] : 0;
        const mag = Math.hypot(gx, gy);
        const ang = mod(Math.atan2(gy, gx) * 180 / Math.PI, 180);
        const pos = ang / (180 / bins) - 0.5;
        const fl = Math.floor(pos);
        const w1 = pos - fl;
        const b0 = mod(fl, bins), b1 = (b0 + 1) % bins;
        const base = (Math.floor(y / cell) * nc + Math.floor(x / cell)) * bins;
        hist[base + b0] += mag * (1 - w1);
        hist[base + b1] += mag * w1;
      }
    }
    const nb = nc - block + 1, len = block * block * bins;
    const feats = new Float64Array(nb * nb * len);
    const eps2 = 1e-12;
    for (let by = 0; by < nb; by++) {
      for (let bx = 0; bx < nb; bx++) {
        const off = (by * nb + bx) * len;
        let k = 0;
        for (let i = 0; i < block; i++) for (let j = 0; j < block; j++) {
          const base = ((by + i) * nc + bx + j) * bins;
          for (let b = 0; b < bins; b++) feats[off + k++] = hist[base + b];
        }
        let ss = 0;
        for (let t = 0; t < len; t++) ss += feats[off + t] ** 2;
        let n = Math.sqrt(ss + eps2);
        ss = 0;
        for (let t = 0; t < len; t++) { feats[off + t] = Math.min(feats[off + t] / n, clip); ss += feats[off + t] ** 2; }
        n = Math.sqrt(ss + eps2);
        for (let t = 0; t < len; t++) feats[off + t] /= n;
      }
    }
    return { feats, hist, nc };
  }

  function decision(img, model) {
    const { feats } = hog(img, model.image_size, model.hog);
    const c = model.logreg.coef, sc = model.scaler;
    if (feats.length !== c.length) throw new Error("feature length mismatch");
    let z = model.logreg.intercept;
    for (let i = 0; i < c.length; i++) {
      let v = feats[i];
      if (sc && sc.mean) v -= sc.mean[i];
      if (sc && sc.scale) v /= sc.scale[i];
      z += v * c[i];
    }
    return z;
  }

  function predict(gray, w, h, model) {
    const img = preprocess(gray, w, h, model);
    const z = decision(img, model);
    const positive = z >= model.threshold;
    return {
      img, decision_score: z, score: 1 / (1 + Math.exp(-z)),
      threshold: 1 / (1 + Math.exp(-model.threshold)), positive,
      label: model.class_names[positive ? 1 : 0],
    };
  }

  // Per-cell orientation histograms for the display only (hard binning, as in kneeclf/viz.py).
  function cellHistograms(img, side, cell, bins) {
    const n = Math.floor(side / cell), H = new Float64Array(n * n * bins);
    for (let y = 0; y < n * cell; y++) for (let x = 0; x < n * cell; x++) {
      const gx = (x > 0 && x < side - 1) ? img[y * side + x + 1] - img[y * side + x - 1] : 0;
      const gy = (y > 0 && y < side - 1) ? img[(y + 1) * side + x] - img[(y - 1) * side + x] : 0;
      const ori = mod(Math.atan2(gy, gx) * 180 / Math.PI, 180);
      const b = Math.min(bins - 1, Math.floor(ori / (180 / bins)));
      H[(Math.floor(y / cell) * n + Math.floor(x / cell)) * bins + b] += Math.hypot(gx, gy);
    }
    return { H, n };
  }

  const api = { rgbaToGray, resizeBox, preprocess, hog, decision, predict, cellHistograms };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Knee = api;
})(typeof window !== "undefined" ? window : globalThis);
