# ADR-0004: Separate folder for experiments outside the library specification

- **Status**: Accepted
- **Date**: 2026-11-11
- **Deciders**: Ridwane777

## Context

The specification allows only scikit-learn, pandas, NumPy, Matplotlib, JupyterLab, Pillow and the standard library
(ADR-0003). The retrospective listed pretrained CNNs and sentence embeddings as the main alternatives, without numbers.
Measuring them needs PyTorch, torchvision and sentence-transformers, which the core must not depend on.

## Options considered

1. **Do not measure**: keep the alternatives as claims. Cheap, but unverifiable.
2. **Optional imports inside `src/`** (`try: import torch`). Blurs what the graded code depends on; a reviewer cannot
   tell at a glance that the core respects the specification.
3. **A separate `beyond_scope/` folder** with its own `requirements.txt` and results under
   `results/beyond_scope/`. It may import from `src/` (to reuse data loading, evaluation and model selection); `src/`
   never imports from it.

## Decision

Option 3. Same protocol as the core (label scheme, balanced train, threshold on validation for recall 0.85, test once;
same cleaned testimonies and same k/algorithm selection), so the numbers are comparable.

## Consequences

- ✅ The core stays within the specification and its tests do not need torch.
- ✅ The alternatives in the retrospective are backed by measured numbers.
- ❌ Two environments to maintain (`requirements.txt` and `beyond_scope/requirements.txt`).
- ❌ Pretrained weights are downloaded at run time (about 100 MB to 450 MB per model), not versioned.
