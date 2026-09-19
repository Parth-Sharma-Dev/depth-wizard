# GAMUS Dataset — Baseline 1

## Purpose

The first training experiment is:

```text
RGB → AGL
```

using a clean subset of GAMUS.

## Files

- `gamus_quality.py` — scene/pixel quality policy.
- `gamus_dataset.py` — PyTorch Dataset with 512×512 crops.
- `build_gamus_manifest.py` — creates the initial clean scene manifest.
- `smoke_test_gamus_dataset.py` — checks tensors before training.

## Initial Baseline 1A policy

For Baseline 1A, a scene is included when:

```text
negative_pct <= 5%
gt50_pct <= 0.05%
flags == NONE
```

The policy is intentionally conservative and is only the first experimental baseline. Flagged scenes can be re-enabled later with `--no-strict-flags` after targeted inspection.

Pixel-level handling:

```text
non-finite AGL → weight 0
negative AGL   → weight 0
AGL > 40 m     → weight 0.25
everything else → weight 1.0
```

Importantly, `AGL > 40 m` is NOT treated as invalid. The lower weight only reduces the influence of potentially suspicious extreme pixels in the first regression experiment.

## Expected local GAMUS layout

```text
data/gamus_local/
├── images/
│   └── train/
├── heights/
│   └── train/
└── classes/
    └── train/
```

For example:

```text
images/train/DC_40_31_RGB.h5
heights/train/DC_40_31_AGL.h5
classes/train/DC_40_31_CLS.h5
```

Each HDF5 file contains the dataset key:

```text
image
```

with:

```text
RGB  = 1024 × 1024 × 3
AGL  = 1024 × 1024
CLS  = 1024 × 1024
```

## Important

Do not start full training until the dataset smoke test passes and the manifest contains the intended scenes.

The next model-side step is to connect this Dataset to Depth Anything V2 Small and add a height-regression head.
