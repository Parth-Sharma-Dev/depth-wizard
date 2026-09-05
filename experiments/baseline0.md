# Baseline 0 — Pretrained Monocular Depth Estimation

## 1. Objective

Evaluate how well a pretrained monocular depth estimation model can infer
relative scene geometry from a single RGB remote-sensing image without
fine-tuning, geospatial calibration, DEM information, or Ground Control Points.

---

## 2. Model

**Model:** Depth Anything V2 Large

**Checkpoint:**
`depth-anything/Depth-Anything-V2-Large-hf`

**Training/Fine-tuning:** None

---

## 3. Hardware

**CPU:** Intel Core i7-13650HX

**GPU:** NVIDIA GeForce RTX 5060 Laptop GPU

**VRAM:** 8 GB

**RAM:** 16 GB

---

## 4. Input

**Image resolution:** 648 × 572

**Image type:** RGB

**Scene:** Urban remote-sensing / aerial imagery

---

## 5. Inference

**Device:** CUDA

**GPU:** NVIDIA GeForce RTX 5060 Laptop GPU

**Model successfully loaded:** Yes

**Inference completed:** Yes

---

## 6. Output

**Raw depth:** `baseline0_depth.npy`

**Visualization:** `baseline0_depth.png`

---

## 7. Initial Observations

### Positive

- The model successfully produces a structured depth map from the single
  overhead RGB image.
- Large buildings are represented as distinct depth structures.
- Several smaller structures are also identifiable.
- The prediction retains meaningful spatial correspondence with the RGB image.
- The pretrained model therefore provides potentially useful geometric
  information even without remote-sensing fine-tuning.

### Limitations

- The predicted values are relative depth and cannot be interpreted as
  physical elevation in metres.
- Some visual structures may receive strong depth responses that do not
  necessarily correspond to their true physical height.
- The model has not yet been adapted specifically to overhead remote-sensing
  imagery.
- Vegetation, shadows, roofs and terrain may introduce ambiguity.

---

## 8. Numerical Issue

The first run reported:

`Std = inf`

despite the predicted range being approximately 28.78–254.25.

This is believed to be a numerical reduction/dtype issue caused by low-precision
floating-point statistics rather than an invalid prediction.

The depth tensor will be converted to float32 before calculating statistics
in the next run.

---

## 9. Conclusion

Baseline 0 successfully demonstrates that a pretrained monocular depth model
can extract meaningful structural information from an overhead RGB scene.
However, the output remains scale-ambiguous and is not yet suitable for
metric DSM generation. The next stage should investigate remote-sensing
adaptation and determine whether fine-tuning improves the structural quality
and quantitative height estimation.

---

## 10. Next Experiment

**Baseline 0.5 — Systematic Evaluation of the pretrained model**