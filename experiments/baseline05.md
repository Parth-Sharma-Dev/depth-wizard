# Baseline 0.5 — Multi-Scene Zero-Shot Remote-Sensing Evaluation

## 1. Objective

Evaluate how well a pretrained monocular depth estimation model generalizes across different remote-sensing / aerial scene types **without any fine-tuning, DEM information, metric calibration, or Ground Control Points (GCPs)**.

Baseline 0.5 extends Baseline 0 from a single image to a small multi-scene collection. Its purpose is to understand the strengths and failure modes of the pretrained model before modifying the model or training pipeline.

---

## 2. Model

**Model:** Depth Anything V2 Large

**Hugging Face checkpoint:**
`depth-anything/Depth-Anything-V2-Large-hf`

**Model type:** Monocular depth estimation

**Training / fine-tuning:** None

**Inference mode:** Zero-shot

---

## 3. Hardware

**CPU:** Intel Core i7-13650HX

**GPU:** NVIDIA GeForce RTX 5060 Laptop GPU

**VRAM:** 8 GB

**RAM:** 16 GB

**Inference device:** CUDA

---

## 4. Input Dataset

A small collection of **7 RGB remote-sensing / aerial images** was used to test different scene characteristics.

| # | Image | Scene type | Resolution |
|---|---|---|---|
| 1 | `forest.png` | Forest / vegetation | 878 × 697 |
| 2 | `industrial.png` | Industrial / built-up | 953 × 647 |
| 3 | `mountain_tunnel.png` | Mountain / complex terrain | 977 × 650 |
| 4 | `mountains.png` | Mountain / terrain | 750 × 721 |
| 5 | `satellite.png` | Urban satellite scene | 648 × 572 |
| 6 | `snowy_mountains.png` | Snowy mountainous terrain | 753 × 592 |
| 7 | `urban.png` | Urban built-up area | 751 × 707 |

**Dataset size:** 7 images

**Georeferencing:** Not used in this experiment

**Metric elevation reference:** None

**Ground-truth DSM:** Not used

---

## 5. Experimental Pipeline

```text
RGB remote-sensing image
        ↓
Depth Anything V2 Small
        ↓
Relative depth prediction
        ↓
Resize to original image dimensions
        ↓
Save depth prediction
        ↓
Visual inspection
```

No remote-sensing fine-tuning or metric-scale calibration was applied.

---

## 6. Results

The model successfully processed all 7 images using the RTX 5060 GPU.

| Image | Min | Max | Mean | Std | Inference time (s) |
|---|---:|---:|---:|---:|---:|
| `forest.png` | 33.6250 | 208.1250 | 117.5439 | 38.2908 | 0.8159 |
| `industrial.png` | 13.3516 | 369.2500 | 100.7995 | 31.9736 | 0.1053 |
| `mountain_tunnel.png` | 25.9375 | 282.2500 | 164.5731 | 42.9708 | 0.1097 |
| `mountains.png` | 24.7812 | 243.6250 | 132.2844 | 51.2387 | 0.0798 |
| `satellite.png` | 28.7812 | 254.2500 | 116.3742 | 44.8313 | 0.0807 |
| `snowy_mountains.png` | 10.1562 | 285.0000 | 192.3089 | 63.5869 | 0.0908 |
| `urban.png` | 19.8906 | 233.6250 | 122.0914 | 41.9468 | 0.0700 |

### Summary of runtime

**Mean inference time across 7 images:** approximately **0.193 s/image**.

The first image took substantially longer than the remaining images, which is consistent with model/GPU warm-up overhead. The steady-state inference times for the remaining scenes were approximately 0.07–0.11 seconds per image.

---

## 7. Qualitative Observations

### 7.1 Urban scenes

The `urban.png` and `satellite.png` predictions show useful structural information.

Observed behavior:

- Large buildings are represented as distinct depth structures.
- Several smaller structures are also differentiated.
- Building footprints and major spatial patterns remain visible in the depth prediction.
- The model therefore extracts useful geometric information even without remote-sensing-specific training.

**Interpretation:** The pretrained model is a promising foundation for DepthWizard and should be adapted rather than replaced immediately.

### 7.2 Industrial scene

The `industrial.png` scene shows strong responses around prominent industrial structures, but the interpretation of surrounding surfaces is less consistent.

Observed behavior:

- Large structures are detected clearly.
- Some surfaces with distinctive visual appearance receive strong depth responses.
- The prediction appears to rely partly on learned object/appearance priors rather than purely on physical elevation.

**Interpretation:** This indicates a domain gap between the model's original training distribution and overhead remote-sensing imagery.

### 7.3 Mountain and complex terrain scenes

The mountain-related scenes expose a major limitation of the zero-shot model.

Observed behavior:

- Broad terrain variation is captured.
- Predictions tend toward smooth large-scale gradients in complex terrain.
- Fine local terrain geometry is not always preserved reliably.
- This suggests that the model recognizes scene-level geometry but does not yet provide sufficiently accurate surface reconstruction for DSM generation.

**Interpretation:** Complex terrain is a key target for remote-sensing adaptation and later metric calibration.

### 7.4 Forest / vegetation scene

The `forest.png` scene demonstrates another difficult case.

Vegetation has inherently ambiguous surface structure from a single optical observation. The resulting prediction should therefore be treated cautiously for accurate surface elevation recovery.

**Interpretation:** Vegetation is likely to remain one of the harder categories and may benefit from semantic/contextual information in later experiments.

---

## 8. Important Interpretation of the Numbers

The numerical depth ranges **must not be interpreted as metres**.

For example, the fact that `snowy_mountains.png` has a higher maximum value than `urban.png` does **not** mean the snowy scene contains physically higher elevations.

The predictions are relative / model-space depth values and are not directly comparable in absolute scale between scenes.

Additionally, the visualization of each scene is normalized independently. Therefore, color values in one image cannot be directly compared with color values in another image as physical height.

---

## 9. Strengths Identified

- Successful zero-shot inference on all 7 test scenes.
- Very fast steady-state inference on the RTX 5060 Laptop GPU.
- Useful structural information is visible in urban scenes.
- Large structures are generally represented distinctly.
- The pretrained model provides a practical starting point for the remote-sensing adaptation stage.

---

## 10. Limitations Identified

- Output is relative depth rather than metric elevation.
- Generalization quality varies significantly by scene type.
- Complex terrain is represented too coarsely for reliable high-resolution DSM generation.
- Vegetation introduces substantial ambiguity.
- Visual depth intensity cannot be directly interpreted as physical height.
- No quantitative DSM accuracy can yet be claimed because no reference elevation data was used.

---

## 11. Baseline 0.5 Conclusion

Baseline 0.5 demonstrates that Depth Anything V2 Small can extract meaningful coarse structural information from overhead RGB imagery without any remote-sensing-specific training. Urban structures are represented reasonably well, while vegetation and complex mountainous terrain reveal significant limitations. The scene-dependent behavior confirms the presence of a domain gap and demonstrates the need for remote-sensing adaptation. The results support continuing with the pretrained backbone while improving its representation through remote-sensing data and subsequently introducing metric-scale calibration.

Baseline 0.5 therefore establishes a qualitative zero-shot reference point for the next stage of DepthWizard.

---

## 12. Next Experiment

**Baseline 1 — Remote-Sensing Adaptation**

### Goal

Fine-tune/adapt the monocular depth foundation model using remote-sensing imagery and height/elevation supervision, then compare the adapted model against Baseline 0.5 on held-out scenes.

### Main questions

1. Does remote-sensing fine-tuning improve structural consistency in urban scenes?
2. Does fine-tuning improve complex terrain representation?
3. Does it improve vegetation/land-cover robustness?
4. Which target representation is most useful for DepthWizard: relative depth, height/nDSM, or a multi-task combination?

---

## 13. Reproducibility Information

**Script:** `src/depth/baseline05.py`

**Input directory:** `data/input/`

**Output directory:** `data/output/baseline05/`

**Summary CSV:** `data/output/baseline05/summary.csv`

**Model:** `depth-anything/Depth-Anything-V2-Large-hf`

**Inference device:** CUDA / RTX 5060 Laptop GPU

**Training:** None

**External elevation reference:** None

