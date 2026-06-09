# Feature Points and Auto-Correlation

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-level overview]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#a. Setup]]
  - [[#b. Steps]]
- [[#A6. Variations]]

---

## A1. Purpose

Feature points (keypoints / interest points) are distinctive local image locations that can be reliably detected, described, and matched across images. They are the foundational building block for any task requiring cross-image correspondence: panorama stitching, Structure from Motion (SfM), SLAM, optical flow, and object retrieval. This file covers the *detection* stage — specifically the SSD auto-correlation measure $E(u,v)$ used to score candidate locations as flat, edge, or corner — and is the direct precursor to the [[harris-corner-detector]], which linearises $E$ for efficiency.

---

## A2. High-level overview

### Panorama stitching pipeline (canonical motivation)
1. **Phase 1 — Detect:** extract keypoints from each image.
2. **Phase 2 — Describe & Match:** compute descriptor vectors; match via $d(\mathbf{x}_1,\mathbf{x}_2) < T$.
3. **Phase 3 — Align:** warp/stitch images using the matched correspondences → [[image-warping-and-stitching]].

### Three components of local features
| Component | Task |
|---|---|
| **Detection** | Find a set of distinctive candidate pixel locations. |
| **Description** | Extract a compact descriptor vector $\mathbf{x}$ around each keypoint → [[feature-descriptors]]. |
| **Matching** | Compute $d(\mathbf{x}_1,\mathbf{x}_2)$; declare a correspondence if $d(\mathbf{x}_1,\mathbf{x}_2) < T$ → [[feature-matching]]. |

### Key definitions
| Term | Definition |
|---|---|
| Feature point / keypoint / interest point | A distinctive local image location used for matching, tracking, or reconstruction. |
| Local feature | A feature occupying a small image area; robust to clutter and occlusion. |
| Patch / window $w$ | Small rectangular region centred at a candidate point; used to compute $E$. |
| $(x_I, y_I)$ | Global image coordinates of the candidate pixel (x horizontal, y vertical). |
| $(x', y')$ | Local patch coordinates centred within window $w$. |
| $(u, v)$ | Shift vector applied to the window during auto-correlation. |
| $w(x', y')$ | Window / weighting function (box or Gaussian). |
| $E(u,v;\,x_I,y_I)$ | SSD auto-correlation error — measures how much patch appearance changes under shift $(u,v)$. |
| $\mathbf{x}_1,\mathbf{x}_2$ | Feature descriptor vectors (one per keypoint). |
| $d(\mathbf{x}_1,\mathbf{x}_2)$ | Distance between descriptor vectors. |
| $T$ | Matching threshold. |

### Five goals of a good feature
1. **Distinctiveness** — descriptor uniquely identifies the point across images (hard with repeated structures, e.g. building facades).
2. **Repeatability** — the same physical point is detected in both images despite geometric variation (translation, rotation, scale, perspective) and photometric variation (illumination, reflectance).
3. **Locality** — feature occupies small image area; robust to clutter and occlusion.
4. **Compactness & Efficiency** — small feature set, fast to compute.
5. **Independence** — detection runs per image without access to the "other" image (essential for object detection at match time).

### Corner criterion (core idea)
A location is a good keypoint if shifting the local patch by any $(u,v)$ produces a **large change** $E > 0$ in pixel appearance. Three canonical cases:

| Region type | $E(u,v)$ behaviour |
|---|---|
| **Flat** | $E \approx 0$ for all $(u,v)$ — no texture, no change. |
| **Edge** | $E \approx 0$ *along* the edge direction; $E > 0$ *perpendicular* to the edge. |
| **Corner** | $E > 0$ for **all** non-zero shifts — the desired case. |

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- Conceptually simple and directly interpretable: maximise $E$ to find corners.
- Window weighting (Gaussian) gives smoother, less noisy responses than a box.
- Flat/edge/corner trichotomy is clear and maps directly to the $E$ surface shape.

**Shortcomings / limitations**
- Naive computation of $E$ for all $(u,v)$ is **slow** — motivates the linearised [[harris-corner-detector]].
- Not invariant to large scale changes or rotations without additional handling.
- **Distinctiveness** is hard in structured environments (regular grids, repeated textures).
- **Independence** constraint: must detect without seeing the other image, making it harder to choose truly distinct points.

---

### Likely exam questions

**Q:** What are the three components of a local feature system? Describe each in one sentence.

**A:** (1) *Detection* — find distinctive candidate pixel locations in each image independently. (2) *Description* — extract a compact descriptor vector $\mathbf{x}$ around each keypoint. (3) *Matching* — declare a correspondence between keypoints in two images when $d(\mathbf{x}_1,\mathbf{x}_2) < T$.

---

**Q:** Write the SSD auto-correlation formula with a window weighting function. Define every symbol.

**A:**
$$E(u,v;\,x_I,y_I) = \sum_{x',\,y' \in w} w(x',y')\bigl[I(x_I+x'+u,\;y_I+y'+v) - I(x_I+x',\;y_I+y')\bigr]^2$$

$(x_I,y_I)$: candidate pixel; $(x',y')$: local patch coords; $(u,v)$: shift; $w(x',y')$: window weight (box or Gaussian); $I(\cdot)$: image intensity.

---

**Q:** For a flat region, an edge, and a corner, describe the shape of the $E(u,v)$ surface. Give concrete $(u,v)$ examples from the slides.

**A:**
- *Flat* (e.g., open sky): $E(0,0)=0$, $E(2,1)=0$, $E(-3,0)=0$ — zero everywhere.
- *Edge* (diagonal): $E(0,0)=0$, $E(-2,2)\approx 0$, $E(2,-2)\approx 0$ (along edge); $E(-2,0)=k_1>0$, $E(2,0)=k_2>0$ (perpendicular to edge) — ridge shape.
- *Corner* (triangle apex): $E(0,0)=0$; every other shift gives $E>0$ and $E > k_1, k_2$ — peaked in all directions.

---

**Q:** List the five goals of a good feature and name the two that are most directly in tension for structured scenes.

**A:** Distinctiveness, Repeatability, Locality, Compactness/Efficiency, Independence. *Distinctiveness* and *Repeatability* are most in tension in structured scenes with repeated elements (e.g., building window grids), where many patches look similar, making it hard to find points that are both repeatable and uniquely distinguishable.

---

**Q:** What are the two choices for the window function $w(x',y')$? Which is preferred and why?

**A:** (1) **Box** — uniform weight 1 inside the window, 0 outside. (2) **Gaussian** — smooth weighting that de-emphasises pixels far from the centre. Gaussian is preferred because it gives a smoother, less noisy $E$ surface and avoids hard boundary artefacts.

---

**Q:** Given three image locations — (1) open sky, (2) a wire/edge, (3) a textured corner — match each to its $E(u,v)$ surface shape (flat, ridge, peaked).

**A:** Location 1 (sky/flat) → surface (a) low/flat; location 2 (wire/edge) → surface (b) ridge shape; location 3 (textured corner) → surface (c) peaked in all directions (large $E$ for all non-zero $(u,v)$).

---

## A4. Directions of reasoning

### Forward / standard (image → keypoint decision)
**Given:** image $I$, candidate pixel $(x_I,y_I)$, window $w$, shift $(u,v)$.  
**Asked:** is $(x_I,y_I)$ a good keypoint?  
**Procedure:** compute $E(u,v;\,x_I,y_I)$ for a range of shifts; if $E$ is large for *all* non-zero $(u,v)$, classify as corner; if large only in one direction, edge; if near zero everywhere, flat.

### Reverse / inferential (surface shape → scene region type)
**Given:** the observed shape of the $E(u,v)$ surface at some location.  
**Asked:** what type of image structure is present?  
**Procedure:**
- $E \approx 0$ everywhere → flat (no texture; unreliable keypoint).
- $E$ forms a ridge (zero along one axis, high along the perpendicular) → edge (one dominant gradient direction).
- $E$ is high in all directions (no zero-axis) → corner (two independent gradient directions; reliable keypoint).

> [!note]
> The axis of the ridge in an edge region is *perpendicular* to the edge itself — a common exam trap. The $E$ surface is low *along* the edge (patches look similar when slid parallel to the edge) and high *across* it.

---

## A5. Standard implementation

### a. Setup
- **Input:** grayscale image $I$, window half-size $r$ (patch is $(2r+1)\times(2r+1)$), set of shift vectors $(u,v)$ to test, window function choice (box or Gaussian), threshold $T_E$.
- **Output:** set of pixel locations classified as corners (high $E$ for all shifts).
- **Notation:** global coords $(x_I,y_I)$; patch local coords $(x',y') \in [-r,r]\times[-r,r]$; shift $(u,v)$.

### b. Steps

1. **For each candidate pixel $(x_I,y_I)$** (typically every pixel, or a dense grid):

2. **For each test shift $(u,v)$**, compute the weighted SSD:
$$E(u,v;\,x_I,y_I) = \sum_{x'=-r}^{r}\;\sum_{y'=-r}^{r} w(x',y')\bigl[I(x_I+x'+u,\;y_I+y'+v) - I(x_I+x',\;y_I+y')\bigr]^2$$

3. **Classify** $(x_I,y_I)$ by the shape of the $E$ surface over all tested shifts:
   - $E \approx 0$ for all $(u,v)$ → **flat** (discard).
   - $E \approx 0$ only along one direction → **edge** (discard).
   - $E > 0$ for all non-zero $(u,v)$ → **corner** (candidate keypoint).

4. **Threshold** candidates: keep $(x_I,y_I)$ only if $\min_{(u,v)\neq(0,0)} E(u,v;\,x_I,y_I) > T_E$.

5. **Suppress non-maxima:** among nearby candidates, keep only the local maximum of the corner score → [[non-maximum-suppression]].

> [!warning]
> The naive approach of computing $E$ for every $(u,v)$ at every pixel is computationally expensive. The slides note this explicitly (p.66) as motivation for the Harris approximation, which avoids iterating over shifts by linearising $I$ with a Taylor expansion — see [[harris-corner-detector]].

---

## A6. Variations

### Unweighted (box) auto-correlation
**Change:** set $w(x',y') = 1$ for all $(x',y')$ in the window.
$$E(u,v;\,x_I,y_I) = \sum_{x',\,y' \in w} \bigl[I(x_I+x'+u,\;y_I+y'+v) - I(x_I+x',\;y_I+y')\bigr]^2$$
**When:** simple baseline; computationally cheaper weight function but produces harder patch boundary artefacts.

### Gaussian-weighted auto-correlation
**Change:** $w(x',y') = \exp\!\bigl(-\tfrac{x'^2+y'^2}{2\sigma^2}\bigr)$ (pixels near centre contribute more).  
**When:** preferred in practice; smoother, less sensitive to noise at patch boundaries; used as the default in [[harris-corner-detector]].

### Harris corner detector (linearised $E$)
**Change:** approximate $I(x_I+x'+u,\;y_I+y'+v)$ via a first-order Taylor expansion $I + u\,I_x + v\,I_y$, converting $E$ into a quadratic form in $(u,v)$ driven by the structure tensor $\mathbf{M}$. Eliminates the need to iterate over shifts entirely.  
**When:** standard efficient alternative to naive SSD. → [[harris-corner-detector]]

### Relationship to feature description and matching
Once corner candidates are selected, the pipeline continues with:
- [[feature-descriptors]] — extract a descriptor vector $\mathbf{x}$ at each keypoint.
- [[feature-matching]] — match descriptors via $d(\mathbf{x}_1,\mathbf{x}_2) < T$.
- [[non-maximum-suppression]] — thin dense corner maps to isolated keypoints.
- [[image-warping-and-stitching]] — use matched keypoints to estimate and apply a geometric transformation.
