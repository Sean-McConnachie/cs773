# Stereo Rectification

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

Given a stereo pair calibrated in an arbitrary (non-canonical) configuration, stereo rectification warps both images so that **every pair of corresponding epipolar lines becomes the same horizontal scanline** — reducing correspondence search from 2D to 1D. Physical mounting of cameras in perfect canonical alignment is impractical (sensitivity, timing, non-identical hardware), so rectification performs the alignment computationally via a pair of projective homographies $\mathbf{H}_1$ and $\mathbf{H}_2$. It is step 6 (or step 3 in a 5-step pipeline) of the binocular stereo pipeline, applied before [[stereo-block-matching]].

---

## A2. High-level overview

**Canonical epipolar geometry** — the target configuration that rectification achieves:
1. Baseline $b$ aligned with the image $x$-axis.
2. Both optical axes parallel, pointing along $z$.
3. Both cameras share the same intrinsic matrix.
4. Square pixels: $f_x = f_y = f$.

**Key definitions:**

| Term | Definition |
|---|---|
| $\mathbf{P}^i_{old} = \mathbf{K}_i \mathbf{M}_i$ | Original $3\times4$ projection matrix for camera $i$ |
| $\mathbf{K}_i$ | $3\times3$ intrinsic matrix (focal length, principal point) |
| $\mathbf{M}_i = [\mathbf{R}_i \mid \mathbf{t}_i]$ | $3\times4$ extrinsic matrix (rotation + translation) |
| $\mathbf{c}_i = -\mathbf{R}_i^\top \mathbf{t}_i$ | Optical centre of camera $i$ in WRF (world coords, mm) |
| $\mathbf{K}_{new}$ | Averaged intrinsic matrix for the canonical pair |
| $\mathbf{R}_{new}$ | New shared rotation whose rows are the canonical axes |
| $\mathbf{v}_x, \mathbf{v}_y, \mathbf{v}_z$ | Unit vectors defining the canonical coordinate frame |
| $\mathbf{H}_i$ | $3\times3$ rectifying homography for image $i$ |
| $\mathbf{P}^i_{new:3}$ | First $3\times3$ sub-matrix (columns 1–3) of $\mathbf{P}^i_{new}$ |

**Algorithm steps (no math yet):**
1. Form old projection models from calibration output.
2. Average intrinsics to get a common $\mathbf{K}_{new}$.
3. Recover optical centres $\mathbf{c}_1, \mathbf{c}_2$.
4. Build canonical axes $\mathbf{v}_x, \mathbf{v}_y, \mathbf{v}_z$ and assemble $\mathbf{R}_{new}$.
5. Form new extrinsic matrices — rotation updated, **translations kept original**.
6. Form new projection matrices.
7. Extract rectifying homographies as ratios of $3\times3$ sub-matrices.
8. Apply $\mathbf{H}_1, \mathbf{H}_2$ by **inverse warping** to each image.

---

## A3. Strengths, shortcomings & limitations

**Strengths:**
- Reduces stereo matching search from 2D to 1D — dramatically lowers compute.
- Compact implementation (~20 lines of Matlab/Python — the Fusiello algorithm).
- Works with any calibrated stereo rig; no physical re-mounting required.
- Starting near canonical geometry (well-mounted rig) minimises image distortion post-warp.

**Shortcomings / limitations:**
- Requires accurate prior calibration ($\mathbf{K}_i, \mathbf{R}_i, \mathbf{t}_i$) — errors propagate directly into the rectified images.
- Image warping introduces resampling artefacts (interpolation error).
- Badly misaligned rigs produce severely distorted rectified images (large warp = poor quality).
- Infinitely many valid rectifications exist (choice of $\mathbf{R}_{new}$ is not unique); different choices trade off distortion differently.
- Homography is only exact for planar scenes or when all points are at infinity; depth causes small residual vertical disparity.

---

### Likely exam questions

**Q:** What is the purpose of stereo rectification and what does the output look like?
**A:** Rectification warps a stereo pair so that epipolar lines are horizontal and coincident across both images — corresponding points in the left and right rectified images lie on the **same row**. This reduces disparity search to a 1D horizontal scan.

---

**Q:** State the four attributes of canonical epipolar geometry.
**A:**
1. Baseline aligned with the $x$-axis.
2. Both optical axes parallel along $z$.
3. Both cameras have identical intrinsics.
4. Square pixels: $f_x = f_y = f$.

---

**Q:** Why are the translation vectors $\mathbf{t}_1, \mathbf{t}_2$ left unchanged in the new extrinsic matrices $\mathbf{M}^i_{new}$?
**A:** The translations define where each camera's optical centre is in space. Changing them would alter the baseline $b = \|\mathbf{c}_1 - \mathbf{c}_2\|$, which invalidates the depth/disparity relationship $Z = fb/d$. Only rotation needs to change; **translation affects geometry, rotation preserves it.**

---

**Q:** Write the formula for the rectifying homography $\mathbf{H}_i$ in both notations seen in the course.
**A:** Using projection sub-matrices (Fusiello / W8L form):
$$\mathbf{H}_i = \mathbf{P}^i_{new:3}\left(\mathbf{P}^i_{old:3}\right)^{-1}$$
Equivalent factored form (W11T):
$$\mathbf{H}_i = \mathbf{K}_n \mathbf{R}_n \mathbf{R}_i^{-1} \mathbf{K}_i^{-1}$$

---

**Q:** Using the numeric example from lectures, give $\mathbf{v}_x$.

> [!example] Numeric rectification example (W8T, pp. 40–44)
> Cameras calibrated on a checkerboard; Tsai parameters yielded:
> - $\mathbf{c}_1 \approx [72.28,\ 76.82,\ 42.94]^\top$, $\mathbf{c}_2 \approx [77.95,\ 71.61,\ 43.41]^\top$
> - Baseline: $\mathbf{b} = \mathbf{c}_2 - \mathbf{c}_1 \approx [5.67,\ {-5.21},\ 0.47]^\top$
> - $\mathbf{v}_x \approx [0.7349,\ {-0.6754},\ 0.0612]$
> - $\mathbf{v}_y \approx [{-0.2654},\ {-0.2034},\ 0.9424]$
> - $\mathbf{v}_z \approx [{-0.6241},\ {-0.7088},\ {-0.3287}]$
> - $\mathbf{H}_1, \mathbf{H}_2$ computed as $3\times3$ matrices and applied.

---

**Q:** Why do infinitely many rectifications exist? How does the Fusiello algorithm choose one?
**A:** Any rotation that maps the baseline to the $x$-axis and makes both optical axes parallel is valid — the remaining degree of freedom (rotation about the baseline) is unconstrained. Fusiello fixes it by anchoring $\mathbf{v}_y$ using camera 1's old $z$-axis (third row of $\mathbf{R}_1$), and minimising distortion by averaging intrinsics.

---

## A4. Directions of reasoning

### Forward (given calibration → produce rectified images)

| Given | Asked | Key step |
|---|---|---|
| $\mathbf{K}_i, \mathbf{R}_i, \mathbf{t}_i$ for $i \in \{1,2\}$ | Rectified image pair | Run Fusiello 7-step algorithm to obtain $\mathbf{H}_1, \mathbf{H}_2$; apply by inverse warping |

### Reverse / inferential

**Why are same-row correspondences guaranteed after rectification?**
After applying $\mathbf{H}_i$, both projection matrices share the same $\mathbf{K}_{new}$ and $\mathbf{R}_{new}$; the only difference is the translation column. For any 3D point $\mathbf{X}$, the projection onto both rectified images produces the same $v$-coordinate (row), because the shared rotation aligns both image planes with the baseline as their $x$-axis and forces epipolar lines to be horizontal rows.

**What does each homography do geometrically?**
$\mathbf{H}_i = \mathbf{P}^i_{new:3}(\mathbf{P}^i_{old:3})^{-1}$ is the map from old image-plane coordinates to new image-plane coordinates induced by the change of camera orientation from $\mathbf{R}_i$ to $\mathbf{R}_{new}$ and from $\mathbf{K}_i$ to $\mathbf{K}_{new}$. The translation columns cancel because they are unchanged, which is why only the $3\times3$ sub-matrices appear (the fourth column contributes zero difference in the ratio).

**Recovering why translation cannot change:**
If $\mathbf{t}_i$ were altered, $\mathbf{c}_i = -\mathbf{R}_{new}^\top \mathbf{t}_i^{new}$ would differ from the physical optical centre, making the triangulation baseline $\|\mathbf{c}_1 - \mathbf{c}_2\|$ inconsistent with the actual rig geometry and corrupting all depth estimates.

---

## A5. Standard implementation

### a. Setup

**Algorithm:** Fusiello, Trucco & Verri — *A compact algorithm for rectification of stereo pairs.*

**Inputs:**
- Calibrated intrinsics $\mathbf{K}_1, \mathbf{K}_2$ (from [[tsai-calibration]] or DLT).
- Calibrated extrinsics $\mathbf{R}_1, \mathbf{t}_1, \mathbf{R}_2, \mathbf{t}_2$.
- Left image $I_L$, right image $I_R$.

**Outputs:**
- Rectifying homographies $\mathbf{H}_1, \mathbf{H}_2$ ($3\times3$).
- Rectified image pair (corresponding points on same row).

**Notation (this file):** scalars italic ($f$, $d$, $Z$); vectors bold lowercase ($\mathbf{c}_i$, $\mathbf{v}_x$, $\mathbf{t}_i$); matrices bold uppercase ($\mathbf{K}$, $\mathbf{R}$, $\mathbf{H}$, $\mathbf{P}$, $\mathbf{M}$); WRF = world reference frame (mm).

> [!note] Sign convention for $\mathbf{v}_x$
> W8L uses $\mathbf{v}_x = \widehat{\mathbf{c}_1 - \mathbf{c}_2}$ (left minus right); W8T and W11T use $\mathbf{v}_x = \widehat{\mathbf{c}_2 - \mathbf{c}_1}$ (right minus left — pointing left-to-right along the baseline). Both are valid rectifications; the sign choice only flips whether $u$ increases toward the right camera. The numeric example values in the summaries ($\mathbf{v}_x \approx [0.7349, -0.6754, 0.0612]$) correspond to the $\mathbf{c}_2 - \mathbf{c}_1$ convention. W11T uses $r_1 = (c_1 - c_2)/\|c_1-c_2\|$ which is the opposite sign — reconciled by noting that $\mathbf{v}_y$ and $\mathbf{v}_z$ are constructed consistently, so the final rectification is equivalent up to a reflection.

### b. Steps

**Step 1 — Old projection models**

$$\mathbf{P}^1_{old} = \mathbf{K}_1 \mathbf{M}_1, \qquad \mathbf{P}^2_{old} = \mathbf{K}_2 \mathbf{M}_2$$

where $\mathbf{M}_i = [\mathbf{R}_i \mid \mathbf{t}_i]$.

---

**Step 2 — Averaged intrinsics**

$$\mathbf{K}_{new} = \frac{\mathbf{K}_1 + \mathbf{K}_2}{2}$$

Both rectified cameras use $\mathbf{K}_{new}$; this satisfies the canonical requirement of identical intrinsics.

---

**Step 3 — Optical centres in WRF**

$$\mathbf{c}_i = -\mathbf{R}_i^\top \mathbf{t}_i$$

Derivation note: $\mathbf{c}_i = \mathbf{M}_i^{-1}(0,0,0,1)^\top$; expanding gives $-\mathbf{R}_i^\top \mathbf{t}_i$.

Baseline vector:

$$\mathbf{b} = \mathbf{c}_2 - \mathbf{c}_1 \quad \text{(or }\mathbf{c}_1 - \mathbf{c}_2\text{, see sign note above)}$$

---

**Step 4 — Canonical coordinate axes**

New $x$-axis = normalised baseline:

$$\mathbf{v}_x = \frac{\mathbf{b}}{\|\mathbf{b}\|}$$

Approximate old $z$ direction (third row of $\mathbf{R}_1$, treated as a column vector):

$$\mathbf{k} = \mathbf{R}_{1,\text{row3}}^\top$$

New $y$-axis (orthogonal to $\mathbf{v}_x$, consistent with camera 1's forward direction):

$$\mathbf{v}_y = \frac{\mathbf{k} \times \mathbf{v}_x}{\|\mathbf{k} \times \mathbf{v}_x\|}$$

New $z$-axis (completes right-hand frame):

$$\mathbf{v}_z = \mathbf{v}_x \times \mathbf{v}_y$$

New rotation matrix (rows are the canonical axes):

$$\mathbf{R}_{new} = \begin{pmatrix} \mathbf{v}_x^\top \\ \mathbf{v}_y^\top \\ \mathbf{v}_z^\top \end{pmatrix}$$

---

**Step 5 — New extrinsic matrices (translation UNCHANGED)**

$$\mathbf{M}^1_{new} = [\mathbf{R}_{new} \mid \mathbf{t}_1], \qquad \mathbf{M}^2_{new} = [\mathbf{R}_{new} \mid \mathbf{t}_2]$$

> [!warning] Why translations are not changed
> Altering $\mathbf{t}_i$ would move the optical centres, changing the physical baseline and invalidating $Z = f b / d$. "Translation affects geometry, rotation preserves it."

---

**Step 6 — New projection matrices**

$$\mathbf{P}^1_{new} = \mathbf{K}_{new} \mathbf{M}^1_{new}, \qquad \mathbf{P}^2_{new} = \mathbf{K}_{new} \mathbf{M}^2_{new}$$

---

**Step 7 — Rectifying homographies**

Let $\mathbf{P}^i_{new:3}$ and $\mathbf{P}^i_{old:3}$ denote the **first three columns** ($3\times3$ sub-matrices) of the respective projection matrices:

$$\mathbf{H}_1 = \mathbf{P}^1_{new:3}\left(\mathbf{P}^1_{old:3}\right)^{-1}, \qquad \mathbf{H}_2 = \mathbf{P}^2_{new:3}\left(\mathbf{P}^2_{old:3}\right)^{-1}$$

The fourth column (translation) does not appear because $\mathbf{t}_i$ is identical in old and new, so it cancels in the ratio. Equivalent factored form (W11T):

$$\mathbf{H}_i = \mathbf{K}_{new} \mathbf{R}_{new} \mathbf{R}_i^{-1} \mathbf{K}_i^{-1}$$

---

**Step 8 — Apply by inverse warping**

For each output pixel $\tilde{\mathbf{u}}'$ in the rectified image: map back via $\mathbf{H}_i^{-1}$ to the source image coordinate, then bilinear-interpolate the intensity. Forward warping is not used (leaves holes). See [[image-warping-and-stitching]].

---

## A6. Variations

### Infinitely many valid rectifications
Any orthonormal rotation mapping the baseline to $\hat{x}$ and making optical axes parallel is a valid $\mathbf{R}_{new}$; the residual degree of freedom (rotation about the baseline) is unconstrained. Different choices produce different amounts of distortion. The Fusiello algorithm fixes this by anchoring the $y$-axis to camera 1's old $z$-direction.

### W11T / practical-session form of $\mathbf{H}_i$
Instead of computing projection matrices explicitly, derive:

$$\mathbf{H}_i = \mathbf{K}_n \mathbf{R}_n \mathbf{R}_i^{-1} \mathbf{K}_i^{-1}$$

Reading right-to-left: (1) undo old calibration ($\mathbf{K}_i^{-1}$), (2) undo old rotation ($\mathbf{R}_i^{-1}$), (3) apply new rotation ($\mathbf{R}_n$), (4) apply new calibration ($\mathbf{K}_n$). Algebraically identical to Step 7 above (translation column cancels in both).

### Validation — vertical residual check
After applying $\mathbf{H}_1, \mathbf{H}_2$ to known correspondences, compute:

$$e_v^{(i)} = \left|v_L^{\prime(i)} - v_R^{\prime(i)}\right|$$

Mean and std of $e_v$ should be near zero. A large residual indicates miscalibration or implementation error.

---

**Related topics:** [[epipolar-geometry]] — epipolar constraint and epipolar lines; [[homography-and-dlt]] — projective homographies and their computation; [[image-warping-and-stitching]] — inverse warping and bilinear interpolation; [[tsai-calibration]] — source of $\mathbf{K}_i, \mathbf{R}_i, \mathbf{t}_i$; [[stereo-block-matching]] — the downstream step that exploits rectification.
