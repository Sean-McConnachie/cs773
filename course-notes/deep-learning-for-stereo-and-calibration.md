# Deep Learning for Stereo and Calibration

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-Level Overview]]
  - [[#Key Definitions]]
- [[#A3. Strengths, Shortcomings & Limitations]]
  - [[#Likely Exam Questions]]
- [[#A4. Directions of Reasoning]]
- [[#A5. Standard Implementation — Concepts & Relationships]]
- [[#A6. Variations]]
  - [[#A6.1 CREStereo (Cascaded REcurrent Stereo, CVPR 2022)]]
  - [[#A6.2 DeepCalib (CVMP 2018)]]
  - [[#A6.3 Parallel RCNN — RGB-D People Detection]]

---

## A1. Purpose

Deep learning replaces or augments hand-engineered components of the stereo and calibration pipeline — learned feature similarity, cost-volume regularisation, and implicit scene priors replace hand-crafted SAD/SSD costs, smoothness terms, and manual target-based calibration procedures. The goal is better performance on textureless regions, occlusions, imperfect rectification, and wide-FOV distortion, where classical methods fail. DL stereo and calibration methods still operate within the same geometric framework ($Z = fb/d$, pinhole projection) — they improve matching quality and parameter estimation, but cannot overcome the fundamental limits imposed by the geometry.

> [!warning]
> The perceptron/neural-network primer slides in W8T (slide 49) are explicitly marked **"Not examinable"** — NN architecture basics are assumed background, not assessed content.

---

## A2. High-Level Overview

### Classical vs. Deep Stereo — Common Pipeline

Both classical and deep stereo follow the same abstract structure:

1. Extract features (classical: raw pixels; deep: CNN features)
2. Build a cost volume (classical: SSD/SAD/NCC scores; deep: learned correlation)
3. Regularise / select disparity (classical: local argmin or global smoothness; deep: recurrent refinement or 3D conv)
4. Output disparity map

The deep approach substitutes steps 1–3 with learned counterparts; the "build cost volume → select or regress disparity" idea is shared.

### Key Definitions

| Term | Definition |
|---|---|
| **Disparity** $d$ | Horizontal pixel offset between corresponding rectified points: $d = x_L - x_R$ |
| **Cost volume** | 3-D array (H × W × D) of matching costs at every pixel for every candidate disparity $d$ |
| **Non-ideal rectification** | Real stereo pairs where corresponding points deviate slightly from the same scanline due to imperfect calibration |
| **RGB-D** | Paired colour + depth image; depth invariant to illumination, robust to colour camouflage |
| **Colour camouflage** | Foreground matches background colour — depth helps disambiguate |
| **Depth camouflage** | Foreground at same depth as background — depth fails to disambiguate |
| **2.5D fusion** | Treat depth as a 2-D image; apply 2-D convolutions |
| **3D fusion** | Volumetric / point-cloud representation; apply 3-D convolutions |
| **Unified spherical model** | Single-parameter ($\xi$) distortion model covering pinhole to catadioptric cameras; fully reversible |
| **AGCL** | Adaptive Group Correlation Layer — CREStereo's local correlation module with deformable search |
| **RUM** | Recurrent Update Module — GRU-based iterative disparity refinement in CREStereo |
| **L2 channel-wise normalisation** | Per-channel L2 norm normalisation of feature maps before fusion; essential to prevent one modality dominating |

---

## A3. Strengths, Shortcomings & Limitations

### Strengths

- **Textureless regions:** learned priors and context aggregation (e.g. attention) fill in where block matching fails
- **Occlusions:** network learns to handle partially visible regions; deformable windows adapt
- **Imperfect rectification:** AGCL's 2D-1D search handles residual vertical offsets classical methods assume away
- **Wide-FOV / large distortion:** DeepCalib's unified spherical model handles fisheye and catadioptric cameras that break the Brown-Conrady model
- **Speed at inference:** a single forward pass vs. iterative optimisation (e.g. SGM); CREStereo runs on GPU in near-real-time
- **Single-image calibration:** DeepCalib estimates $f$ and $\xi$ from one in-the-wild image in ~50 ms; classical toolboxes need ~30 checkerboard images
- **Modality fusion:** RGB-D fusion (Parallel RCNN) improves detection in occluded / low-light scenes where RGB alone fails

### Shortcomings & Limitations

- **Cannot override geometry:** DL cannot break $Z = fb/d$ — at large depth, even perfect matching gives large $\Delta Z$ because disparity is quantised; $\Delta Z \approx Z^2 / (fb + Z)$
- **Needs training data:** CREStereo uses millions of synthetic frames; DeepCalib uses 67,000 panoramas; Parallel RCNN requires RGB-D annotation
- **Domain mismatch:** RGB ImageNet pre-trained features do not transfer cleanly to depth images (different statistical distribution)
- **Interpretability:** learned cost volumes and GRU states are opaque; failure modes are harder to diagnose
- **L2 normalisation required for fusion:** naive concatenation of RGB and depth features without normalisation hurts mAP (89.6% vs. 90.0% baseline — worse than RGB alone)
- **DeepCalib failure modes:** strong motion blur, over-exposure, rolling shutter, ambiguity between $(f, \xi)$ pairs producing the same image

### Likely Exam Questions

**Q:** How does deep stereo differ from classical stereo at each stage of the pipeline?

**A:** Classical stereo uses hand-designed costs (SSD/SAD/NCC), local or semi-global optimisation (block matching, SGM), and explicit smoothness assumptions (Potts model). Deep stereo uses learned feature similarity (CNN features + attention), learned cost-volume regularisation (recurrent GRU updates or 3-D convolutions), and implicit priors from training data. Both still build a cost volume and select/regress disparity — the inputs and operators are replaced, not the structure.

---

**Q:** Why is L2 channel-wise normalisation essential in RGB-D feature fusion? What happens without it?

**A:** RGB and depth CNNs have different feature magnitude distributions. Without normalisation, the modality with larger values dominates, causing the fused network to ignore the other stream. In Parallel RCNN: RGB-only Faster RCNN achieves 90.0% mAP; fusion **without** normalisation drops to 89.6% (worse); fusion **with** normalisation reaches 91.5%. The learnable scaling parameter $s_i$ (Eq. 3: $F_i = s_i f_i'$) compensates for reduced magnitude after normalisation so learning is not slowed.

---

**Q:** What problem does AGCL solve, and what are its three components?

**A:** AGCL (Adaptive Group Correlation Layer) solves the problem that real-world cameras produce imperfectly rectified stereo pairs — corresponding points may not lie on the same scanline. Components: (1) **Local feature attention** (self- + cross-attention, inspired by LoFTR) aggregates global context before correlation; (2) **2D-1D alternate local search** — alternates between 1D (horizontal-only, $g(d)=0$) and 2D ($k \times k$ grid with dilation $l$) to handle residual vertical offset; (3) **Deformable search window** — learned per-pixel offsets $(dx, dy)$ shift sampling positions adaptively, handling occlusions and textureless areas. Features are also split into $\mathcal{G}$ groups for group-wise correlation.

---

**Q:** Why does DeepCalib use the unified spherical model instead of the Brown-Conrady model?

**A:** Brown-Conrady is not suitable for large distortions (fisheye/catadioptric), and its polynomial back-projection is numerically difficult (hardly reversible). The unified spherical model uses a single scalar $\xi \in [0, 1]$ (slightly above 1 for some catadioptric cameras), has a closed-form back-projection (Eq. 2), is **fully reversible**, and covers the entire range from pinhole ($\xi=0$) to catadioptric cameras with one formula. This makes it ideal for a CNN classification head over a discretised $\xi$ grid.

---

**Q:** Why does classification outperform regression for DeepCalib, and which architecture is best?

**A:** Discretising $f$ and $\xi$ into class bins and using softmax + cross-entropy loss yields higher accuracy than predicting continuous values with sigmoid + logcosh loss. SingleNet-Classification (one Inception-V3 network with two output heads, one for $f$, one for $\xi$) is the best architecture — highest accuracy and ~2× faster than DualNet or SeqNet.

---

**Q:** Can deep learning overcome the geometric depth-accuracy limit $\Delta Z \approx Z^2/(fb)$?

**A:** No. DL can improve **matching accuracy** (reduce mismatches) but cannot change the geometric fact that disparity is a discrete integer. A 1-pixel disparity error always produces a depth error of $\Delta Z = fb / [d(d+1)] \approx Z^2 / (fb + Z)$. At large $Z$ this grows quadratically regardless of how good the matching network is. (E.g., at $Z = 6000$ mm with $f_{px} = 2000$, $b = 120$ mm: $\Delta Z = 146$ mm per pixel.) Only hardware changes (larger baseline, smaller pixels) improve this limit.

---

## A4. Directions of Reasoning

### Forward / Standard

| Stage | Given | Derived |
|---|---|---|
| Feature extraction | Left/right images $I_1, I_2$ | Feature maps $\mathbf{F}_1, \mathbf{F}_2$ |
| Cost volume | $\mathbf{F}_1, \mathbf{F}_2$, disparity candidates | Local correlation volume $H \times W \times D$ |
| Disparity estimation | Cost volume + RUM iterations | Refined disparity map $d(u,v)$ |
| Depth recovery | $d$, $f$, $b$ | $Z = fb/d$ |
| RGB-D fusion | RGB features + normalised depth features | Fused detection (Parallel RCNN) |
| DeepCalib forward | Single wide-FOV image | Predicted $f$, $\xi$ |

### Reverse / Inferential

| Question | Given | Inferred |
|---|---|---|
| Why does fusion fail without normalisation? | mAP drops from 90.0% to 89.6% | Depth features dominate (large norms), RGB information discarded — normalisation required to equalise |
| Why does AGCL use 2D search? | Imperfect rectification → vertical offset | 1D-only search misses corresponding point; 2D search covers a $k \times k$ window around the epipolar line |
| What does $\xi = 0$ imply in DeepCalib? | Eq. (1) with $\xi = 0$ | Denominator becomes $Z$; reduces to pinhole: $x = Xf/Z + u_0$ — model is backward compatible |
| Why does stacked-cascade inference help CREStereo? | High-res images → small objects lost by downsampling | Running same trained network on image pyramid combines large receptive field (low-res pass) with fine detail (high-res pass); Bad2.0 drops from 6.46 → 4.53 |

---

## A5. Standard Implementation — Concepts & Relationships

> [!note]
> This topic is conceptual (multiple distinct methods). A5 gives the key definitions, equations, and inter-relationships; A6 gives the step-by-step algorithm for each named variant.

### Classical vs. Deep Stereo — Contrast Table

| Aspect | Classical | Deep |
|---|---|---|
| Cost function | Hand-designed: SSD, SAD, NCC | Learned feature similarity (dot-product / attention) |
| Cost-volume regularisation | Local argmin or SGM global smoothness | 3-D convolutions or recurrent GRU refinement |
| Smoothness prior | Explicit (Potts model, $\alpha \cdot \text{Smooth}$) | Implicit (learned from data) |
| Post-processing | Median filter, left-right consistency | Often end-to-end |
| Examples | Block matching, ELAS, SGM | PSMNet, RAFT-Stereo, CREStereo |
| Textureless / occlusion | Poor — ambiguous costs | Better — context from attention |
| Interpretability | High — cost scores are readable | Low — opaque GRU states |
| Shared abstraction | **Build cost volume → select or regress disparity** | |

### Fusion Strategies for RGB-D

| Strategy | Mechanism | Notes |
|---|---|---|
| Pixel-level | 4-channel (R,G,B,D) input to one network | Simple; early fusion |
| Feature-level | Separate CNNs per modality; concatenate feature maps | Requires L2 normalisation to equalise scales |
| Decision-level | Separate networks; combine output predictions | Late fusion; most robust to domain mismatch |
| 2.5D | Depth as 2-D image; 2-D convolutions | Computationally light |
| 3D | Volumetric / point cloud; 3-D convolutions | Richer geometry, more expensive |

### Geometric Limits (DL cannot break these)

$$Z = \frac{fb}{d} \qquad \Delta Z = \frac{Z^2}{fb + Z} \approx \frac{Z^2}{fb} \text{ for } Z \ll fb$$

DL can reduce matching errors (improve $d$ accuracy) but cannot change the $1/d$ dependence of $Z$ or the quadratic growth of $\Delta Z$ with range.

---

## A6. Variations

### A6.1 CREStereo (Cascaded REcurrent Stereo, CVPR 2022)

**Full name:** Cascaded REcurrent Stereo matching network  
**Paper:** Li et al., "Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation," CVPR 2022  
**Code:** https://github.com/megvii-research/CREStereo

#### Problem it solves

Three practical obstacles in consumer/smartphone stereo:
1. Fine structure / thin object detail loss at high resolution
2. Non-ideal rectification — corresponding points off the epipolar scanline
3. Hard-case scenes (non-texture, repetitive texture) not covered by standard synthetic data

#### Architecture

**Feature extraction:**
- Shared-weight CNN with positional encoding + **self-attention and cross-attention** (linear attention, LoFTR-style) applied at the first cascade stage to aggregate global context into local feature maps $\mathbf{F}_1, \mathbf{F}_2 \in \mathbb{R}^{C \times H \times W}$

**AGCL — Adaptive Group Correlation Layer:**

Standard local correlation (Eq. 1):
$$\mathrm{Corr}(x, y, d) = \frac{1}{C} \sum_{i=1}^{C} \mathbf{F}_1(i, x, y)\, \mathbf{F}_2(i, x', y')$$
where $x' = x + f(d)$, $y' = y + g(d)$.

Deformable (adaptive) local correlation (Eq. 2):
$$\mathrm{Corr}(x, y, d) = \frac{1}{C} \sum_{i=1}^{C} \mathbf{F}_1(i, x, y)\, \mathbf{F}_2(i, x'', y'')$$
where $x'' = x + f(d) + dx$, $y'' = y + g(d) + dy$ with learned offsets $dx, dy$.

**2D-1D Alternate Local Search:**
- **1D mode:** $g(d) = 0$, $f(d) \in [-r, r]$ with $r = 4$ — search only along epipolar line
- **2D mode:** $k \times k$ grid with dilation $l$, $k = \sqrt{2r+1}$ — small vertical range to handle rectification error
- Alternating between 1D and 2D at different cascade levels is critical; using only one degrades accuracy

**Group-wise correlation:** features split into $\mathcal{G}$ groups; correlation computed per group; $\mathcal{G}$ volumes of $D \times H \times W$ concatenated → $\mathcal{G}D \times H \times W$ cost volume

**RUM — Recurrent Update Module:**
- GRU-based; receives correlation volume from AGCL
- Iteratively refines disparity: $f_i = f_{i-1} + \Delta f_i$
- All RUMs across cascade levels share weights

**Cascaded coarse-to-fine refinement:**

| Cascade stage $s$ | Scale | Initialisation |
|---|---|---|
| Stage 1 | $1/16$ | Zero disparity |
| Stage 2 | $1/8$ | Upsampled stage-1 output |
| Stage 3 | $1/4$ | Upsampled stage-2 output |

After stage 3: convex upsampling to full resolution (following RAFT).

**Stacked Cascades at Inference:**
- Same trained network run on a downsampled image pyramid (1, 2, or 3 stacked stages)
- No fine-tuning needed — all stages share weights
- Addresses tension between large receptive field (low-res) and small-object detail (high-res)
- 2 stacked stages: Middlebury Bad2.0 drops from 6.46 → 4.53 at 1536×2048 resolution

**Training loss** (Eq. 3):
$$\mathcal{L} = \sum_{s} \sum_{i=1}^{n} \gamma^{n-i} \left\| \mathbf{d}_{\mathrm{gt}} - \mu_s(\mathbf{f}_i^s) \right\|_1, \quad \gamma = 0.9$$
Exponentially weighted L1 over all intermediate predictions and all cascade stages; $\mu_s$ upsamples to full resolution before comparing.

#### Results

| Benchmark | Metric | CREStereo | Prior SOTA improvement |
|---|---|---|---|
| Middlebury | Bad2.0 | 3.71 | −21.73% |
| Middlebury | AvgErr | 1.15 | — |
| ETH3D | Bad1.0 | 0.98 | −59.84% |
| ETH3D | AvgErr | 0.13 | — |

Ranked 3rd on Middlebury v3 overall at publication; outperforms AANet, HSMNet, GwcNet, LEAStereo, RAFT-Stereo on fine-detail depth recovery.

#### Why AGCL handles imperfect rectification

Classical block matching and global methods assume perfectly horizontal epipolar lines. In real smartphone/consumer camera modules, lens differences and mechanical tolerances mean corresponding points may be offset vertically by a few pixels. The 2D-1D alternate search covers a small $k \times k$ grid (not just the horizontal line), and the deformable offsets further adapt the search window per-pixel — together these tolerate residual vertical misalignment that would cause classical methods to fail.

#### Cost volume size comparison

| Method | Cost volume size |
|---|---|
| RAFT-Stereo (all-pairs) | $H \times W \times W$ (3-D) |
| CREStereo (local) | $H \times W \times D$, $D \ll W$ |

CREStereo's local correlation is much smaller in memory and compute.

> [!note]
> Notation mapping: the paper uses $\mathbf{F}_1, \mathbf{F}_2$ for attended feature maps; $f(d), g(d)$ for fixed offsets; $dx, dy$ for learned deformable offsets; $\mathcal{G}$ for number of groups. Cascade scales are fractions of input resolution (1/16, 1/8, 1/4).

---

### A6.2 DeepCalib (CVMP 2018)

**Full name:** DeepCalib: A Deep Learning Approach for Automatic Intrinsic Calibration of Wide Field-of-View Cameras  
**Paper:** Bogdan et al., CVMP 2018  
**Task:** Estimate intrinsic parameters ($f$, $\xi$) from a **single general-scene image** — no calibration target needed

#### Unified Spherical Camera Model

A single distortion parameter $\xi \in [0, 1]$ (slightly above 1 for some catadioptric cameras) controls the model. The projection centre $O_c$ is located at $(0, 0, \xi)$ above the sphere centre $O$.

**Eq. (1) — Forward projection** (3-D world point $\mathbf{P}_w = (X, Y, Z)$ → 2-D image point $(x, y)$):

$$\mathbf{p} = (x, y) = \left( \frac{Xf}{\xi\sqrt{X^2+Y^2+Z^2}+Z} + u_0,\quad \frac{Yf}{\xi\sqrt{X^2+Y^2+Z^2}+Z} + v_0 \right)$$

When $\xi = 0$: reduces to pinhole projection $x = Xf/Z + u_0$, $y = Yf/Z + v_0$.

**Eq. (2) — Back-projection** (2-D image → unit sphere point $\mathbf{P}_s$, fully reversible):

$$\mathbf{P}_s = (\omega\hat{x},\ \omega\hat{y},\ \omega - \xi) \quad \text{with} \quad \omega = \frac{\xi + \sqrt{1 + (1-\xi^2)(\hat{x}^2 + \hat{y}^2)}}{\hat{x}^2 + \hat{y}^2 + 1}$$

where $[\hat{x}, \hat{y}, 1]^T \simeq \mathbf{K}^{-1}\mathbf{p}$ with $\mathbf{K} = \begin{bmatrix} f & 0 & u_0 \\ 0 & f & v_0 \\ 0 & 0 & 1 \end{bmatrix}$.

**Why unified spherical model over Brown-Conrady or division model:**
- Brown-Conrady: polynomial radial distortion; not suitable for large distortions; back-projection difficult (not closed-form)
- Division model [Fitzgibbon]: fisheye-only; theoretically irreversible
- Unified spherical: fully reversible (Eq. 2 is closed-form), single scalar $\xi$, covers perspective to catadioptric with one formula

#### Synthetic Dataset Generation

- Source: 67,000 equirectangular panoramas (9104 × 4552 px, internet + SUN360)
- Pipeline: panorama pixel $(x, y)$ → azimuth $\theta \in (0, 2\pi)$, elevation $\phi \in (-\pi/2, \pi/2)$ → map to unit sphere → back-project via Eq. (2) with desired $f \in [50, 500]$ px (step 10) and $\xi \in [0, 1.2]$ (step 0.02) → 299 × 299 synthetic wide-FOV images with known ground truth
- Backward mapping (Eq. 2) used to avoid forward-mapping holes

#### Network Architectures

All use Inception-V3 pretrained on ImageNet, fine-tuned on generated dataset ($lr = 10^{-5}$, batch 64, 80/10/10 split).

| Architecture | Description | Speed |
|---|---|---|
| **SingleNet** | One Inception-V3; two output dense layers (one for $f$, one for $\xi$) | Fastest — **best overall** |
| **DualNet** | Two independent Inception-V3 networks; one per parameter | ~2× slower than SingleNet |
| **SeqNet** | Sequential: net-1 predicts $f$; net-2 takes image + predicted $f$ (concatenated into dense layer) to predict $\xi$ | ~2× slower than SingleNet |

**Classification beats regression:** discretise output space (step sizes above); softmax + cross-entropy > sigmoid + logcosh. SingleNet-Classification is the best architecture.

#### Undistortion Procedure

1. Apply Eq. (2) with estimated $(f, \xi)$ to back-project distorted image onto unit sphere
2. Re-project with $\xi = 0$, $f = 150$ px to obtain perspective (pinhole) output

#### Performance vs. Classical Toolboxes

| Method | Input | Mean reprojection error | Time |
|---|---|---|---|
| DeepCalib | 1 general image | ~1–3.4 px | ~50 ms |
| Checkerboard toolboxes (Mei, OpenCV) | ~30 checkerboard images | < 1 px | 30–60 min |

Classical toolboxes achieve sub-pixel accuracy but require a calibration target and may fail (N/A) on wide-FOV cameras (OpenCV Brown, OpenCV Fisheye). DeepCalib works on any single in-the-wild image.

**Acceptable distortion error:** $\Delta\xi \leq 0.2$ (user study: perceived distortion score < 2 for any scene category) → 78% success rate.

**Failure modes:** strong motion blur, over-exposure, rolling shutter, ambiguity between $(f, \xi)$ pairs producing the same image.

> [!note]
> Notation mapping: paper uses $\mathbf{P}_w = (X, Y, Z)$ for WRF point; $\mathbf{P}_s = (X_s, Y_s, Z_s)$ for sphere point; $\mathbf{p} = (x, y)$ for image point; $\xi$ for distortion; $f$ for focal length (square pixels assumed); $\omega$ as auxiliary scalar. Standard O3 scheme: WRF point $(X, Y, Z)$, image point $(u, v)$ — here $(x, y)$ in the paper maps to image-plane coordinates $(x_p, y_p)$ plus principal point $(u_0, v_0)$.

---

### A6.3 Parallel RCNN — RGB-D People Detection

**Paper:** Ren, Du, Zheng (Xi'an Jiaotong University), CISP-BMEI 2017  
**Task:** Pedestrian detection from RGB-D images using dual-stream Faster RCNN

#### Depth Encoding Strategies

Raw depth: 16-bit unsigned integer (distance in mm from Kinect v2). Must encode as 3-channel image for ImageNet-pretrained CNNs.

| Encoding | Method | mAP (depth-only) | Notes |
|---|---|---|---|
| Grayscale | Normalise to [0,255], replicate to 3 channels | 74.8% | Discards most geometric information |
| HHA | Three channels: Horizontal disparity, Height above ground, Angle between surface normal and gravity | 76.3% | Rich geometric encoding; computationally expensive |
| **Jet colormap** | Normalise to [0,255], apply jet colormap → 3-channel | **79.4%** | **Best mAP, computationally efficient — recommended** |

#### Architecture

Two parallel VGG-16 streams (one for RGB, one for encoded depth) → L2 normalise each stream's feature maps → concatenate → shared RPN + Fast RCNN detection head.

**L2 channel-wise normalisation** (per-channel, independently):

$$f_i'(x, y) = \frac{f_i(x, y)}{\|f_i\|_2} \tag{Eq. 1}$$

$$\|f_i\|_2 = \left(\sum_{x=1}^{r} \sum_{y=1}^{c} |f_i(x, y)|^2\right)^{1/2} \tag{Eq. 2}$$

**Learnable scaling** (per-channel scalar $s_i$ trained by backprop):

$$F_i = s_i \cdot f_i' \tag{Eq. 3}$$

$s_i$ compensates for reduced feature magnitudes after normalisation so that learning is not slowed.

**Why normalisation is essential:**

| System | mAP |
|---|---|
| Faster RCNN (RGB only) | 90.0% |
| Parallel RCNN **without** normalisation | **89.6%** — *worse than RGB alone* |
| Parallel RCNN **with** normalisation | **91.5%** |

Without normalisation, depth features (different scale distribution from RGB) dominate and RGB information is suppressed, degrading performance.

#### Anchor modification

Standard Faster RCNN: 1:1, 2:1, 1:2 aspect ratios. Parallel RCNN modifies to **1:1, 1:2, 1:3** to better cover standing, sitting, and squatting people.

#### Training procedure

1. Train RGB Faster RCNN on RGB stream
2. Train Depth Faster RCNN on encoded depth stream
3. Discard fully-connected parts; retain convolutional parts from both
4. Concatenate convolutional parts as initialisation for Parallel RCNN
5. Fine-tune Parallel RCNN end-to-end following Faster RCNN procedure

#### Results

| Metric | Faster RCNN (RGB) | Parallel RCNN (with norm) |
|---|---|---|
| Precision | 91.5% | 92.2% |
| Recall | 91.7% | 93.3% |
| F1 score | 91.6% | 92.7% |
| mAP | 90.0% | **91.5%** |

**Why depth helps:** depth is invariant to illumination and robust to colour camouflage. Depth also provides edge information that improves bounding box localisation accuracy at higher IOU thresholds.

> [!note]
> Notation mapping: paper uses $i$ for channel index; $r, c$ for feature map rows/columns; $f_i(x,y)$ for raw feature value; $f_i'$ for normalised value; $s_i$ for learnable scale; $F_i$ for final scaled value. These match O3 scalar convention.

---

## Cross-links

- [[stereo-block-matching]] — classical SSD/SAD/NCC cost functions that deep stereo replaces
- [[stereo-rectification]] — canonical epipolar geometry that CREStereo's AGCL relaxes via 2D search
- [[depth-from-disparity]] — $Z = fb/d$ geometric law that DL cannot override
- [[stereo-depth-accuracy]] — $\Delta Z = Z^2/(fb+Z)$ limit
- [[tsai-calibration]] — classical calibration that DeepCalib replaces for wide-FOV cameras
- [[lens-distortion-and-removal]] — Brown-Conrady model that unified spherical model supersedes for large distortions
- [[pinhole-camera-model]] — unified spherical model reduces to pinhole when $\xi = 0$
