# Stereo Depth Accuracy

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-level overview]]
  - [[#Key Definitions]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps — $\Delta Z$ derivation]]
- [[#A6. Variations]]

---

## A1. Purpose

Stereo depth ($Z = fb/d$) is fundamentally limited because disparity $d$ is a **discrete integer**: a 1-pixel change in disparity causes a finite jump in depth called $\Delta Z$. This topic quantifies that depth resolution, derives $\Delta Z$ as a function of $Z$, establishes the system's measurable depth range ($Z_{\min}$, $Z_{\max}$), and identifies three strategies to improve accuracy. Reach for this analysis whenever you need to know how precise a stereo rig's depth estimates can be, or to spec a rig for a required accuracy.

---

## A2. High-level overview

1. Observe that $d$ is integer-valued → consecutive disparities $d$ and $d+1$ map to different depths → $\Delta Z$ is the gap between them.
2. Derive $\Delta Z$ from the difference formula, then substitute $d = fb/Z$ to express it in terms of $Z$.
3. Identify $Z_{\min}$ (max disparity = image width) and $Z_{\max}$ (desired $\Delta Z$ tolerance or min disparity = 1 px).
4. Apply three improvement strategies: widen baseline, shrink pixel size, add camera views.

### Key Definitions

| Symbol | Meaning |
|--------|---------|
| $f$ | Focal length divided by pixel size (dimensionless, pixels): $f = f_{mm}/s$ |
| $f_{mm}$ | Focal length in millimetres |
| $s$ | Pixel size (mm/pixel) |
| $b$ | Baseline — physical separation of the two optical centres (mm) |
| $d$ | Disparity in pixels — **discrete integer**, $d = x_L - x_R$ |
| $Z$ | Scene depth along optical axis (mm) |
| $\Delta Z$ | Depth resolution — depth change for a 1-pixel disparity step (mm) |
| $Z_{\min}$ | Closest measurable depth (maximum disparity $\approx$ image width $w$) |
| $Z_{\max}$ | Farthest measurable depth — set by desired $\Delta Z$ tolerance, or by $d=1$ pixel |
| $w$ | Image width in pixels |
| $\mu$ | Pixel width in mm (same as $s$, alternate notation) |

> [!note]
> The slides use $f$ for both focal length in mm and focal length in pixels depending on context. In this file $f$ always denotes the **dimensionless** pixel-units value ($f_{mm}/s$) unless explicitly subscripted $f_{mm}$. This matches the derivation of $\Delta Z = Z^2/(fb+Z)$.

---

## A3. Strengths, shortcomings & limitations

**Strengths**
- The $\Delta Z$ formula is closed-form and easy to evaluate for any rig configuration.
- Makes explicit which parameters matter: increasing $fb$ directly reduces $\Delta Z$.
- Provides principled design equations for specifying a stereo rig to meet an accuracy requirement.

**Shortcomings / Limitations**
- Assumes **perfect integer-pixel disparity** — sub-pixel matching can do better but requires additional processing.
- $\Delta Z$ grows **quadratically** with $Z$: depth accuracy deteriorates rapidly for distant objects.
- Wide baseline (larger $b$) improves depth resolution but makes correspondence matching harder ("wide baseline stereo is harder than narrow baseline").
- Smaller pixel size $s$ (larger $f$) improves $\Delta Z$ but collects less light → more noise, more computation, more sensitive to calibration errors.
- Multi-view improves the depth estimate statistically but does not change the geometric limit per view.
- Deep learning can improve matching quality but **cannot overcome the geometric $Z = fb/d$ limit**.

### Likely exam questions

**Q1:** Why does stereo depth have finite resolution? Derive the expression for $\Delta Z$.

**A:** Disparity $d$ is a discrete integer. The smallest possible disparity change is 1 pixel. The resulting depth gap is:
$$\Delta Z = \frac{fb}{d} - \frac{fb}{d+1} = \frac{fb}{d(d+1)}$$
Substituting $d = fb/Z$:
$$\boxed{\Delta Z = \frac{Z^2}{fb + Z} \approx \frac{Z^2}{fb} \text{ for } Z \gg fb}$$

---

**Q2:** Using $f_{mm} = 8\,\text{mm}$, $b = 120\,\text{mm}$, $s = 0.004\,\text{mm/px}$ and disparity $d = 40\,\text{px}$, compute $Z$ and $\Delta Z$.

**A:**
- $f = 8/0.004 = 2000\,\text{px}$
- $Z = (2000 \times 120)/40 = 6000\,\text{mm}$
- $Z(d+1) = (2000 \times 120)/41 \approx 5854\,\text{mm}$
- $\Delta Z = 6000 - 5854 = 146\,\text{mm}$ (≈15 cm at 6 m)

---

**Q3:** For the same rig, what is the depth error at $Z = 10\,\text{m}$?

**A:**
- $d = (2000 \times 120)/10000 = 24\,\text{px}$
- $\Delta Z = (2000 \times 120)/(24 \times 25) = 240000/600 = 400\,\text{mm}$ (40 cm at 10 m)

---

> [!example]
> **Likely exam question (BONUS hint — "which part needs more and how?"):**
>
> **Q4:** A stereo rig struggles to resolve depth accurately for **far objects**. Which parameter should you increase, and why?
>
> **A:** Far objects have **small disparity** (the limiting case). $\Delta Z \approx Z^2/(fb)$ for large $Z$, so $\Delta Z$ is inversely proportional to $fb$. To halve $\Delta Z$ at a given far distance: either **double the baseline** $b$ (doubles disparity for the same $Z$, directly halving $\Delta Z$) or **double $f$** (halve pixel size $s$, which also doubles the pixel-unit disparity). The baseline increase is usually more practical; the trade-off is that wide-baseline matching becomes harder as appearance change increases. This is the **far-depth accuracy** bottleneck — the exam may ask "which part of the scene needs more baseline?" → **distant parts** with small disparity.

---

**Q5:** Given a required depth accuracy $\Delta Z_{\text{req}}$, write the formula for the maximum usable range $Z_{\max}$.

**A:** Rearrange $\Delta Z = Z^2/(fb+Z)$ as $Z^2 - \Delta Z_{\text{req}}\,Z - \Delta Z_{\text{req}}\,fb = 0$:
$$Z_{\max} = \frac{\Delta Z_{\text{req}} + \sqrt{\Delta Z_{\text{req}}^2 + 4\,\Delta Z_{\text{req}}\cdot fb}}{2}$$
(take the positive root; the $\pm$ in some slide formulations refers to both roots of the quadratic, only the positive one is physically meaningful.)

---

**Q6:** What are the three strategies for improving stereo depth accuracy, and what are their trade-offs?

**A:**

| Strategy | Effect on $\Delta Z$ | Trade-off |
|----------|---------------------|-----------|
| Wider baseline $b$ | $\downarrow \Delta Z$ (larger $d$ for same $Z$) | Harder correspondence matching (greater appearance change) |
| Smaller pixel size $s$ (higher resolution) | $\downarrow \Delta Z$ (larger $f$) | Less light per pixel (more noise); more computation; calibration errors more significant |
| More camera views (multi-view bundle) | Depth estimates → Gaussian distribution; mean = MLE | Requires more cameras, synchronisation, and computation |

---

## A4. Directions of reasoning

### Forward (standard): rig parameters → depth resolution

- **Given:** $f_{mm}$, $s$, $b$, $d$ (or equivalently $Z$)
- **Asked:** $\Delta Z$ (how precise is this rig at this depth?)
- **Steps:** compute $f = f_{mm}/s$; if $d$ is known use $\Delta Z = fb/[d(d+1)]$; if $Z$ is known use $\Delta Z = Z^2/(fb+Z)$.

### Reverse / inferential: required accuracy → rig design or range limit

- **Given:** desired $\Delta Z_{\text{req}}$ and rig parameters $f, b$
- **Asked:** $Z_{\max}$ (how far can you reliably measure?) or required $b$ (what baseline do you need at range $Z$?)
- **Key manipulation:**
  - For $Z_{\max}$: solve the quadratic $Z^2 - \Delta Z_{\text{req}}\,(fb+Z) = 0$:
    $$Z_{\max} = \frac{\Delta Z_{\text{req}} + \sqrt{\Delta Z_{\text{req}}^2 + 4\,\Delta Z_{\text{req}}\cdot fb}}{2}$$
  - For required baseline at range $Z$ with tolerance $\Delta Z_{\text{req}}$: rearrange $\Delta Z_{\text{req}} = Z^2/(fb+Z)$ to get:
    $$b_{\min} = \frac{Z^2 - \Delta Z_{\text{req}}\,Z}{\Delta Z_{\text{req}}\,f} = \frac{Z(Z - \Delta Z_{\text{req}})}{\Delta Z_{\text{req}}\,f}$$
  - For $Z_{\min}$: maximum disparity $\approx$ image width in pixels: $Z_{\min} = bf/w$.

---

## A5. Standard implementation

### A5a. Setup

- **Inputs:** calibrated stereo rig — $f_{mm}$ (focal length in mm), $s$ (pixel size in mm/px), $b$ (baseline in mm), $w$ (image width in pixels).
- **Derived parameter:** $f = f_{mm}/s$ (focal length in pixels, dimensionless).
- **Output of interest:** $\Delta Z$ at a given depth $Z$, or $Z_{\max}$ for a required $\Delta Z$.
- **Assumption:** disparity $d$ is an integer (no sub-pixel refinement); images are rectified (horizontal search only); pinhole camera model.

### A5b. Steps — $\Delta Z$ derivation

**Step 1.** Write depth for disparity $d$ and $d+1$:

$$Z(d) = \frac{fb}{d}, \qquad Z(d+1) = \frac{fb}{d+1}$$

**Step 2.** Compute the difference (depth resolution):

$$\Delta Z = Z(d) - Z(d+1) = \frac{fb}{d} - \frac{fb}{d+1} = \frac{fb(d+1) - fbd}{d(d+1)} = \frac{fb}{d(d+1)}$$

**Step 3.** Substitute $d = fb/Z$ to express in terms of depth:

$$\Delta Z = \frac{fb}{\dfrac{fb}{Z}\!\left(\dfrac{fb}{Z}+1\right)} = \frac{fb}{\dfrac{fb^2 + fbZ}{Z^2}} = \frac{fb \cdot Z^2}{fb(fb+Z)}$$

$$\boxed{\Delta Z = \frac{Z^2}{fb + Z}}$$

**Step 4.** Far-object approximation ($Z \gg fb$):

$$\Delta Z \approx \frac{Z^2}{fb}$$

Depth error grows **quadratically** with distance.

**Step 5.** Depth range limits:

$$Z_{\min} = \frac{bf}{w} \quad \text{(max disparity = image width } w \text{ px)}$$

$$Z_{\max} = \frac{\Delta Z_{\text{req}} + \sqrt{\Delta Z_{\text{req}}^2 + 4\,\Delta Z_{\text{req}}\cdot fb}}{2} \quad \text{(from required accuracy)}$$

> [!example]
> **Depth-error-vs-distance table** ($f = 2000\,\text{px}$, $b = 120\,\text{mm}$):
>
> | Disparity $d$ (px) | Depth $Z$ (mm) | $\Delta Z$ for 1 px (mm) |
> |---------------------|----------------|--------------------------|
> | 100 | 2400 | 24 |
> | 40 | 6000 | 146 |
> | 20 | 12000 | 571 |
> | 10 | 24000 | **2182** |
>
> At $d = 10\,\text{px}$ (24 m), a 1-pixel disparity error produces **over 2 m** of depth error.

> [!example]
> **W8T Example 4 — unit conversion ($f_{mm}$ → dimensionless $f$):**
>
> Given: $f_{mm} = 2\,\text{mm}$, $b = 65\,\text{mm}$, $s = 0.002\,\text{mm/px}$, $d_{px} = 170$
>
> 1. Convert: $f = f_{mm}/s = 2/0.002 = 1000\,\text{px}$
> 2. $Z = (1000 \times 65)/170 = 382.35\,\text{mm}$
> 3. $\Delta Z = 382.35^2\,/(1000 \times 65 + 382.35) = 146337/(65382.35) \approx \mathbf{2.23\,mm}$
>
> At close range (382 mm) with large disparity (170 px), depth resolution is excellent (2.2 mm).

---

## A6. Variations

### Variant 1: $f$ in mm, $d$ in pixels (mixed units)

When the problem gives focal length in mm directly and disparity in pixels, the depth formula becomes $Z = fb/(sd)$ (insert pixel size $s$ to convert $d$ to mm). The depth resolution then is:

$$\Delta Z = \frac{fb}{sd} - \frac{fb}{s(d+1)} = \frac{fb}{sd(d+1)}$$

Substituting $d = fb/(sZ)$:

$$\boxed{\Delta Z = \frac{sZ^2}{fb + sZ}}$$

This reduces to $Z^2/(fb+Z)$ when $f$ is expressed in pixels (since the $s$ factors cancel). Use this form when $f_{mm}$ is given and you do not want to convert explicitly.

> [!note]
> Original slide notation (W8T p. 16): the formula $\Delta Z = sZ^2/(fb+sZ)$ uses $f$ in mm throughout. The dimensionless form $\Delta Z = Z^2/(fb+Z)$ uses $f$ in pixels. They are algebraically equivalent — choose whichever unit set the problem provides.

### Variant 2: $Z_{\min}$ and $Z_{\max}$ from disparity range

When the maximum disparity is determined by the image width $w$ (in pixels), and the minimum meaningful disparity is 1 pixel:

$$Z_{\min} = \frac{bf}{w}, \qquad Z_{\max} = \frac{bf}{1} = bf$$

In mixed units (mm for $f$, pixels for $w$):

$$Z_{\min} = \frac{bf}{\mu w}, \qquad Z_{\max} = \frac{bf}{\mu}$$

where $\mu$ is pixel width in mm. These bounds come from the disparity range $[1, w-1]$ pixels; the slides use these to analyse the effect of baseline on the measurable depth range.

### Variant 3: Multi-view (bundle) depth estimation

With $N$ cameras observing the same scene point, each yielding a noisy depth estimate, the joint distribution converges to a **Gaussian** as $N$ increases. The **mean** of the distribution is the maximum likelihood estimate (MLE) of depth. This does not change $\Delta Z$ per view but reduces the overall depth uncertainty statistically. (Reference: Eade et al., "Scalable Monocular SLAM".)

---

**Related topics:** [[depth-from-disparity]] · [[stereo-block-matching]] · [[stereo-rectification]]
