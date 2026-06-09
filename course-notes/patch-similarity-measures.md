# Patch Similarity Measures

## Table of Contents

- [[#A1. Purpose]]
- [[#A2. High-level overview]]
  - [[#Definitions]]
- [[#A3. Strengths, shortcomings & limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#a. Setup]]
  - [[#b. Steps — NCC (canonical form)]]
- [[#A6. Variations]]
  - [[#SSD — Sum of Squared Differences]]
  - [[#SAD — Sum of Absolute Differences]]
  - [[#CC — Cross-Correlation]]
  - [[#NCC stereo-cost form]]
  - [[#NCC precomputed / factored form]]

---

## A1. Purpose

Patch similarity measures quantify how alike two small image regions (patches) are, providing the core comparison function shared by three distinct tasks: (1) **feature descriptor matching** — decide whether two detected keypoints correspond; (2) **template matching** — locate a template in a search image; and (3) **stereo block-matching** — find the disparity that best aligns a left-image patch with its counterpart in the right image. Choosing the right measure governs robustness to illumination change, noise, and outlier pixels.

---

## A2. High-level overview

1. Extract a flat intensity vector from a $b \times b$ patch around each point of interest.
2. Compute the chosen measure between two patch vectors $\boldsymbol{f}$ and $\boldsymbol{g}$.
3. Interpret the score: SSD/SAD — **small = good match**; CC/NCC — **large = good match**.
4. Compare score against a threshold $T$ (descriptor matching) or take the $\arg\min / \arg\max$ over a search range (template/stereo).

### Definitions

| Symbol | Meaning |
|--------|---------|
| $\boldsymbol{f} = (f_1, \ldots, f_n)$ | Flattened intensity vector of patch $\boldsymbol{f}$ |
| $\boldsymbol{g} = (g_1, \ldots, g_n)$ | Flattened intensity vector of patch $\boldsymbol{g}$ |
| $\bar{f},\ \bar{g}$ | Mean intensities: $\bar{f} = \frac{1}{n}\sum_i f_i$ |
| $n$ | Number of pixels in the patch ($b \times b$ for a $b\times b$ window) |
| $I_L(x,y),\ I_R(x,y)$ | Left/right rectified stereo images (stereo context) |
| $d$ | Disparity (horizontal pixel offset, stereo context) |
| $W$ | Set of $(u,v)$ offsets covering the $b\times b$ block |
| $g = af + b$ | Affine intensity transform (gain $a$, bias $b$); NCC is invariant to this |

---

## A3. Strengths, shortcomings & limitations

| Measure | Small/Large = match | Robust to bias/gain? | Robust to outliers? | Notes |
|---------|--------------------|-----------------------|---------------------|-------|
| SSD | **Small** | No | No (squares amplify outliers) | Assumes Gaussian error model |
| SAD | **Small** | No | More than SSD | Assumes Laplace error model |
| CC | **Large** | No | — | Assumes linear intensity relation |
| NCC | **Large**, range $[-1,1]$ | **Yes** — invariant to $g=af+b$ | — | Undefined when either patch is uniform |

**Key shortcomings:**

- **SSD:** A single outlier pixel (e.g., value 20 vs 10 in a uniform patch) contributes $100$ to SSD vs only $10$ to SAD — SSD disproportionately penalises outliers. Not robust to a global brightness offset (bias) or contrast scaling (gain).
- **SAD:** Still not robust to affine intensity changes.
- **CC:** Relies on a linear relationship between the two patches; uncentred, so large-magnitude regions dominate.
- **NCC degeneracy:** If either patch is uniform (all pixels the same value), $\sqrt{\sum_i(f_i - \bar{f})^2} = 0$, giving division by zero. Must check standard deviation $\neq 0$ before computing.

### Likely exam questions

**Q:** Write the NCC formula and state its range and the condition under which it is undefined.

**A:** $$\mathrm{NCC}(\boldsymbol{f},\boldsymbol{g}) = \frac{\displaystyle\sum_i (f_i-\bar{f})(g_i-\bar{g})}{\displaystyle\sqrt{\sum_i(f_i-\bar{f})^2}\;\sqrt{\sum_i(g_i-\bar{g})^2}}$$ Range $[-1,1]$. Undefined when the denominator is zero, i.e. when either patch is uniform (std dev $= 0$).

---

**Q:** Why is SSD not robust to a single outlier pixel, and what measure is preferred in that case?

**A:** SSD squares the per-pixel difference, so an outlier with a large deviation contributes quadratically (e.g. diff of 10 → SSD contribution of 100). SAD is preferred because it contributes only linearly (10), making it less sensitive to outliers.

---

**Q:** Two patches: $\boldsymbol{f}$ uniform at 10, $\boldsymbol{g}$ uniform at 10 but with centre pixel 20. Compute SAD and SSD (3×3 patches).

**A:** $\mathrm{SAD} = |20-10| = 10$. $\mathrm{SSD} = (20-10)^2 = 100$. (All other pixels contribute 0.)

---

**Q:** Given three candidates with SSD scores 4, 96, 36, which is the best match and why?

**A:** SSD = 4 is the best match because SSD should be **minimised** — lower score = more similar patches.

---

**Q:** Why is NCC invariant to affine intensity changes $g_i = af_i + b$ while SSD is not?

**A:** Subtracting the mean removes bias $b$; dividing by the standard deviation removes gain $a$. SSD has no such normalisation, so a global shift or scale of one patch changes SSD but leaves NCC unchanged (as long as $a \neq 0$).

---

**Q:** Compute NCC for two 3×3 patches: $\boldsymbol{f}$ with centre 20 and all others 10; $\boldsymbol{g}$ with bottom-right 11 and all others 10. (W3L worked example)

**A:** $\bar{f} = 100/9$, $\bar{g} = 91/9$. Numerator $= -10/9$. Denominator $= \sqrt{(800/9)(8/9)} = 80/9$. $$\mathrm{NCC} = \frac{-10/9}{80/9} = -0.125$$

---

## A4. Directions of reasoning

### Forward / standard (patches → score)

**Given:** two patch vectors $\boldsymbol{f}$, $\boldsymbol{g}$ (or left/right image windows + disparity $d$).  
**Asked:** similarity score.  
**Steps:** compute means → compute deviations → plug into formula.

### Reverse / inferential (scores → best match)

**Given:** a set of similarity scores for candidate patches or disparities.  
**Asked:** which candidate is the best match, or whether a match is valid.

- For SSD/SAD: pick the candidate with the **lowest** score.
- For NCC: pick the candidate with the **highest** score (closest to $+1$).

> [!example]
> W7L stereo example: $\mathrm{SSD}(L, R_1) = 4$, $\mathrm{SSD}(L, R_2) = 96$ → $R_1$ is the better match.

**Given:** a score, decide quality of match — e.g. NCC $\approx 0.9946$ is a near-perfect match (W4T example); NCC $= -0.125$ is a poor/inverted-correlation match (W3L).

---

## A5. Standard implementation

### a. Setup

| Parameter | Value / note |
|-----------|-------------|
| Window size | $b \times b$ pixels (e.g. $3\times3$, $5\times5$, $15\times15$ for assignment) |
| Input $\boldsymbol{f}$ | Flattened pixel intensities from patch centred on interest point in image $I$ |
| Input $\boldsymbol{g}$ | Flattened pixel intensities from candidate patch in image $J$ |
| Output | Scalar similarity score |
| Convention | SSD/SAD: small = match; NCC: large (near $+1$) = match |

### b. Steps — NCC (canonical form)

1. **Extract patches.** Collect the $n = b\times b$ pixel values around each point; form $\boldsymbol{f}$ and $\boldsymbol{g}$.

2. **Compute means.**
$$\bar{f} = \frac{1}{n}\sum_{i=1}^n f_i, \qquad \bar{g} = \frac{1}{n}\sum_{i=1}^n g_i$$

3. **Compute mean-subtracted values.**
$$A_i = f_i - \bar{f}, \qquad B_i = g_i - \bar{g}$$

4. **Compute sum-of-squares (scalars $C$ and $D$).**
$$C = \sum_i A_i^2 = \sum_i (f_i - \bar{f})^2, \qquad D = \sum_i B_i^2 = \sum_i (g_i - \bar{g})^2$$

5. **Check degeneracy.** If $C = 0$ or $D = 0$ (uniform patch), NCC is undefined — skip or handle separately.

6. **Compute numerator.**
$$\mathrm{num} = \sum_i A_i B_i = \sum_i (f_i-\bar{f})(g_i-\bar{g})$$

7. **Compute NCC.**
$$\mathrm{NCC}(\boldsymbol{f},\boldsymbol{g}) = \frac{\mathrm{num}}{\sqrt{C \cdot D}}$$

> [!example]
> **W3L full example** ($3\times3$, $n=9$): $\boldsymbol{f}$ has eight 10s and one 20; $\boldsymbol{g}$ has eight 10s and a corner 11.  
> Step 2: $\bar{f}=100/9$, $\bar{g}=91/9$.  
> Step 4: $C=800/9$, $D=8/9$.  
> Step 6: $\mathrm{num}=7\times(10/81) - (80/81) - (80/81) = -10/9$.  
> Step 7: $\mathrm{NCC} = (-10/9)/(80/9) = \mathbf{-0.125}$.

> [!example]
> **W4T example** ($5\times5$, $n=25$): $\boldsymbol{f} = 1,2,\ldots,25$ (row-major); $\boldsymbol{g}$ is a slight permutation (first two entries in each row swapped, last entry 24 instead of 25).  
> $\bar{f}=\bar{g}=13$. $C=D=1300$.  
> $\sum_i A_i B_i = 1293$.  
> $\mathrm{NCC} = 1293/1300 \approx \mathbf{0.9946}$ — near-perfect match.

---

## A6. Variations

### SSD — Sum of Squared Differences

$$\mathrm{SSD}(\boldsymbol{f},\boldsymbol{g}) = \sum_i (f_i - g_i)^2 \quad \text{(small = match)}$$

**When to use:** fast to compute; appropriate when residuals are Gaussian-distributed.  
**Robustness:** not robust to bias, gain, or single outlier pixels.

> [!example]
> **W6T template matching** ($3\times3$ patches): SSD $= 81+9+0+0+1+49+9+16+16 = \mathbf{181}$.

> [!example]
> **W7L stereo** ($3\times3$ block): $\mathrm{SSD}(L,R_1)=(20-18)^2=4$; $\mathrm{SSD}(L,R_2)=8\times4+64=96$. $R_1$ is the best match.

> [!example]
> **W3L (two pairs both SSD=36):** $\boldsymbol{f}$ all-10 vs $\boldsymbol{g}$ all-12 → SSD $= 9\times4=36$. Also $\boldsymbol{f}$ with centre-14 vs $\boldsymbol{g}$ with centre-12 → SSD $= 9\times4=36$. Same score, qualitatively different patches — SSD cannot distinguish them.

---

### SAD — Sum of Absolute Differences

$$\mathrm{SAD}(\boldsymbol{f},\boldsymbol{g}) = \sum_i |f_i - g_i| \quad \text{(small = match)}$$

**When to use:** preferred when residuals follow a Laplace distribution, or when outlier robustness matters more than speed.  
**Robustness:** more robust to outliers than SSD; still not robust to affine intensity change.

> [!example]
> **W6T template matching:** SAD $= 9+3+0+0+1+7+3+4+4 = \mathbf{31}$.

> [!example]
> **W3L outlier comparison** ($3\times3$, centre pixel 20 vs 10, all others identical): SAD $= 10$, SSD $= 100$.

---

### CC — Cross-Correlation

$$\mathrm{CC}(\boldsymbol{f},\boldsymbol{g}) = \sum_i f_i g_i \quad \text{(large = match)}$$

**When to use:** rarely used alone; NCC is almost always preferred.  
**Robustness:** assumes a linear relationship between patches; better than SSD/SAD under uniform illumination change, but still affected by absolute intensity levels.

> [!note]
> The W3L slide corrects the CC example from 1000 to **1010** after careful pixel counting: $7\times100 + 200 + 110 = 1010$ for the $3\times3$ pair with $\bar{f}$ having centre 20 and $\bar{g}$ having bottom-right 11.

---

### NCC stereo-cost form

In stereo block-matching the same NCC formula is re-expressed with left/right image notation and a 2D block sum. Written as a cost $E$ (to **maximise** for best disparity):

$$E(x,y,d) = \frac{\displaystyle\sum_{u=1}^{b}\sum_{v=1}^{b} I_L(x-u,\,y-v)\cdot I_R(x-u-d,\,y-v)}{\displaystyle\sqrt{\sum_{u}\sum_{v} I_L(x-u,\,y-v)^2 \;\cdot\; \sum_{u}\sum_{v} I_R(x-u-d,\,y-v)^2}}$$

> [!note]
> This is the same mathematical object as the descriptor NCC above with $\boldsymbol{f} \leftrightarrow$ left-image block and $\boldsymbol{g} \leftrightarrow$ right-image block at shift $d$. The stereo form omits mean-subtraction in some slide formulations (W7L/W8T) — this is equivalent to using CC normalised by RMS rather than the fully centred NCC. Be aware of which slide version is expected; the mean-subtracted form is the general definition.

**Usage context:** inside the block-matching algorithm (see [[stereo-block-matching]]). The best disparity is $d^* = \arg\max_d E(x,y,d)$.

Similarly for the stereo SSD and SAD cost forms (W7L/W8T):

$$E_{\mathrm{SSD}}(x,y,d) = \sum_{u,v}\bigl[I_L(x-u,y-v) - I_R(x-u-d,y-v)\bigr]^2$$

$$E_{\mathrm{SAD}}(x,y,d) = \sum_{u,v}\bigl|I_L(x-u,y-v) - I_R(x-u-d,y-v)\bigr|$$

Best disparity for SSD/SAD: $d^* = \arg\min_d E(x,y,d)$.

---

### NCC precomputed / factored form

To avoid recomputing means and sums-of-squares for every pair at match time, **precompute per-descriptor components** and store them:

$$\hat{f}_i = \frac{f_i - \bar{f}}{\sqrt{\sum_j(f_j-\bar{f})^2}}, \qquad \hat{g}_i = \frac{g_i - \bar{g}}{\sqrt{\sum_j(g_j-\bar{g})^2}}$$

Then at match time:

$$\mathrm{NCC}(\boldsymbol{f},\boldsymbol{g}) = \sum_i \hat{f}_i \cdot \hat{g}_i = \hat{\boldsymbol{f}}^\top \hat{\boldsymbol{g}}$$

This reduces matching to a **dot product** between two stored unit-norm, mean-subtracted vectors — $O(n)$ per pair with no extra divisions or square roots at query time.

**Storage per descriptor:** the $n$ values $\hat{f}_i$ (mean-subtracted + $\ell_2$-normalised) — or equivalently, the mean-subtracted values $A_i = f_i - \bar{f}$ plus the scalar $C = \sum_i A_i^2$, from which $\hat{f}_i = A_i/\sqrt{C}$.

> [!note]
> W4T shorthand: $A_i = f_i-\bar{f}$, $B_i = g_i-\bar{g}$, $C = \sum A_i^2$, $D = \sum B_i^2$. Then $\mathrm{NCC} = \sum(A_i B_i)/\sqrt{CD}$.

**Degeneracy check:** before storing, verify $C \neq 0$ (standard deviation $\neq 0$); if zero, the patch is uniform and NCC is undefined.

---

## Appearance contexts summary

| Context | Notation used | Score direction | Typical window |
|---------|--------------|-----------------|----------------|
| [[feature-descriptors]] matching | $\boldsymbol{f}, \boldsymbol{g}$ as $D$-dim vectors | NCC large; SSD/SAD small | $11\times11$, $15\times15$ |
| Template matching | same $\boldsymbol{f}, \boldsymbol{g}$ notation | same | variable |
| [[stereo-block-matching]] cost | $I_L, I_R$ with shift $d$, block $b\times b$ | NCC large; SSD/SAD small | $3\times3$ upward |

See also: [[feature-matching]], [[non-maximum-suppression]].

> [!warning]
> The stereo-cost NCC formula in W7L/W8T does not subtract the patch mean from $I_L$ and $I_R$ before multiplying — it normalises by the root-mean-square rather than by the standard deviation. This is a simplified (non-zero-mean) variant. The fully centred formula (subtracting $\bar{f}$ and $\bar{g}$) is the standard NCC and is what is used in the descriptor context. When answering exam questions, use the centred form unless a specific stereo-cost formula is requested.
