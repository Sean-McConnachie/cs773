# Image Filtering and Edge Detection

## Table of Contents
- [[#A1. Purpose]]
- [[#A2. High-level overview]]
  - [[#Key definitions]]
- [[#A3. Strengths, shortcomings and limitations]]
  - [[#Likely exam questions]]
- [[#A4. Directions of reasoning]]
- [[#A5. Standard implementation]]
  - [[#A5a. Setup]]
  - [[#A5b. Steps]]
- [[#A6. Variations]]

---

## A1. Purpose

Image filtering suppresses noise and prepares images for higher-level processing such as edge detection and feature extraction. Edge detection locates boundaries between regions by finding locations where intensity changes rapidly — used as the first step in corner detectors ([[harris-corner-detector]]), feature-point pipelines ([[feature-points-and-autocorrelation]]), and stereo matching. The two core filters (mean and median) trade noise suppression against edge preservation; the gradient-based detector then converts the smoothed image into a binary or magnitude map of edges.

---

## A2. High-level overview

1. Compute the **image histogram** to understand intensity distribution.
2. Apply a **spatial filter** (mean or median) to the image to suppress noise.
3. Compute **image gradients** $g_x, g_y$ using finite-difference or Sobel/Prewitt kernels.
4. Compute **gradient magnitude** $g = \sqrt{g_x^2 + g_y^2}$.
5. Detect edges as locations where the gradient magnitude is locally maximal (extrema of the first derivative).

### Key definitions

| Term | Definition |
|------|-----------|
| Image $f(x,y)$ | Discrete intensity function; each pixel has intensity $\in \{0, 1, \ldots, 255\}$ |
| Histogram $H(q)$ | Count of pixels with intensity $q$; maps each grey level to a non-negative integer |
| Dirac delta $\delta(s)$ | $1$ if $s=0$, else $0$; used to formulate the histogram as a sum |
| Mean filter | Linear $n \times n$ kernel replacing each pixel by the average of its neighbourhood |
| Median filter | Non-linear filter replacing each pixel by the median of its neighbourhood |
| $g_x(x,y)$ | Horizontal image gradient (partial derivative w.r.t. $x$) |
| $g_y(x,y)$ | Vertical image gradient (partial derivative w.r.t. $y$) |
| Gradient magnitude $g$ | $\sqrt{g_x^2 + g_y^2}$; scalar measure of edge strength at each pixel |
| Sobel kernel | Weighted finite-difference kernel; normalised by $\tfrac{1}{8}$ in this course |
| Prewitt kernel | Uniform-weight finite-difference kernel (no weighting of centre column/row) |
| Edge | Pixel location where the first derivative (gradient magnitude) is at a local maximum |

---

## A3. Strengths, shortcomings and limitations

**Image histogram**
- Compact global descriptor of intensity distribution; $O(K)$ to compute ($K$ = number of pixels).
- **Limitation:** Loses all spatial information — two images with completely different structures can share an identical histogram.

**Mean filter**
- Simple, fast (linear convolution); effective against Gaussian noise.
- **Limitation:** Blurs edges because it averages across boundaries; sensitive to outliers (a single very bright/dark pixel raises the mean).

**Median filter**
- Highly effective at removing salt-and-pepper noise while preserving sharp edges.
- **Limitation:** Non-linear — cannot be implemented as a convolution; slower than mean filter for large kernels; can distort fine texture.

**Gradient-based edge detection**
- Directly targets intensity transitions; works well on clean or pre-smoothed images.
- **Limitation:** Sensitive to noise (amplifies high-frequency content); must choose a threshold for binarisation.

### Likely exam questions

**Q:** Write the formula for the image histogram $H(q)$ using Dirac deltas.

**A:**
$$H(q) = \sum_{(x,y)=(0,0)}^{(M,N)} \delta\bigl(I(x,y) - q\bigr), \quad q \in \{0,1,\ldots,255\}$$
where $\delta(s)=1$ if $s=0$, else $0$. The sum over all intensities satisfies $\sum_{q=0}^{255} H(q) = K$ (total pixel count).

---

**Q:** Can two different images produce the same histogram? Explain.

**A:** Yes. The histogram records only how many pixels have each intensity, not where they are. A checkerboard and a uniform grey image can share the same histogram if their pixel-value counts match. Spatial arrangement is discarded.

---

**Q:** A 3×3 neighbourhood around pixel $(3,2)$ has values $\{32, 8, 8, 8, 2, 4, 2, 16, 2\}$. What does a mean filter return? What does a median filter return?

> [!example]
> **Mean filter:** Sum $= 32+8+8+8+2+4+2+16+2 = 82$; mean $= 82/9 = 9.11 \to \mathbf{9}$ (integer truncation).
>
> **Median filter:** Sorted: $2, 2, 2, 4, \mathbf{8}, 8, 8, 16, 32$. Median (5th of 9) $= \mathbf{8}$.

---

**Q:** Write the central-difference formulas for $g_x$ and $g_y$.

**A:**
$$g_x(x,y) = \frac{f(x+1,y) - f(x-1,y)}{2}, \qquad g_y(x,y) = \frac{f(x,y+1) - f(x,y-1)}{2}$$

---

**Q:** Apply the vertical Sobel kernel (normalised by $1/8$) to the following 3×3 neighbourhood and report the result.

> [!example]
> Sobel kernel (detects horizontal edges / vertical gradient):
> $$K = \frac{1}{8}\begin{bmatrix}-1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1\end{bmatrix}$$
> From Q13 (W6T): neighbourhood top row $= (64, 64, 16)$, bottom row $= (16, 16, 16)$, middle row zeroed by kernel.
> $$\frac{(-1)(64) + (-2)(64) + (-1)(16) + (1)(16) + (2)(16) + (1)(16)}{8} = \frac{-144}{8} = \mathbf{-18}$$

---

**Q:** Why is the median filter preferred over the mean filter for salt-and-pepper noise?

**A:** Salt-and-pepper noise introduces extreme outlier values (0 or 255). The mean is pulled far from the true value by even one outlier, whereas the median is robust — extreme values are moved to the tails of the sorted list and do not influence the centre value. The median also preserves edges because it selects an actual pixel value rather than blending across a boundary.

---

## A4. Directions of reasoning

### Forward / standard (image → filtered image or edge map)

- **Given:** Raw image $f(x,y)$ plus a chosen filter kernel.
- **Asked:** Filtered pixel value or gradient magnitude at a given location.
- **Key manipulation:** Place the kernel over the neighbourhood, multiply element-wise, sum (and divide by normalisation factor for Sobel). For median: sort neighbourhood and pick centre value.

### Reverse / inferential (output → kernel or filter choice)

- **Given:** Observed smoothing behaviour (edge blurring vs. edge preservation) or edge-detection performance.
- **Asked:** Which filter was used, or why edges appear/disappear.
- **Key inference:** If edges are blurred, a linear (mean) filter was used; if edges are sharp in the output and only noise removed, a median filter was used. If gradient magnitudes are high at a boundary but noisy, the image was not pre-smoothed.

---

## A5. Standard implementation

### A5a. Setup

- **Input:** Greyscale image $f$, size $M \times N$ pixels, intensity $\in \{0,\ldots,255\}$.
- **Filter:** $n \times n$ kernel (typically $3 \times 3$); centred on the target pixel.
- **Output:** Filtered image or gradient magnitude image $g$.
- **Assumption:** Boundary pixels are either padded (zero-padding or mirror) or excluded.
- **Notation:** $f(x,y)$ = original intensity; $g_x, g_y$ = gradients; $g$ = gradient magnitude.

### A5b. Steps

**Step 1 — Compute the histogram (optional diagnostic)**

$$H(q) = \sum_{(x,y)} \delta\bigl(f(x,y) - q\bigr), \quad q = 0, 1, \ldots, 255$$

Verify: $\sum_q H(q) = K = M \times N$.

**Step 2a — Apply mean filter (linear)**

For each interior pixel $(x,y)$, extract the $n \times n$ neighbourhood and compute:

$$\hat{f}(x,y) = \frac{1}{n^2} \sum_{(i,j) \in \mathcal{N}(x,y)} f(i,j)$$

Truncate to integer if required.

**Step 2b — Apply median filter (non-linear, alternative to Step 2a)**

For each interior pixel $(x,y)$, extract the $n \times n$ neighbourhood, sort all $n^2$ values, and take the middle value:

$$\hat{f}(x,y) = \operatorname{median}\bigl\{f(i,j) : (i,j) \in \mathcal{N}(x,y)\bigr\}$$

**Step 3 — Compute horizontal gradient $g_x$**

Central differences (or Sobel/Prewitt kernel applied via convolution):

$$g_x(x,y) = \frac{f(x+1,y) - f(x-1,y)}{2}$$

With **horizontal Sobel kernel** (detects vertical edges):

$$K_x = \frac{1}{8}\begin{bmatrix}-1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{bmatrix}$$

**Step 4 — Compute vertical gradient $g_y$**

$$g_y(x,y) = \frac{f(x,y+1) - f(x,y-1)}{2}$$

With **vertical Sobel kernel** (detects horizontal edges):

$$K_y = \frac{1}{8}\begin{bmatrix}-1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1\end{bmatrix}$$

**Step 5 — Compute gradient magnitude**

$$g(x,y) = \sqrt{g_x^2(x,y) + g_y^2(x,y)}$$

**Step 6 — Detect edges**

Edges correspond to **local maxima of $g(x,y)$** (extrema of the first derivative of intensity). Apply a threshold or non-maximum suppression (see [[non-maximum-suppression]]) to produce a binary edge map.

> [!note]
> The Sobel kernel combines smoothing (Gaussian-like weighting) with differentiation in one pass, making it more noise-robust than the bare central-difference formula. The course normalisation factor is $\tfrac{1}{8}$.

---

## A6. Variations

### Mean filter vs. Median filter

| Property | Mean filter | Median filter |
|----------|-------------|---------------|
| Linearity | Linear (convolution) | Non-linear |
| Noise type | Gaussian noise | Salt-and-pepper noise |
| Edge handling | Blurs edges | Preserves edges |
| Outlier sensitivity | Sensitive (outlier shifts mean) | Robust (outlier moves to tail) |
| Speed | Fast (FFT-based convolution) | Slower for large kernels |

Use **mean** when noise is Gaussian and edge blurring is acceptable. Use **median** when impulse/salt-and-pepper noise is present and edges must be preserved.

### Central differences vs. Sobel kernel

| Property | Central differences | Sobel kernel |
|----------|--------------------|----|
| Formula | $\frac{f(x+1,y)-f(x-1,y)}{2}$ | Weighted 3×3 kernel, $\div 8$ |
| Noise robustness | Low (pure derivative) | Higher (Gaussian smoothing built in) |
| Computational cost | 2 lookups per direction | 6 multiplications + 5 additions per direction |
| Course notation | Step 3–4 above | Sobel $K_x$, $K_y$ above |

### Prewitt kernel

Replaces Sobel's weighted centre row/column with uniform weights:

$$K_x^{\text{Prewitt}} = \frac{1}{6}\begin{bmatrix}-1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1\end{bmatrix}, \quad K_y^{\text{Prewitt}} = \frac{1}{6}\begin{bmatrix}-1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1\end{bmatrix}$$

Slightly less noise-robust than Sobel; used when isotropy in the smoothing component is not required.

### Connection to Harris corner detector

The same gradient maps $g_x, g_y$ (or $I_x, I_y$ in Harris notation) are the inputs to the Harris structure tensor. See [[harris-corner-detector]] and [[feature-points-and-autocorrelation]] for how gradient products $I_x^2, I_xI_y, I_y^2$ are accumulated into the $2\times2$ matrix $\mathbf{H}$. See also [[vector-matrix-algebra]] for the linear algebra underpinning convolution and matrix operations.
