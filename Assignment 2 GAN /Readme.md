# README — Learning an Unknown PDF from Data Only using a Simple GAN (TensorFlow)

## 1. Problem Statement
We are given **samples only** (no analytical probability density function).  
The objective is to learn the distribution of a **transformed random variable** `z`, derived from NO₂ concentration `x`, using a **Generative Adversarial Network (GAN)**.

- **Real samples:** `z`
- **Fake samples:** `z_f = G(ε)` where `ε ~ N(0,1)`
- After training, the learned PDF is approximated using:
  - Histogram density estimation
  - Kernel Density Estimation (KDE)

Dataset used: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

---

## 2. Transformation Parameters (Using Roll Number)

**University Roll Number:** `r = 102303982`

### Transformation Equation
$$
z = x + a_r \sin(b_r x)
$$

### Parameter Definitions
$$
a_r = 0.5 (r \bmod 7), \quad
b_r = 0.3 ((r \bmod 5) + 1)
$$

### Computation
- $r \bmod 7 = 4 \Rightarrow a_r = 0.5 \times 4 = 2.0$
- $r \bmod 5 = 2 \Rightarrow b_r = 0.3 \times (2 + 1) = 0.9$

### Final Parameter Values
| Parameter | Value |
|---|---:|
| r | 102303982 |
| a_r | 2.0 |
| b_r | 0.9 |

---

## 3. Methodology

### Step A — Data Loading and Cleaning
1. Upload the CSV file from the local machine in Google Colab.
2. Read the dataset (with encoding fallback if required).
3. Extract NO₂ concentration as input variable `x`.
4. Clean the data:
   - Convert values to numeric
   - Remove missing values
   - Keep finite values only
   - Ensure `x ≥ 0`

---

### Step B — Transformation of x to z
The nonlinear transformation applied to each NO₂ value is:

$$
z = x + a_r \sin(b_r x)
$$

This produces the **real samples** `z`, whose probability distribution is unknown.

---

### Step C — Normalization
For stable GAN training, `z` is normalized:

$$
z_{\text{norm}} = \frac{z - \mu}{\sigma}
$$

where:
- $\mu$ is the mean of `z`
- $\sigma$ is the standard deviation of `z`

The GAN is trained on `z_norm`.

---

### Step D — GAN Architecture

A simple **MLP-based GAN** is used since the data is one-dimensional.

#### Generator (G)
- **Input:** Noise vector $ε \sim \mathcal{N}(0,1)$
- **Output:** Generated sample $z_f$

Architecture:
Dense(32) → ReLU → Dense(32) → ReLU → Dense(1)

#### Discriminator (D)
- **Input:** Real `z` or fake `z_f`
- **Output:** Probability of being real

Architecture:
Dense(32) → ReLU → Dense(32) → ReLU → Dense(1) → Sigmoid

---

### Step E — Training Procedure
Training proceeds iteratively as follows:

1. **Discriminator Training**
   - Real samples labeled as 1
   - Fake samples labeled as 0

2. **Generator Training**
   - Generate fake samples
   - Update generator so discriminator classifies them as real

**Loss Function:** Binary Cross Entropy (BCE)  
**Optimizer:** Adam (learning rate = 0.001)  
**Epochs:** 200  
**Batch Size:** 256

---

### Step F — De-normalization and PDF Estimation
After training, generator outputs are converted back to original scale:

$$
z_f = z_{f,\text{norm}} \cdot \sigma + \mu
$$

The PDF of the learned distribution is estimated using:
- Histogram density estimation
- Kernel Density Estimation (KDE)

---

## 4. Results and Visualizations

### Plot 1 — Histogram of Real z
<img width="660" height="446" alt="image" src="https://github.com/user-attachments/assets/5f13d83c-c068-4a55-9c26-a224c950d321" />

Displays the empirical distribution of the transformed variable `z`.

**Observation:**
- Distribution is right-skewed
- High density near lower values
- Long tail toward higher values

---

### Plot 2 — Histogram PDF from GAN Samples
<img width="671" height="450" alt="image" src="https://github.com/user-attachments/assets/17f21a08-1fd1-448a-9507-328cbbf5556c" />

Shows the density learned by the generator from fake samples `z_f`.

**Observation:**
- GAN captures the overall skewness
- Distribution is smoother than the real histogram

---

### Plot 3 — KDE: Real vs GAN Generated Distribution
<img width="819" height="434" alt="image" src="https://github.com/user-attachments/assets/f3b57185-16d6-47a7-bc09-3edc80448a9b" />

KDE comparison between:
- Real distribution `p(z)`
- GAN-generated distribution `p̂(z)`

**Observation:**
- Real KDE exhibits multiple sharp peaks (multi-modal)
- GAN KDE is smoother and misses several narrow modes
- Indicates **mode dropping**, common in vanilla GANs

---

## 5. Numerical Results

### Mean and Standard Deviation
| Metric | Real z | GAN z_f |
|---|---:|---:|
| Mean | 25.7543 | 28.1982 |
| Standard Deviation | 18.6193 | 16.2823 |

### Quantile Comparison (5%, 50%, 95%)
| Quantile | Real z | GAN z_f |
|---|---:|---:|
| 5% | 4.6748 | 11.2308 |
| 50% (Median) | 23.6273 | 22.9441 |
| 95% | 58.8824 | 60.4856 |

---

## 6. Interpretation of Results

### Mode Coverage
- Real distribution is highly multi-modal
- Generator captures general shape but misses small peaks
- This behavior is known as **mode dropping**

### Training Stability
- Training converges without divergence
- Loss values remain bounded
- Vanilla GANs are still sensitive to hyperparameters

### Quality of Generated Distribution
**Strengths**
- Correct skewness and tail behavior
- Median and upper quantiles are close to real values

**Weaknesses**
- Lower quantile mismatch
- KDE shows oversmoothing

---

## 7. Limitations
- Simple GAN architecture (beginner-level)
- BCE-based GAN struggles with multimodal PDFs
- KDE bandwidth affects smoothness of density estimates


