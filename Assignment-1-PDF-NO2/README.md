# Assignment-1: Learn Probability Density Functions using Roll-Number-Parameterized Non-Linear Model

## Title
**Learn Probability Density Functions using Roll-Number-Parameterized Non-Linear Model**

## Dataset
- Kaggle Dataset: **India Air Quality Data**
- Feature used: **NO2 (x)**
- Link: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

---

## Objective
This assignment aims to:
1. Transform the NO2 feature values \(x\) into a new variable \(z\) using a roll-number-based non-linear transformation.
2. Learn the parameters of the probability density function:

\[
\hat{p}(z) = c \cdot e^{-\lambda(z-\mu)^2}
\]

where the parameters to estimate are: **\(\lambda\), \(\mu\), and \(c\)**.

---

## Roll Number Details
University Roll Number:

\[
r = 102303982
\]

The transformation parameters are defined as:

\[
a_r = 0.05 \cdot (r \bmod 7)
\]
\[
b_r = 0.3 \cdot ((r \bmod 5)+1)
\]

Calculated values:

- \(r \bmod 7 = 4 \Rightarrow a_r = 0.05 \times 4 = 0.20\)
- \(r \bmod 5 = 2 \Rightarrow b_r = 0.3 \times (2+1) = 0.90\)

So the final transformation becomes:

\[
z = x + 0.20 \cdot \sin(0.90x)
\]

---

## Step 1: Data Preprocessing
- Loaded the dataset CSV file.
- Selected the **NO2** column.
- Converted NO2 values to numeric format.
- Removed missing values (NaN) and invalid entries.
- Final valid sample size:

\[
n = 419509
\]

---

## Step 2: Transformation (x → z)
For each NO2 value \(x_i\), transformed value \(z_i\) is computed as:

\[
z_i = x_i + 0.20 \cdot \sin(0.90x_i)
\]

---

## Step 3: Learning the PDF Parameters
We are given the PDF model:

\[
\hat{p}(z) = c \cdot e^{-\lambda(z-\mu)^2}
\]

To estimate parameters, we use the MLE (Maximum Likelihood Estimation) approach:

### Mean
\[
\mu = \frac{1}{n}\sum_{i=1}^{n} z_i
\]

### Variance (MLE)
\[
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(z_i-\mu)^2
\]

### Lambda
\[
\lambda = \frac{1}{2\sigma^2}
\]

### Constant c
To keep the density normalized:

\[
c = \sqrt{\frac{\lambda}{\pi}}
\]

---

## Final Estimated Parameters
The estimated parameters obtained from the transformed dataset are:

- **Mean (\(\mu\))** = `25.804091267939`
- **Variance (\(\sigma^2\))** = `342.610945104577`
- **Lambda (\(\lambda\))** = `0.001459381281`
- **Constant (c)** = `0.021553085382`

---

## Results Visualization

### 1) Histogram of Transformed Variable z (Density)
This plot shows the distribution of transformed values \(z\).

![Transformed variable z histogram](outputs/histogram.png)

---

### 2) Histogram of z with Fitted PDF Curve
The orange curve represents the learned density function:

\[
\hat{p}(z) = c \cdot e^{-\lambda(z-\mu)^2}
\]

![Histogram of z with fitted curve](outputs/fitted_curve.png)

---

## Conclusion
- The NO2 values were successfully transformed into \(z\) using the roll-number-based non-linear transformation.
- The parameters \(\mu\), \(\sigma^2\), \(\lambda\), and \(c\) were estimated using MLE.
- The fitted density curve provides an approximate probabilistic representation of the transformed data distribution.

---

## Files Included in This Repository
- `Assignment1_NO2_PDF.ipynb` → Colab notebook with full implementation
- `outputs/histogram.png` → Histogram of transformed variable \(z\)
- `outputs/fitted_curve.png` → Histogram + fitted density curve
- `outputs/estimated_parameters.csv` → Saved estimated values of parameters

---
