# Assignment-1: Learn Probability Density Functions using Roll-Number-Parameterized Non-Linear Model

## Title
**Learn Probability Density Functions using Roll-Number-Parameterized Non-Linear Model**

## Dataset
- Kaggle Dataset: **India Air Quality Data**
- Feature used: **NO2 (x)**
- Link: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

---

## Objective
1. Transform the NO2 feature values **x** into a new variable **z** using a roll-number-based non-linear transformation.
2. Learn the parameters of the probability density function:

**p̂(z) = c * exp(-λ * (z - μ)^2)**

where the parameters to estimate are: **λ**, **μ**, and **c**.

---

## Roll Number Details

University Roll Number:  
**r = 102303982**

The transformation parameters are defined as:

- **a_r = 0.05 * (r mod 7)**
- **b_r = 0.3 * ((r mod 5) + 1)**

Calculated values:

- r mod 7 = 4  →  a_r = 0.05 * 4 = **0.20**
- r mod 5 = 2  →  b_r = 0.3 * (2 + 1) = **0.90**

So the final transformation becomes:

**z = x + 0.20 * sin(0.90 * x)**

---

## Step 1: Data Preprocessing
- Loaded the dataset CSV file.
- Selected the **NO2** column.
- Converted NO2 values to numeric format.
- Removed missing values (NaN) and invalid entries.
- Final valid sample size:

**n = 419509**

---

## Step 2: Transformation (x → z)
For each NO2 value (x_i), transformed value (z_i) is computed as:

**z_i = x_i + 0.20 * sin(0.90 * x_i)**

---

## Step 3: Learning the PDF Parameters

We are given the PDF model:

**p̂(z) = c * exp(-λ * (z - μ)^2)**

To estimate parameters, we use MLE (Maximum Likelihood Estimation):

### Mean
**μ = (1/n) * Σ z_i**

### Variance (MLE)
**σ² = (1/n) * Σ (z_i - μ)²**

### Lambda
**λ = 1 / (2 * σ²)**

### Constant c (normalization)
**c = sqrt(λ / π)**

---

## Final Estimated Parameters

- **Mean (μ)** = `25.804091267939`
- **Variance (σ²)** = `342.610945104577`
- **Lambda (λ)** = `0.001459381281`
- **Constant (c)** = `0.021553085382`

---

## Results Visualization

### 1) Histogram of Transformed Variable z (Density)
![Transformed variable z histogram](outputs/histogram.png)

### 2) Histogram of z with Fitted PDF Curve
The curve represents:

**p̂(z) = c * exp(-λ * (z - μ)^2)**

![Histogram of z with fitted curve](outputs/fitted_curve.png)

---

## Conclusion
- The NO2 values were transformed into **z** using the roll-number-based non-linear transformation.
- The parameters **μ**, **σ²**, **λ**, and **c** were estimated using MLE.
- The fitted density curve provides an approximate probabilistic representation of the transformed data distribution.

---

## Files Included in This Repository
- `Assignment1_NO2_PDF.ipynb` → Colab notebook with full implementation
- `outputs/histogram.png` → Histogram of transformed variable z
- `outputs/fitted_curve.png` → Histogram + fitted density curve
- `outputs/estimated_parameters.csv` → Saved estimated parameter values
