# Assignment: Data Generation using Modelling and Simulation for Machine Learning

## Simulator Chosen
**SimPy** (Python discrete-event simulation library) was used to model a real-world queuing system: **M/M/1/K** queue:
- Exponential arrivals (rate λ)
- Exponential service (rate μ)
- Single server
- Finite system capacity K (drops occur if full)

This models real systems such as:
- bank counter queue
- call center
- API request queue / server backlog
- hospital reception queue

## Methodology

### Step 1–2: Install & Explore
Installed SimPy in Google Colab and implemented the M/M/1/K discrete-event simulation using:
- Arrival process (random exponential inter-arrival time)
- Customer process (wait + service)
- Capacity constraint (drop if system is full)

### Step 3: Parameters & Bounds
Random parameter generation bounds:

| Parameter | Meaning | Lower | Upper |
|---|---|---:|---:|
| arrival_rate (λ) | arrivals/min | 0.2 | 5.0 |
| service_rate (μ) | services/min | 0.3 | 6.0 |
| capacity (K) | max customers in system | 5 | 100 |
| sim_time | minutes | 300 | 2000 |
| warmup_time | ignore transient | 0 | 200 |

### Step 4–5: Dataset Generation
- Generated 1000 random parameter sets
- For each set, ran one simulation
- Recorded outputs:
  - avg_wait_time
  - avg_system_time
  - utilization
  - throughput
  - drop_rate

Saved as: `simpy_mm1k_dataset.csv`

### Step 6: ML Models Comparison
Goal: predict **avg_wait_time** from simulation parameters.

Models compared:
- LinearRegression
- Ridge
- Lasso
- ElasticNet
- KNN
- SVR(RBF)
- RandomForest
- ExtraTrees
- GradientBoosting
- AdaBoost

Evaluation metrics:
- MAE
- RMSE
- R²

Saved results table as: `model_comparison.csv`

## Results
- The best model was selected based on **lowest RMSE**.
- Graphs include:
  - distribution of avg_wait_time
  - arrival_rate vs avg_wait_time scatter
  - model RMSE comparison bar chart

## How to Run
1. Open the notebook in Colab
2. Run all cells
3. Outputs will be generated:
   - simpy_mm1k_dataset.csv
   - model_comparison.csv
   - plots in notebook
