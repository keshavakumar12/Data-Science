import pandas as pd
import numpy as np

class TopsisError(Exception):
    pass

def parse_weights_impacts(weights_str: str, impacts_str: str, n_criteria: int):
    try:
        weights = [float(x.strip()) for x in weights_str.split(",")]
    except Exception:
        raise TopsisError("Weights must be numeric values separated by commas.")

    impacts = [x.strip() for x in impacts_str.split(",")]

    if len(weights) != n_criteria or len(impacts) != n_criteria:
        raise TopsisError("Number of weights, impacts, and criteria columns must be the same.")

    for imp in impacts:
        if imp not in ["+", "-"]:
            raise TopsisError("Impacts must be either '+' or '-' separated by commas.")

    if any(w <= 0 for w in weights):
        raise TopsisError("Weights must be positive numbers.")

    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return weights, impacts

def run_topsis(df: pd.DataFrame, weights: np.ndarray, impacts: list[str]) -> pd.DataFrame:
    if df.shape[1] < 3:
        raise TopsisError("Input file must contain three or more columns.")

    criteria_df = df.iloc[:, 1:].copy()

    for col in criteria_df.columns:
        criteria_df[col] = pd.to_numeric(criteria_df[col], errors="coerce")

    if criteria_df.isnull().any().any():
        bad_cols = criteria_df.columns[criteria_df.isnull().any()].tolist()
        raise TopsisError(f"Non-numeric values found in criteria columns: {bad_cols}")

    X = criteria_df.to_numpy(dtype=float)

    denom = np.sqrt((X ** 2).sum(axis=0))
    if np.any(denom == 0):
        raise TopsisError("At least one criteria column has all zeros.")

    R = X / denom
    V = R * weights

    ideal_best = np.zeros(V.shape[1])
    ideal_worst = np.zeros(V.shape[1])

    for j, imp in enumerate(impacts):
        col = V[:, j]
        if imp == "+":
            ideal_best[j] = np.max(col)
            ideal_worst[j] = np.min(col)
        else:
            ideal_best[j] = np.min(col)
            ideal_worst[j] = np.max(col)

    s_pos = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    s_neg = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    denom2 = s_pos + s_neg
    if np.any(denom2 == 0):
        raise TopsisError("Division by zero in score computation.")

    score = s_neg / denom2

    out = df.copy()
    out["Topsis Score"] = score
    out["Rank"] = out["Topsis Score"].rank(ascending=False, method="dense").astype(int)
    return out
