import sys
import os
import pandas as pd
import numpy as np


def _error_exit(msg: str):
    print(f"Error: {msg}")
    sys.exit(1)


def parse_weights_impacts(weights_str: str, impacts_str: str, n_criteria: int):
    # weights: "1,2,3"
    try:
        weights = [float(x.strip()) for x in weights_str.split(",")]
    except Exception:
        _error_exit("Weights must be numeric values separated by commas (e.g., \"1,2,3\").")

    impacts = [x.strip() for x in impacts_str.split(",")]

    if len(weights) != n_criteria or len(impacts) != n_criteria:
        _error_exit("Number of weights, impacts, and criteria columns (from 2nd to last) must be the same.")

    for imp in impacts:
        if imp not in ["+", "-"]:
            _error_exit("Impacts must be either '+' or '-' separated by commas (e.g., \"+,-,+\").")

    if any(w <= 0 for w in weights):
        _error_exit("Weights must be positive numbers.")

    weights = np.array(weights, dtype=float)
    # normalize weights (common practice; safe if user gives any scale)
    weights = weights / weights.sum()
    return weights, impacts


def topsis(df: pd.DataFrame, weights: np.ndarray, impacts: list[str]) -> pd.DataFrame:
    if df.shape[1] < 3:
        _error_exit("Input file must contain three or more columns (1st as name/id, rest criteria).")

    names = df.iloc[:, 0]
    criteria_df = df.iloc[:, 1:].copy()

    # Validate numeric columns from 2nd to last
    for col in criteria_df.columns:
        criteria_df[col] = pd.to_numeric(criteria_df[col], errors="coerce")

    if criteria_df.isnull().any().any():
        bad_cols = criteria_df.columns[criteria_df.isnull().any()].tolist()
        _error_exit(f"Non-numeric values found in criteria columns: {bad_cols}")

    X = criteria_df.to_numpy(dtype=float)

    # Step 1: normalization
    denom = np.sqrt((X ** 2).sum(axis=0))
    if np.any(denom == 0):
        _error_exit("At least one criteria column has all zeros, cannot normalize.")
    R = X / denom

    # Step 2: weighted normalized matrix
    V = R * weights

    # Step 3: ideal best and worst
    ideal_best = np.zeros(V.shape[1])
    ideal_worst = np.zeros(V.shape[1])

    for j, imp in enumerate(impacts):
        col = V[:, j]
        if imp == "+":
            ideal_best[j] = np.max(col)
            ideal_worst[j] = np.min(col)
        else:  # "-"
            ideal_best[j] = np.min(col)
            ideal_worst[j] = np.max(col)

    # Step 4: distances
    s_pos = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    s_neg = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    # Step 5: score
    denom2 = s_pos + s_neg
    if np.any(denom2 == 0):
        _error_exit("Division by zero encountered in score computation (check input data).")
    score = s_neg / denom2

    out = df.copy()
    out["Topsis Score"] = score

    # Step 6: rank (descending score)
    out["Rank"] = out["Topsis Score"].rank(ascending=False, method="dense").astype(int)
    return out


def main():
    # Requirement: correct number of parameters
    if len(sys.argv) != 5:
        _error_exit(
            "Incorrect number of parameters.\n"
            "Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <OutputFile>"
        )

    input_file = sys.argv[1]
    weights_str = sys.argv[2]
    impacts_str = sys.argv[3]
    output_file = sys.argv[4]

    # File not found handling
    if not os.path.isfile(input_file):
        _error_exit("Input file not found.")

    # Read CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        _error_exit(f"Unable to read input file as CSV: {e}")

    n_criteria = df.shape[1] - 1
    if n_criteria < 2:
        _error_exit("Input file must contain at least 2 criteria columns (total columns >= 3).")

    weights, impacts = parse_weights_impacts(weights_str, impacts_str, n_criteria)
    result = topsis(df, weights, impacts)

    try:
        result.to_csv(output_file, index=False)
    except Exception as e:
        _error_exit(f"Unable to write output file: {e}")

    print(f"Success: TOPSIS result saved to {output_file}")


if __name__ == "__main__":
    main()
