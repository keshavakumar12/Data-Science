import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def topsis(df, criteria, weights, impacts):
    X = df[criteria].astype(float).to_numpy()

    
    norm = np.sqrt((X**2).sum(axis=0))
    R = X / norm

    
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    V = R * w

    
    ideal_best = np.zeros(V.shape[1])
    ideal_worst = np.zeros(V.shape[1])

    for j, imp in enumerate(impacts):
        if imp == "+":
            ideal_best[j] = V[:, j].max()
            ideal_worst[j] = V[:, j].min()
        else:
            ideal_best[j] = V[:, j].min()
            ideal_worst[j] = V[:, j].max()

    
    d_pos = np.sqrt(((V - ideal_best)**2).sum(axis=1))
    d_neg = np.sqrt(((V - ideal_worst)**2).sum(axis=1))

    score = d_neg / (d_pos + d_neg)

    out = df.copy()
    out["topsis_score"] = score
    out["rank"] = out["topsis_score"].rank(ascending=False, method="dense").astype(int)
    out = out.sort_values(["rank", "topsis_score"], ascending=[True, False])
    return out

def parse_list(s, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--criteria", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--impacts", required=True)
    ap.add_argument("--out_csv", default="outputs/topsis_ranking.csv")
    ap.add_argument("--out_plot", default="outputs/topsis_scores.png")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    criteria = [x.strip() for x in args.criteria.split(",")]
    weights = parse_list(args.weights, float)
    impacts = [x.strip() for x in args.impacts.split(",")]

    if not (len(criteria) == len(weights) == len(impacts)):
        raise ValueError("criteria, weights, impacts must have same length")

    ranked = topsis(df, criteria, weights, impacts)

    os.makedirs("outputs", exist_ok=True)
    ranked.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(ranked["model"], ranked["topsis_score"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("TOPSIS Score")
    plt.title("TOPSIS Scores (Text Classification)")
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    plt.close()

    print("Saved:", args.out_csv)
    print("Saved:", args.out_plot)
    print("Best model:", ranked.iloc[0]["model"], "Score:", ranked.iloc[0]["topsis_score"])

if __name__ == "__main__":
    main()

