from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score



MODELS = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/bert-base-uncased-SST-2",
    "siebert/sentiment-roberta-large-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "nlptown/bert-base-multilingual-uncased-sentiment",
]

DATASET = ("glue", "sst2")   
SPLIT = "validation"
MAX_SAMPLES = 500           
BATCH_SIZE = 16             
LATENCY_SAMPLES = 200       



def run_in_batches(pipe, texts: list[str], batch_size: int = 16):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs.extend(pipe(batch, truncation=True))
    return outputs


def label_to_binary(label: str) -> int:

    lab = str(label).upper().strip()

    
    if "LABEL_1" in lab or lab.endswith("_1"):
        return 1
    if "LABEL_0" in lab or lab.endswith("_0"):
        return 0

    
    if "POS" in lab:
        return 1
    if "NEG" in lab:
        return 0

    
    digits = [ch for ch in lab if ch.isdigit()]
    if digits:
        try:
            star = int(digits[0])
            return 1 if star >= 3 else 0
        except Exception:
            pass

 
    return 0


def estimate_latency_ms(pipe, texts: list[str], batch_size: int = 16) -> float:
   
    start = time.time()
    _ = run_in_batches(pipe, texts, batch_size=batch_size)
    end = time.time()
    total_sec = end - start
    return (total_sec / max(len(texts), 1)) * 1000.0


def main():
    os.makedirs("outputs", exist_ok=True)

    
    ds = load_dataset(*DATASET)[SPLIT]
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

    
    texts = [str(x) for x in ds["sentence"] if x is not None]
    y_true = [int(x) for x in ds["label"]][:len(texts)]
    texts = texts[:len(y_true)]

    rows = []

    for m in MODELS:
        print(f"Evaluating: {m}")

        
        clf = pipeline(
            "text-classification",
            model=m,
            tokenizer=m,
            device=-1,
            framework="pt",
        )

  
        preds = run_in_batches(clf, texts, batch_size=BATCH_SIZE)
        y_pred = [label_to_binary(p["label"]) for p in preds]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        
        lat_texts = texts[:min(LATENCY_SAMPLES, len(texts))]
        latency_ms = estimate_latency_ms(clf, lat_texts, batch_size=BATCH_SIZE)

        
        params_millions = np.nan
        vram_gb = np.nan

        rows.append([m, acc, f1, latency_ms, params_millions, vram_gb])

    out = pd.DataFrame(
        rows,
        columns=["model", "accuracy", "f1", "latency_ms", "params_millions", "vram_gb"],
    )

    out_path = "outputs/evaluation_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("Next: copy this file to data/results.csv and fill params_millions + vram_gb manually.")


if __name__ == "__main__":
    main()

