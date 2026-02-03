import sys
import os
import pandas as pd
from topsis_keshava_102303982.topsis import run_topsis, parse_weights_impacts, TopsisError


def main():
    if len(sys.argv) != 5:
        print("Error: Incorrect number of parameters.")
        print("Usage: python -m Topsis_keshava_102303982 <InputDataFile> <Weights> <Impacts> <OutputFile>")
        sys.exit(1)

    input_file, weights_str, impacts_str, output_file = sys.argv[1:5]

    if not os.path.isfile(input_file):
        print("Error: Input file not found.")
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
        n_criteria = df.shape[1] - 1
        weights, impacts = parse_weights_impacts(weights_str, impacts_str, n_criteria)
        result = run_topsis(df, weights, impacts)
        result.to_csv(output_file, index=False)
        print(f"Success: TOPSIS result saved to {output_file}")
    except TopsisError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
