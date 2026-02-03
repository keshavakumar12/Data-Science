import os
import re
import uuid
import smtplib
from email.message import EmailMessage

import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"


class TopsisError(Exception):
    pass


def parse_weights_impacts(weights_str: str, impacts_str: str, n_criteria: int):
    try:
        weights = [float(x.strip()) for x in weights_str.split(",")]
    except Exception:
        raise TopsisError("Weights must be numeric values separated by commas.")

    impacts = [x.strip() for x in impacts_str.split(",")]

    if len(weights) != n_criteria or len(impacts) != n_criteria:
        raise TopsisError("Number of weights must be equal to number of impacts and criteria columns.")

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


def send_email_with_attachment(to_email: str, file_path: str):
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    from_email = os.environ.get("FROM_EMAIL", smtp_user)

    if not all([smtp_host, smtp_user, smtp_pass, from_email]):
        raise TopsisError("SMTP is not configured on server. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, FROM_EMAIL.")

    msg = EmailMessage()
    msg["Subject"] = "TOPSIS Result File"
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content("Attached is your TOPSIS output CSV.")

    with open(file_path, "rb") as f:
        data = f.read()

    filename = os.path.basename(file_path)
    msg.add_attachment(data, maintype="text", subtype="csv", filename=filename)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    file = request.files.get("file")
    weights_str = request.form.get("weights", "")
    impacts_str = request.form.get("impacts", "")
    email = request.form.get("email", "")

    if not file or file.filename == "":
        return "Error: No file uploaded.", 400

    if not re.match(EMAIL_REGEX, email):
        return "Error: Invalid email format.", 400

    file_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    file.save(input_path)

    try:
        df = pd.read_csv(input_path)
        n_criteria = df.shape[1] - 1
        weights, impacts = parse_weights_impacts(weights_str, impacts_str, n_criteria)
        result = run_topsis(df, weights, impacts)

        output_path = os.path.join(OUTPUT_DIR, f"topsis_result_{file_id}.csv")
        result.to_csv(output_path, index=False)

        send_email_with_attachment(email, output_path)

        return "Success: Result generated and emailed to you."
    except TopsisError as e:
        return f"Error: {e}", 400
    except Exception as e:
        return f"Error: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)

