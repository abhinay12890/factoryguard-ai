from flask import Flask, request, render_template
import joblib
import pandas as pd
import time

# ---------------------------
# Initialize Flask app
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load trained model
# ---------------------------
MODEL_PATH = "/mnt/f/Zaalima Internship/Zaalima Project/factoryguard-ai/output/final_model.pkl"
final_model = joblib.load(MODEL_PATH)

# Exact features used during training
MODEL_FEATURES = final_model.booster_.feature_name()

# ---------------------------
# Home route (UI)
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", form_data={})

# ---------------------------
# Prediction route (HTML form)
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    # Read form values
    data = {k: float(v) for k, v in request.form.items()}
    df = pd.DataFrame([data])

    # Ensure correct feature order
    X = df[MODEL_FEATURES]

    prob = final_model.predict_proba(X)[0][1]

    def categorize_risk(score):
        if score >= 0.85:
            return "CRITICAL"
        elif score >= 0.60:
            return "HIGH"
        elif score >= 0.30:
            return "MEDIUM"
        else:
            return "LOW"

    # Risk decision
    risk = categorize_risk(prob)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return render_template(
        "index.html",
        prediction=round(float(prob), 4),
        risk=risk,
        latency=latency_ms,
        form_data=data
    )


# ---------------------------
# Run Flask app
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False)
