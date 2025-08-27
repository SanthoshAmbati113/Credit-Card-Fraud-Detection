from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get transaction amount
        amount = float(request.form["amount"])

        # Get the features (V1 - V29)
        features_input = request.form["features"]
        features_list = [float(x) for x in features_input.split(",")]

        # Combine into final feature array (Amount + V1..V29)
        final_features = [amount] + features_list
        final_features = np.array(final_features).reshape(1, -1)

        # Sanity check
        if final_features.shape[1] != 30:
            return render_template(
                "index.html",
                prediction_text=f"❌ Error: Expected 30 features, got {final_features.shape[1]}"
            )

        # Predict
        prediction = model.predict(final_features)[0]
        output = "Fraudulent Transaction 🚨" if prediction == 1 else "Legit Transaction ✅"

        return render_template("index.html", prediction_text=f"Prediction: {output}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
