from flask import Flask, request, jsonify
import joblib
from utils.predict import make_prediction
from utils.explain_utils import SHAPExplainer
from utils.preprocess_utils import clean_input
from utils.logger_utils import setup_logger

app = Flask(__name__)
logger = setup_logger()

# Load model and scaler once
model = joblib.load("C:/Users/somas/PycharmProjects/ADHD_PREDICTION_MODEL/model/models/adhd_xgb_model.pkl")
scaler = joblib.load("C:/Users/somas/PycharmProjects/ADHD_PREDICTION_MODEL/model/models/adhd_scaler.pkl")
explainer = SHAPExplainer(model, scaler)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        logger.info("Received input: %s", data)

        data = clean_input(data)
        pred, prob, X_user, X_scaled = make_prediction(model, scaler, data)
        shap_vals = explainer.get_shap_values(X_scaled)
        summary_img = explainer.plot_summary(shap_vals, X_user)
        force_img = explainer.plot_force(shap_vals, X_user)

        return jsonify({
            "prediction": pred,
            "probability": prob,
            "shap_summary_img": summary_img,
            "shap_force_img": force_img
        })

    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)