# predict_utils.py

import pandas as pd

def make_prediction(model, scaler, input_data: dict):
    """
    Takes model, scaler, and raw input dict. Returns prediction and probability.
    """
    X_user = pd.DataFrame([input_data])
    X_scaled = scaler.transform(X_user)

    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])

    return prediction, probability, X_user, X_scaled
