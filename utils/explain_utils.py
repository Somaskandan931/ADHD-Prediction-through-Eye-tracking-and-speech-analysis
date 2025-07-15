# explain_utils.py

import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class SHAPExplainer:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.explainer = shap.Explainer(self.model)

    def prepare_data(self, X):
        return self.scaler.transform(X)

    def get_shap_values(self, X_scaled):
        return self.explainer(X_scaled)

    def plot_summary(self, shap_values, X):
        plt.figure()
        shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def plot_force(self, shap_values, X, sample_index=0):
        shap.force_plot(shap_values[sample_index], matplotlib=True, show=False)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
