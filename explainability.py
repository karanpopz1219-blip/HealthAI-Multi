"""
Explainability helpers and examples using SHAP and LIME.

This script demonstrates how to run SHAP explanations for a scikit-learn
classifier (RandomForest) on a small synthetic dataset. It is intended as
an example to adapt against the real `module1` trained model.

Notes:
- SHAP and LIME are optional heavy packages. Install them only when needed:
  `pip install shap lime`
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None


def train_demo_model(random_state=42):
    # Create a small synthetic dataset
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame({
        'age': rng.randint(20, 80, size=500),
        'bmi': rng.uniform(18, 40, size=500),
        'systolic_bp': rng.randint(90, 180, size=500),
        'cholesterol': rng.randint(150, 280, size=500),
        'smoker': rng.randint(0, 2, size=500),
    })
    y = ((X['age'] > 60).astype(int) | (X['bmi'] > 30).astype(int) | (X['systolic_bp'] > 140).astype(int)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=50, random_state=random_state)
    clf.fit(X_train_s, y_train)

    return clf, scaler, X_train, X_test, X_train_s, X_test_s


def demo_shap():
    clf, scaler, X_train, X_test, X_train_s, X_test_s = train_demo_model()
    if shap is None:
        print("SHAP not installed. Install with `pip install shap` to run this demo.")
        return

    explainer = shap.TreeExplainer(clf)
    # Explain a small sample
    sample = X_test_s[:5]
    shap_values = explainer.shap_values(sample)
    print("SHAP values computed. Use shap.summary_plot to visualize in a notebook.")


def demo_lime():
    clf, scaler, X_train, X_test, X_train_s, X_test_s = train_demo_model()
    if LimeTabularExplainer is None:
        print("LIME not installed. Install with `pip install lime` to run this demo.")
        return

    explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), class_names=["low","high"], discretize_continuous=True)
    exp = explainer.explain_instance(X_test.values[0], lambda x: clf.predict_proba(scaler.transform(x)), num_features=4)
    print(exp.as_list())


if __name__ == '__main__':
    print("Explainability demo")
    demo_shap()
    demo_lime()
