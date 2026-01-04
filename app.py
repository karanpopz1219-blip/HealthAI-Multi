from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import importlib

app = FastAPI(title='HealthAI Models API')

# Import the Streamlit module which initializes models (health_care_bot)
try:
    import health_care_bot
except Exception as e:
    health_care_bot = None

class RiskInput(BaseModel):
    age: int
    bmi: float
    systolic_bp: float
    cholesterol: float
    blood_glucose: float
    diabetes: int
    hypertension: int
    smoker: int
    prev_hospitalizations: int
    gender: str

class LOSInput(RiskInput):
    pass

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": health_care_bot is not None}

@app.post("/predict/risk")
def predict_risk(payload: RiskInput):
    if health_care_bot is None:
        raise HTTPException(status_code=503, detail="Health app module unavailable")
    try:
        m = health_care_bot.module1_models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Build DataFrame
    import pandas as pd
    df = pd.DataFrame([payload.dict()])
    # Preprocess similarly to app
    try:
        new_patient = df.copy()
        new_patient['BP_level'] = pd.cut(new_patient['systolic_bp'], bins=[0,80,120,200], labels=['Low','Normal','High'])
        new_patient['medication_history'] = ((new_patient['diabetes']==1)|(new_patient['hypertension']==1)|(new_patient['smoker']==1)).astype(int)
        new_patient['BP_level'] = m['le_synthetic_BP'].transform(new_patient['BP_level'])
        if m['le_gender'] is not None and new_patient['gender'].dtype == 'object':
            new_patient['gender'] = m['le_gender'].transform(new_patient['gender'])
    except Exception:
        # Best-effort default behavior
        pass
    # Align features
    cols = m['clf_all_feature_names_before_select']
    new_aligned = pd.DataFrame(columns=cols)
    for c in cols:
        new_aligned[c] = new_patient.get(c, 0)
    X_scaled = m['scaler_clf'].transform(new_aligned)
    X_sel = m['selector_clf'].transform(X_scaled)
    pred = m['clf_model'].predict(X_sel)
    label = m['le_synthetic_risk'].inverse_transform(pred)[0]
    return {"predicted_risk": label}

@app.post("/predict/los")
def predict_los(payload: LOSInput):
    if health_care_bot is None:
        raise HTTPException(status_code=503, detail="Health app module unavailable")
    m = health_care_bot.module1_models
    import pandas as pd
    df = pd.DataFrame([payload.dict()])
    # Preprocess minimal
    try:
        df['BP_level'] = pd.cut(df['systolic_bp'], bins=[0,80,120,200], labels=['Low','Normal','High'])
        df['medication_history'] = ((df['diabetes']==1)|(df['hypertension']==1)|(df['smoker']==1)).astype(int)
        df['BP_level'] = m['le_synthetic_BP'].transform(df['BP_level'])
        if m['le_gender'] is not None and df['gender'].dtype == 'object':
            df['gender'] = m['le_gender'].transform(df['gender'])
    except Exception:
        pass
    try:
        X_reg = df[m['reg_feature_names']]
        pred = m['reg_pipeline'].predict(X_reg)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"predicted_los_days": float(pred)}

@app.post("/cluster")
def assign_cluster(payload: RiskInput):
    if health_care_bot is None:
        raise HTTPException(status_code=503, detail="Health app module unavailable")
    m = health_care_bot.module1_models
    import pandas as pd
    df = pd.DataFrame([payload.dict()])
    try:
        df['BP_level'] = pd.cut(df['systolic_bp'], bins=[0,80,120,200], labels=['Low','Normal','High'])
        df['medication_history'] = ((df['diabetes']==1)|(df['hypertension']==1)|(df['smoker']==1)).astype(int)
        df['BP_level'] = m['le_synthetic_BP'].transform(df['BP_level'])
        if m['le_gender'] is not None and df['gender'].dtype == 'object':
            df['gender'] = m['le_gender'].transform(df['gender'])
    except Exception:
        pass
    try:
        X = df[m['cluster_feature_names']]
        Xs = m['scaler_cluster'].transform(X)
        lab = int(m['kmeans_model'].predict(Xs)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"cluster": lab}
