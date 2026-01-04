# HealthAI Dashboard & API

This repository contains:

- `health_care_bot.py` — Streamlit dashboard integrating multiple modules (analytics, association rules, LSTM, BERT, generative models, chatbot/translator).
- `healthai/` — helper utilities (persistence helpers).
- `backend/app.py` — FastAPI backend exposing prediction endpoints that use the models initialized by the Streamlit app.
- `backend/test_api.py` — simple pytest tests for the API.
- `requirements.txt` — Python dependencies (pinned).
- `saved_models/` — directory where trained models will be persisted.

Getting started

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Streamlit dashboard:

```bash
streamlit run health_care_bot.py
```

3. Run the FastAPI backend (recommended to run separately):

```bash
uvicorn backend.app:app --reload --port 8000
```

4. Run API tests:

```bash
pytest backend/test_api.py -q
```

Model persistence

You can save models from the running Streamlit module using `healthai.persistence.save_all_models_from_streamlit_module(health_care_bot)`.

Notes

- Some features require optional heavy ML packages (PyTorch, Transformers, TensorFlow). The app detects missing packages and shows warnings. To suppress the in-app yellow warning banners set `HEALTHAI_SUPPRESS_WARNINGS=1` in your environment.
- If you experience TensorFlow crashes on macOS, consider using a Conda environment with a compatible TensorFlow build.

Docker / Containerization
------------------------

You can run the app in Docker for an isolated environment. Build and run with:

```bash
docker build -t healthai-app .
docker run -p 8501:8501 healthai-app
```

Or use docker-compose to run the Streamlit UI and FastAPI backend:

```bash
docker-compose up --build
```

Model interpretability (SHAP & LIME)
-----------------------------------

An example explainability script is available at `healthai/explainability.py`. It demonstrates using SHAP and LIME on a toy RandomForest trained on synthetic data. SHAP and LIME are optional; install them with `pip install shap lime` and run:

```bash
python3 healthai/explainability.py
```

Ethical AI in healthcare
------------------------

See `docs/ethical_ai.md` for a short checklist and guidance when developing models for healthcare. Follow those steps before releasing models for clinical use.
