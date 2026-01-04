import os
import joblib

# Try to import optional libraries
try:
    import torch
    TORCH = True
except Exception:
    torch = None
    TORCH = False

try:
    from tensorflow.keras.models import save_model as tf_save_model, load_model as tf_load_model
    TFK = True
except Exception:
    tf_save_model = None
    tf_load_model = None
    TFK = False


SAVED_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR, exist_ok=True)


def save_sklearn_model(obj, name: str):
    path = os.path.join(SAVED_DIR, f"{name}.joblib")
    joblib.dump(obj, path)
    return path


def load_sklearn_model(name: str):
    path = os.path.join(SAVED_DIR, f"{name}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def save_torch_model(model, name: str):
    if not TORCH:
        raise RuntimeError("PyTorch not available")
    path = os.path.join(SAVED_DIR, f"{name}_torch.pt")
    torch.save(model.state_dict(), path)
    return path


def save_tf_model(model, name: str):
    if not TFK:
        raise RuntimeError("TensorFlow/Keras not available")
    path = os.path.join(SAVED_DIR, name)
    tf_save_model(model, path, overwrite=True, include_optimizer=False)
    return path


def save_all_models_from_streamlit_module(streamlit_module):
    """Save models present in the running Streamlit module (health_care_bot) to disk.
    This inspects expected keys and saves if available.
    """
    saved = {}
    # sklearn models from module1
    try:
        m1 = getattr(streamlit_module, 'module1_models', None)
        if m1:
            if m1.get('clf_model') is not None:
                saved['clf_model'] = save_sklearn_model(m1['clf_model'], 'clf_model')
            if m1.get('reg_pipeline') is not None:
                saved['reg_pipeline'] = save_sklearn_model(m1['reg_pipeline'], 'reg_pipeline')
            if m1.get('kmeans_model') is not None:
                saved['kmeans_model'] = save_sklearn_model(m1['kmeans_model'], 'kmeans_model')
    except Exception:
        pass

    # torch / transformers models from module4 and 5
    try:
        m4 = getattr(streamlit_module, 'module4_models', None)
        if m4 and m4.get('model_sentiment') is not None and TORCH:
            try:
                save_torch_model(m4['model_sentiment'], 'model_sentiment')
                saved['model_sentiment'] = 'saved'
            except Exception:
                pass
    except Exception:
        pass

    try:
        m5 = getattr(streamlit_module, 'module5_models', None)
        if m5 and m5.get('model_biobert_embeddings') is not None and TORCH:
            try:
                save_torch_model(m5['model_biobert_embeddings'], 'model_biobert_embeddings')
                saved['model_biobert_embeddings'] = 'saved'
            except Exception:
                pass
    except Exception:
        pass

    # Keras LSTM
    try:
        m3 = getattr(streamlit_module, 'module3_models', None)
        if m3 and m3.get('model_lstm') is not None and TFK:
            try:
                save_tf_model(m3['model_lstm'], 'model_lstm')
                saved['model_lstm'] = 'saved'
            except Exception:
                pass
    except Exception:
        pass

    return saved
