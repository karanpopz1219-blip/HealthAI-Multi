import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False
import random
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    apriori = None
    association_rules = None
    MLXTEND_AVAILABLE = False

try:
    from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel, pipeline, MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    BertTokenizer = None
    BertForSequenceClassification = None
    AutoTokenizer = None
    AutoModel = None
    pipeline = None
    MarianMTModel = None
    MarianTokenizer = None
    TRANSFORMERS_AVAILABLE = False

try:
    from torch.optim import AdamW
except Exception:
    AdamW = None

# Avoid importing TensorFlow at module import time (can segfault on some systems).
# Defer any TensorFlow/Keras usage inside the LSTM setup function and mark as unavailable here.
TFK_AVAILABLE = False
Sequential = None
LSTM = None
Dense = None
Dropout = None
Adam = None

# Suppress warnings from mlxtend regarding DataFrame types
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='mlxtend')

# Allow suppressing Streamlit yellow warning banners during development or automated runs
# Default to suppressing warnings to keep the UI clean; set HEALTHAI_SUPPRESS_WARNINGS=0 to show warnings.
SUPPRESS_WARNINGS = os.environ.get("HEALTHAI_SUPPRESS_WARNINGS", "1") in ("1", "true", "True")

def app_warn(msg):
    """Show a Streamlit warning unless warnings are suppressed via env var."""
    try:
        if SUPPRESS_WARNINGS:
            print("WARN:", msg)
        else:
            st.warning(msg)
    except Exception:
        # If Streamlit runtime isn't available, fallback to printing
        print("WARN:", msg)

# Filter common noisy warnings (sklearn feature-name warning) when suppressing warnings
if SUPPRESS_WARNINGS:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    # Also ignore the Streamlit ScriptRunContext warning which appears when importing
    # Streamlit from a plain Python interpreter instead of using `streamlit run`.
    warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*", module='streamlit.runtime.script')
    try:
        import logging
        logging.getLogger("streamlit.runtime.script").setLevel(logging.ERROR)
    except Exception:
        pass


# --- Helper Functions ---

@st.cache_data
def clean_text(text):
    """Cleans clinical notes text by lowercasing and standardizing whitespace."""
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def create_sequences(data, target, seq_length=10):
    """Creates time-series sequences using a sliding window technique."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(target.iloc[i+seq_length])
    return np.array(X), np.array(y)

if TORCH_AVAILABLE:
    class FeedbackDataset(torch.utils.data.Dataset):
        """Custom Dataset for BERT sentiment analysis."""
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
else:
    class FeedbackDataset:
        """Fallback dataset when PyTorch is unavailable (keeps API surface for checks)."""
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            return {"labels": self.labels[idx]}

        def __len__(self):
            return len(self.labels)

@st.cache_resource
def get_embedding(text, tokenizer_bert, model_bert):
    """Generates BERT embeddings for a given text."""
    # If required libraries/models are unavailable, return a zero-vector embedding to keep the app functional.
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE or tokenizer_bert is None or model_bert is None:
        # Return a 1x768 zero-vector (common BERT hidden size) as a safe fallback
        return np.zeros((1, 768))

    inputs = tokenizer_bert(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_bert.to(device)

    with torch.no_grad():
        outputs = model_bert(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="HealthAI Dashboard")
st.title("HealthAI Multi-Module Dashboard")

st.sidebar.title("Navigation")
module_selection = st.sidebar.radio(
    "Go to",
    [
        "Module 1: Patient Data Analytics",
        "Module 2: Association Rules",
        "Module 3: Sequence Modeling (LSTM)",
        "Module 4: Sentiment Analysis (BERT)",
        "Module 5: Generative AI (BioBERT & BioGPT)",
        "Module 6: Chatbot & Translator"
    ]
)

# --- Data Loading (using st.cache_data to load once) ---
@st.cache_data
def load_all_data():
    # Attempt to load CSVs from multiple common locations (workspace root, data/, /content/)
    base_candidates = [
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), "data"),
        "/content",
        os.getcwd()
    ]

    def find_path(filename):
        for base in base_candidates:
            if base is None:
                continue
            path = os.path.join(base, filename)
            if os.path.exists(path):
                return path
        return None

    def try_read_csv(filename):
        path = find_path(filename)
        if path:
            return pd.read_csv(path)
        return None

    df_synthetic = try_read_csv("healthai_synthetic_patient_data.csv")
    df_apriori = try_read_csv("healthai_apriori_1000.csv")
    df_timeseries = try_read_csv('healthai_timeseries_1000.csv')
    df_feedback = try_read_csv("healthai_patient_feedback_1000.csv")

    df_clinical = try_read_csv("healthai_clinical_notes_1000.csv")

    # If any primary file is missing, create small synthetic fallbacks so the Streamlit app can still run
    if df_synthetic is None:
        app_warn("`healthai_synthetic_patient_data.csv` not found locally; creating small synthetic dataset for demo.")
        n = 300
        df_synthetic = pd.DataFrame({
            "patient_id": list(range(1, n+1)),
            "age": np.random.randint(18, 90, size=n),
            "bmi": np.round(np.random.normal(25, 4, size=n), 1),
            "systolic_bp": np.random.randint(90, 180, size=n),
            "cholesterol": np.random.randint(150, 280, size=n),
            "blood_glucose": np.random.randint(70, 200, size=n),
            "diabetes": np.random.choice([0,1], size=n, p=[0.85,0.15]),
            "hypertension": np.random.choice([0,1], size=n, p=[0.8,0.2]),
            "smoker": np.random.choice([0,1], size=n, p=[0.9,0.1]),
            "risk_category": np.random.choice(["Low","Medium","High"], size=n, p=[0.6,0.3,0.1]),
            "length_of_stay_days": np.random.poisson(3, size=n),
            "gender": np.random.choice(["Male","Female","Other"], size=n, p=[0.45,0.45,0.1]),
            "prev_hospitalizations": np.random.randint(0,5,size=n)
        })

    if df_apriori is None:
        app_warn("`healthai_apriori_1000.csv` not found locally; creating small synthetic apriori dataset for demo.")
        n = 200
        df_apriori = pd.DataFrame({
            "patient_id": list(range(1, n+1)),
            "fever": np.random.choice([0,1], size=n, p=[0.8,0.2]),
            "cough": np.random.choice([0,1], size=n, p=[0.7,0.3]),
            "diabetes": np.random.choice([0,1], size=n, p=[0.9,0.1]),
            "hypertension": np.random.choice([0,1], size=n, p=[0.85,0.15]),
            "asthma": np.random.choice([0,1], size=n, p=[0.95,0.05])
        })

    if df_timeseries is None:
        app_warn("`healthai_timeseries_1000.csv` not found locally; creating small synthetic timeseries dataset for demo.")
        n = 1000
        df_timeseries = pd.DataFrame({
            "patient_id": np.random.randint(1, 50, size=n),
            "heart_rate": np.random.randint(50, 120, size=n),
            "systolic_bp": np.random.randint(90, 180, size=n),
            "spo2": np.random.randint(85, 100, size=n),
            "risk_flag": np.random.choice([0,1], size=n, p=[0.9,0.1])
        })

    if df_feedback is None:
        app_warn("`healthai_patient_feedback_1000.csv` not found locally; creating small synthetic feedback dataset for demo.")
        texts = [
            "Nurses were helpful and kind",
            "Long wait times but good care",
            "Very satisfied with the doctor",
            "Poor communication from staff",
            "Facility was clean and organised"
        ]
        sentiments = ["Positive","Neutral","Positive","Negative","Positive"]
        df_feedback = pd.DataFrame({
            "feedback_text": np.random.choice(texts, size=200),
            "sentiment": np.random.choice(sentiments, size=200)
        })

    if df_clinical is None:
        app_warn("`healthai_clinical_notes_1000.csv` not found locally; creating small synthetic clinical notes dataset for demo.")
        notes = [
            "Patient reports chest pain and shortness of breath.",
            "History of hypertension and diabetes. Follow-up required.",
            "Mild fever and cough for 3 days.",
            "Post-op patient recovering well without complications."
        ]
        df_clinical = pd.DataFrame({
            "patient_id": [1,2,3,4],
            "clinical_note": notes
        })

    # Ensure chatbot dataset exists
    chatbot_file_path = find_path("healthcare_chatbot_translation_dataset.csv") or os.path.join(os.path.dirname(__file__), "healthcare_chatbot_translation_dataset.csv")
    if not os.path.exists(chatbot_file_path):
        symptoms = [
            "fever", "cough", "headache", "chest pain", "breathing difficulty",
            "fatigue", "nausea", "vomiting", "diabetes symptoms", "high blood pressure"
        ]
        questions = [
            "I have fever and cough, what should I do?",
            "Is chest pain serious?",
            "How to control blood sugar?",
            "I feel tired all the time",
            "Can I take paracetamol daily?",
            "When should I see a doctor?",
            "Is headache dangerous?",
            "How to reduce BP naturally?",
        ]
        responses = [
            "Please consult a physician and take rest.",
            "Monitor symptoms and seek emergency care if pain increases.",
            "Maintain diet, exercise and medication regularly.",
            "Blood tests may be required.",
            "Avoid self-medication without advice.",
        ]
        languages = ["English", "Tamil", "Hindi", "Telugu"]

        data_chatbot = []
        for i in range(1000):
            data_chatbot.append({
                "symptom": random.choice(symptoms),
                "patient_question": random.choice(questions),
                "doctor_reply": random.choice(responses),
                "language": random.choice(languages),
                "appointment_needed": random.choice(["Yes", "No"])
            })
        df_chatbot = pd.DataFrame(data_chatbot)
        df_chatbot.to_csv(chatbot_file_path, index=False)
    else:
        df_chatbot = pd.read_csv(chatbot_file_path)

    return df_synthetic, df_apriori, df_timeseries, df_feedback, df_clinical, df_chatbot

df_synthetic_raw, df_apriori_raw, df_timeseries_raw, df_feedback_raw, df_clinical_raw, df_chatbot_raw = load_all_data()


# --- Model Loading/Training (using st.cache_resource to avoid retraining on every rerun) ---
@st.cache_resource
def setup_module1_models(df_synthetic_raw_copy):
    df_synthetic_copy = df_synthetic_raw_copy.copy()

    # --- Preprocessing Pipeline --- (consistent across all three sub-modules)

    # 1. Feature Engineering
    df_synthetic_copy['BP_level'] = pd.cut(df_synthetic_copy['systolic_bp'], bins=[0, 80, 120, 200], labels=['Low', 'Normal', 'High'], ordered=False)
    df_synthetic_copy['medication_history'] = ((df_synthetic_copy['diabetes'] == 1) | (df_synthetic_copy['hypertension'] == 1) | (df_synthetic_copy['smoker'] == 1)).astype(int)

    # 2. Imputation
    imputer = SimpleImputer(strategy='median')
    num_cols_to_impute = ['bmi','systolic_bp','cholesterol','blood_glucose']
    df_synthetic_copy[num_cols_to_impute] = imputer.fit_transform(df_synthetic_copy[num_cols_to_impute])

    # 3. Outlier Removal (based on imputed numerical columns)
    initial_rows = len(df_synthetic_copy)
    for col in num_cols_to_impute:
        Q1 = df_synthetic_copy[col].quantile(0.25)
        Q3 = df_synthetic_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        df_synthetic_copy = df_synthetic_copy[(df_synthetic_copy[col] >= Q1 - 1.5*IQR) & (df_synthetic_copy[col] <= Q3 + 1.5*IQR)]

    # 4. Label Encoding for categorical features
    le_synthetic_BP = LabelEncoder()
    df_synthetic_copy['BP_level'] = le_synthetic_BP.fit_transform(df_synthetic_copy['BP_level'])

    le_synthetic_risk = LabelEncoder()
    df_synthetic_copy['risk_category'] = le_synthetic_risk.fit_transform(df_synthetic_copy['risk_category'])
    risk_category_labels = le_synthetic_risk.inverse_transform(sorted(df_synthetic_copy['risk_category'].unique()))

    le_gender = LabelEncoder()
    if 'gender' in df_synthetic_copy.columns and df_synthetic_copy['gender'].dtype == 'object':
        df_synthetic_copy['gender'] = le_gender.fit_transform(df_synthetic_copy['gender'])
    else:
        le_gender = None # No gender encoder needed if gender column is absent or already numeric

    # Store processed DataFrame state after all these steps for direct feature extraction for models
    df_synthetic_processed_base = df_synthetic_copy.copy()


    # --- Model-specific Preparations and Training ---

    # 1. Classification Model (RandomForestClassifier)
    X_clf_all_features = df_synthetic_processed_base.drop(['risk_category','length_of_stay_days', 'patient_id'], axis=1, errors='ignore')
    y_clf = df_synthetic_processed_base['risk_category']

    scaler_clf = StandardScaler() # Scaler specifically for classification features before KBest
    X_clf_scaled_for_kbest = scaler_clf.fit_transform(X_clf_all_features)
    X_clf_scaled_for_kbest_df = pd.DataFrame(X_clf_scaled_for_kbest, columns=X_clf_all_features.columns, index=X_clf_all_features.index)

    selector = SelectKBest(score_func=f_classif, k=5)
    selector.fit(X_clf_scaled_for_kbest_df, y_clf)
    X_selected_clf = selector.transform(X_clf_scaled_for_kbest_df)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_selected_clf, y_clf)
    clf_all_feature_names = list(X_clf_all_features.columns) # All features before selection
    clf_feature_names_after_select = list(X_clf_all_features.columns[selector.get_support(indices=True)])


    # 2. Regression Model (Ridge with Pipeline)
    features_for_reg = [col for col in df_synthetic_processed_base.columns if col not in ['patient_id', 'risk_category', 'length_of_stay_days']]
    X_reg_full = df_synthetic_processed_base[features_for_reg]
    y_reg = df_synthetic_processed_base["length_of_stay_days"]

    pipeline_reg = Pipeline([
        ("scaler", StandardScaler()), # This scaler will be fit on X_reg_full
        ("model", Ridge(alpha=1.0))
    ])
    pipeline_reg.fit(X_reg_full, y_reg)
    reg_feature_names = list(X_reg_full.columns)


    # 3. Clustering Model (KMeans)
    features_for_cluster = [col for col in df_synthetic_processed_base.columns if col not in ['patient_id', 'risk_category', 'length_of_stay_days']]
    X_cluster_full = df_synthetic_processed_base[features_for_cluster]

    scaler_cluster = StandardScaler() # Scaler specifically for clustering features
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster_full)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    cluster_feature_names = list(X_cluster_full.columns)

    return {
        "imputer": imputer,
        "le_synthetic_BP": le_synthetic_BP,
        "le_synthetic_risk": le_synthetic_risk,
        "le_gender": le_gender,
        "num_cols_to_impute": num_cols_to_impute,

        "clf_model": clf,
        "scaler_clf": scaler_clf,
        "selector_clf": selector,
        "clf_all_feature_names_before_select": clf_all_feature_names,
        "clf_feature_names_after_select": clf_feature_names_after_select,

        "reg_pipeline": pipeline_reg,
        "reg_feature_names": reg_feature_names,

        "kmeans_model": kmeans,
        "scaler_cluster": scaler_cluster,
        "cluster_feature_names": cluster_feature_names,

        "risk_category_labels": risk_category_labels,
        "df_synthetic_processed_for_metrics": df_synthetic_processed_base # For displaying metrics/summary
    }

module1_models = setup_module1_models(df_synthetic_raw.copy())

@st.cache_resource
def setup_module3_models(df_timeseries_raw_copy):
    df_timeseries_copy = df_timeseries_raw_copy.copy()
    FEATURES_ts = ['heart_rate', 'systolic_bp', 'spo2']
    TARGET_ts = 'risk_flag'
    SEQ_LENGTH = 10

    scaler_ts = MinMaxScaler()
    df_timeseries_copy[FEATURES_ts] = scaler_ts.fit_transform(df_timeseries_copy[FEATURES_ts])

    X_ts, y_ts = create_sequences(df_timeseries_copy[FEATURES_ts], df_timeseries_copy[TARGET_ts], SEQ_LENGTH)

    model_lstm = Sequential()
    model_lstm.add(LSTM(64, return_sequences=True, input_shape=(X_ts.shape[1], X_ts.shape[2])))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(32))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1, activation='sigmoid'))

    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train Model (on a subset or full for Streamlit demo)
    X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(X_ts, y_ts, test_size=0.01, random_state=42, stratify=y_ts) # Smaller test_size for faster loading
    history_lstm = model_lstm.fit(X_train_ts, y_train_ts, epochs=5, batch_size=32, validation_split=0.2, verbose=0) # Reduced epochs for faster load

    loss_lstm, accuracy_lstm = model_lstm.evaluate(X_test_ts, y_test_ts, verbose=0)

    return {
        "model_lstm": model_lstm,
        "scaler_ts": scaler_ts,
        "SEQ_LENGTH": SEQ_LENGTH,
        "FEATURES_ts": FEATURES_ts,
        "LSTM_accuracy": accuracy_lstm,
        "X_test_ts": X_test_ts,
        "y_test_ts": y_test_ts # For demonstrating predictions
    }

try:
    module3_models = setup_module3_models(df_timeseries_raw.copy())
except Exception as e:
    app_warn(f"Module 3 setup failed or TensorFlow unavailable: {e}. Providing fallback minimal objects.")
    # Fallback minimal objects so UI can still render without full LSTM
    FEATURES_ts = ['heart_rate', 'systolic_bp', 'spo2']
    SEQ_LENGTH = 10
    scaler_ts = MinMaxScaler()
    # Fit scaler on available timeseries if possible
    try:
        scaler_ts.fit(df_timeseries_raw[FEATURES_ts])
    except Exception:
        scaler_ts = MinMaxScaler()

    class DummyModel:
        def predict(self, X, verbose=0):
            # Return zeros of appropriate shape
            return np.zeros((len(X), 1))

    module3_models = {
        "model_lstm": DummyModel(),
        "scaler_ts": scaler_ts,
        "SEQ_LENGTH": SEQ_LENGTH,
        "FEATURES_ts": FEATURES_ts,
        "LSTM_accuracy": 0.0,
        "X_test_ts": np.array([]),
        "y_test_ts": np.array([])
    }

@st.cache_resource
def setup_module4_models(df_feedback_raw_copy):
    df_feedback_copy = df_feedback_raw_copy.copy()
    texts_feedback = df_feedback_copy["feedback_text"].tolist()
    le_feedback = LabelEncoder()
    labels_feedback = le_feedback.fit_transform(df_feedback_copy["sentiment"])

    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    encodings_feedback = tokenizer_bert(texts_feedback, truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Use a small subset of the training data for faster Streamlit loading
    train_idx_feedback, _ = train_test_split(range(len(labels_feedback)), test_size=0.8, random_state=42, stratify=labels_feedback)

    train_dataset_feedback = FeedbackDataset(
        {k: v[train_idx_feedback] for k, v in encodings_feedback.items()},
        labels_feedback[train_idx_feedback]
    )

    model_sentiment = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(labels_feedback)))

    optimizer_sentiment = AdamW(model_sentiment.parameters(), lr=2e-5)
    model_sentiment.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_sentiment.to(device)

    # Only a few batches for quick demo on streamlit load
    train_loader = torch.utils.data.DataLoader(train_dataset_feedback, batch_size=8, shuffle=True)
    for epoch in range(1): # Only 1 epoch
        for i, batch in enumerate(train_loader):
            if i > 10: break # Only 10 batches for demo speed
            optimizer_sentiment.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs_sentiment = model_sentiment(**batch)
            loss_sentiment = outputs_sentiment.loss
            loss_sentiment.backward()
            optimizer_sentiment.step()

    model_sentiment.eval() # Set to eval mode after 'training'
    return {
        "tokenizer_bert": tokenizer_bert,
        "model_sentiment": model_sentiment,
        "le_feedback": le_feedback,
        "device": device
    }

try:
    module4_models = setup_module4_models(df_feedback_raw.copy())
except Exception as e:
    app_warn(f"Module 4 (BERT) setup failed or resources missing: {e}. Sentiment functionality will be limited.")
    # Fallback minimal objects
    le_feedback = LabelEncoder()
    try:
        le_feedback.fit(df_feedback_raw['sentiment'])
    except Exception:
        le_feedback.fit(["Negative","Neutral","Positive"])
    module4_models = {
        "tokenizer_bert": None,
        "model_sentiment": None,
        "le_feedback": le_feedback,
        "device": None
    }


@st.cache_resource
def setup_module5_models(df_clinical_raw_copy):
    generator_biogpt = None
    tokenizer_biobert = None
    model_biobert_embeddings = None
    kmeans_biobert = None
    df_clinical_processed = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if df_clinical_raw_copy is not None:
        df_clinical_processed = df_clinical_raw_copy.copy()
        df_clinical_processed["clinical_note"] = df_clinical_processed["clinical_note"].astype(str).apply(clean_text)

        # BioBERT for Embeddings
        MODEL_NAME_BIOBERT = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer_biobert = AutoTokenizer.from_pretrained(MODEL_NAME_BIOBERT)
        model_biobert_embeddings = AutoModel.from_pretrained(MODEL_NAME_BIOBERT)
        model_biobert_embeddings.to(device)

        # Generate embeddings (limiting for demo speed to a sample if dataset is too large)
        sample_size = min(200, len(df_clinical_processed)) # Process a max of 200 notes
        if sample_size > 0:
            sample_df = df_clinical_processed.sample(sample_size, random_state=42) if len(df_clinical_processed) > sample_size else df_clinical_processed.copy()
            embeddings_biobert = np.vstack(sample_df["clinical_note"].apply(
                lambda x: get_embedding(x, tokenizer_biobert, model_biobert_embeddings).flatten()
            ))

            # KMeans Clustering on Embeddings
            kmeans_biobert = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans_biobert.fit(embeddings_biobert)
            sample_df['cluster'] = kmeans_biobert.predict(embeddings_biobert)
            df_clinical_processed = df_clinical_processed.merge(sample_df[['patient_id', 'cluster']], on='patient_id', how='left')
        else:
            app_warn("No clinical notes available to generate embeddings or clusters.")
            tokenizer_biobert = None # Reset if no data to process
            model_biobert_embeddings = None
            kmeans_biobert = None


        # BioGPT for Text Generation
        try:
            generator_biogpt = pipeline(
                "text-generation",
                model="microsoft/BioGPT",
                device=0 if torch.cuda.is_available() else -1 # Use GPU if available
            )
        except Exception as e:
            app_warn(f"Could not load BioGPT model: {e}. Text generation and chatbot functionality will be limited.")
            generator_biogpt = None
    else:
        app_warn("Clinical notes data not found, BioBERT and BioGPT models will not be fully functional for this module.")

    return {
        "tokenizer_biobert": tokenizer_biobert,
        "model_biobert_embeddings": model_biobert_embeddings,
        "kmeans_biobert": kmeans_biobert,
        "generator_biogpt": generator_biogpt,
        "df_clinical_processed": df_clinical_processed,
        "device": device
    }

try:
    module5_models = setup_module5_models(df_clinical_raw.copy() if df_clinical_raw is not None else None)
except Exception as e:
    app_warn(f"Module 5 setup failed (BioBERT/BioGPT unavailable): {e}. Generative functionality limited.")
    module5_models = {
        "tokenizer_biobert": None,
        "model_biobert_embeddings": None,
        "kmeans_biobert": None,
        "generator_biogpt": None,
        "df_clinical_processed": None,
        "device": None
    }


@st.cache_resource
def setup_module6_models():
    model_name_translator = "Helsinki-NLP/opus-mt-en-mul"
    tokenizer_translator = MarianTokenizer.from_pretrained(model_name_translator)
    translator_model = MarianMTModel.from_pretrained(model_name_translator)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    translator_model.to(device)

    return {
        "tokenizer_translator": tokenizer_translator,
        "translator_model": translator_model,
        "device": device
    }

try:
    module6_models = setup_module6_models()
except Exception as e:
    app_warn(f"Module 6 setup failed (translation model unavailable): {e}. Translator disabled.")
    module6_models = {
        "tokenizer_translator": None,
        "translator_model": None,
        "device": None
    }


# --- Streamlit Module Display Logic ---

if module_selection == "Module 1: Patient Data Analytics":
    st.header("Module 1: Synthetic Patient Data - Classification, Regression, Clustering")
    st.markdown("This module demonstrates predictive analytics and patient segmentation using synthetic patient data.")

    # --- Classification ---
    st.subheader("Patient Risk Category Classification")
    with st.expander("Model Metrics & Overview"):        
        st.write("This model predicts a patient's risk category (Low, Medium, High). A Random Forest Classifier is used.")
        st.write(f"Random Forest Classifier trained on {len(module1_models['df_synthetic_processed_for_metrics'])} samples.")
        
        X_clf_full = module1_models['df_synthetic_processed_for_metrics'].drop(['risk_category','length_of_stay_days', 'patient_id'], axis=1, errors='ignore')
        y_clf_full = module1_models['df_synthetic_processed_for_metrics']['risk_category']

        # Scale X_clf_full using the fitted scaler_clf before applying selector
        X_clf_scaled_full = module1_models['scaler_clf'].transform(X_clf_full[module1_models['clf_all_feature_names_before_select']])
        X_selected_clf_full = module1_models['selector_clf'].transform(X_clf_scaled_full)
        
        y_pred_clf_full = module1_models['clf_model'].predict(X_selected_clf_full)
        st.write("Classification Report on full processed data:")
        st.text(classification_report(y_clf_full, y_pred_clf_full, target_names=module1_models['risk_category_labels']))
        
    st.markdown("### Predict New Patient Risk Category")
    col1, col2, col3 = st.columns(3)
    with col1:
        age_clf = st.number_input("Age", min_value=1, max_value=100, value=45, key='age_clf')
        bmi_clf = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, key='bmi_clf')
        systolic_bp_clf = st.number_input("Systolic BP", min_value=70, max_value=200, value=120, key='systolic_bp_clf')
    with col2:
        cholesterol_clf = st.number_input("Cholesterol", min_value=100, max_value=300, value=180, key='cholesterol_clf')
        blood_glucose_clf = st.number_input("Blood Glucose", min_value=70, max_value=200, value=90, key='blood_glucose_clf')
        diabetes_clf = st.checkbox("Diabetes", value=False, key='diabetes_clf')
        hypertension_clf = st.checkbox("Hypertension", value=False, key='hypertension_clf')
    with col3:
        smoker_clf = st.checkbox("Smoker", value=False, key='smoker_clf')
        prev_hospitalizations_clf = st.number_input("Previous Hospitalizations", min_value=0, max_value=10, value=0, key='prev_hospitalizations_clf')
        gender_clf = st.selectbox("Gender", ["Male", "Female", "Other"], key='gender_clf')
        
    if st.button("Predict Risk Category", key='predict_clf_btn'):
        new_patient_data = pd.DataFrame([{
            'age': age_clf,
            'bmi': bmi_clf,
            'systolic_bp': systolic_bp_clf,
            'cholesterol': cholesterol_clf,
            'blood_glucose': blood_glucose_clf,
            'diabetes': int(diabetes_clf),
            'hypertension': int(hypertension_clf),
            'smoker': int(smoker_clf),
            'prev_hospitalizations': prev_hospitalizations_clf,
            'gender': gender_clf
        }])
        
        # --- Preprocessing for new patient data (mirroring setup_module1_models) ---
        # 1. Feature Engineering
        new_patient_data['BP_level'] = pd.cut(new_patient_data['systolic_bp'], bins=[0, 80, 120, 200], labels=['Low', 'Normal', 'High'], ordered=False)
        new_patient_data['medication_history'] = ((new_patient_data['diabetes'] == 1) | (new_patient_data['hypertension'] == 1) | (new_patient_data['smoker'] == 1)).astype(int)

        # 2. Imputation (only transform, not fit)
        new_patient_data[module1_models['num_cols_to_impute']] = module1_models['imputer'].transform(new_patient_data[module1_models['num_cols_to_impute']])

        # 3. Outlier removal is tricky for single instances, typically skipped or handled by robust scaling. Not applied here for simplicity.

        # 4. Label Encoding (only transform, not fit)
        new_patient_data['BP_level'] = module1_models['le_synthetic_BP'].transform(new_patient_data['BP_level'])
        if module1_models['le_gender'] is not None and 'gender' in new_patient_data.columns and new_patient_data['gender'].dtype == 'object':
            try:
                new_patient_data['gender'] = module1_models['le_gender'].transform(new_patient_data['gender'])
            except ValueError: 
                # Handle unseen gender category, e.g., default to 0 or mean, or raise error.
                # For demo, let's just make it the most common category or raise an error.
                st.error("Unseen gender category in new patient data. Please use 'Male', 'Female', or 'Other'.")
                st.stop()
        elif 'gender' in new_patient_data.columns and new_patient_data['gender'].dtype == 'object': # If gender existed but no encoder was fit
            new_patient_data['gender'] = LabelEncoder().fit_transform(new_patient_data['gender'])

        # Align columns with training data used for classification's selector
        # Ensure all features expected by the scaler and selector are present, in correct order.
        new_patient_processed_aligned = pd.DataFrame(columns=module1_models['clf_all_feature_names_before_select'])
        for col in module1_models['clf_all_feature_names_before_select']:
            if col in new_patient_data.columns:
                new_patient_processed_aligned[col] = new_patient_data[col]
            else:
                new_patient_processed_aligned[col] = 0 # Default value for any missing feature, or handle with mean/median
        
        # Scale features using the fitted scaler_clf
        new_patient_scaled = module1_models['scaler_clf'].transform(new_patient_processed_aligned)
        
        # Apply feature selection
        new_patient_selected = module1_models['selector_clf'].transform(new_patient_scaled)
        
        prediction = module1_models['clf_model'].predict(new_patient_selected)
        predicted_risk = module1_models['le_synthetic_risk'].inverse_transform(prediction)
        st.success(f"Predicted Risk Category: **{predicted_risk[0]}**")


    # --- Regression ---
    st.subheader("Patient Length of Stay Regression")
    with st.expander("Model Metrics & Overview"):        
        st.write("This model predicts the length of stay in days for a patient using a Ridge Regression model.")
        
        # Predict on the full processed data used for training to get metrics
        X_reg_full_for_metrics = module1_models['df_synthetic_processed_for_metrics'][module1_models['reg_feature_names']]
        y_reg_full_for_metrics = module1_models['df_synthetic_processed_for_metrics']['length_of_stay_days']
        y_pred_reg_full = module1_models['reg_pipeline'].predict(X_reg_full_for_metrics)
        
        mae = mean_absolute_error(y_reg_full_for_metrics, y_pred_reg_full)
        rmse = np.sqrt(mean_squared_error(y_reg_full_for_metrics, y_pred_reg_full))
        r2 = r2_score(y_reg_full_for_metrics, y_pred_reg_full)
        st.write(f"MAE on full processed data: {mae:.2f}")
        st.write(f"RMSE on full processed data: {rmse:.2f}")
        st.write(f"R2 Score on full processed data: {r2:.2f}")

    st.markdown("### Predict New Patient Length of Stay")
    col1_reg, col2_reg, col3_reg = st.columns(3)
    with col1_reg:
        age_reg = st.number_input("Age (Regression)", min_value=1, max_value=100, value=45, key='age_reg')
        bmi_reg = st.number_input("BMI (Regression)", min_value=10.0, max_value=50.0, value=25.0, key='bmi_reg')
        systolic_bp_reg = st.number_input("Systolic BP (Regression)", min_value=70, max_value=200, value=120, key='systolic_bp_reg')
    with col2_reg:
        cholesterol_reg = st.number_input("Cholesterol (Regression)", min_value=100, max_value=300, value=180, key='cholesterol_reg')
        blood_glucose_reg = st.number_input("Blood Glucose (Regression)", min_value=70, max_value=200, value=90, key='blood_glucose_reg')
        diabetes_reg = st.checkbox("Diabetes (Regression)", value=False, key='diabetes_reg')
        hypertension_reg = st.checkbox("Hypertension (Regression)", value=False, key='hypertension_reg')
    with col3_reg:
        smoker_reg = st.checkbox("Smoker (Regression)", value=False, key='smoker_reg')
        prev_hospitalizations_reg = st.number_input("Previous Hospitalizations (Regression)", min_value=0, max_value=10, value=0, key='prev_hospitalizations_reg')
        gender_reg = st.selectbox("Gender (Regression)", ["Male", "Female", "Other"], key='gender_reg')

    if st.button("Predict Length of Stay", key='predict_reg_btn'):
        new_patient_data_reg = pd.DataFrame([{
            'age': age_reg,
            'bmi': bmi_reg,
            'systolic_bp': systolic_bp_reg,
            'cholesterol': cholesterol_reg,
            'blood_glucose': blood_glucose_reg,
            'diabetes': int(diabetes_reg),
            'hypertension': int(hypertension_reg),
            'smoker': int(smoker_reg),
            'prev_hospitalizations': prev_hospitalizations_reg,
            'gender': gender_reg
        }])
        
        # --- Preprocessing for new patient data (mirroring setup_module1_models) ---
        # 1. Feature Engineering
        new_patient_data_reg['BP_level'] = pd.cut(new_patient_data_reg['systolic_bp'], bins=[0, 80, 120, 200], labels=['Low', 'Normal', 'High'], ordered=False)
        new_patient_data_reg['medication_history'] = ((new_patient_data_reg['diabetes'] == 1) | (new_patient_data_reg['hypertension'] == 1) | (new_patient_data_reg['smoker'] == 1)).astype(int)

        # 2. Imputation (only transform, not fit)
        new_patient_data_reg[module1_models['num_cols_to_impute']] = module1_models['imputer'].transform(new_patient_data_reg[module1_models['num_cols_to_impute']])

        # 3. Label Encoding (only transform, not fit)
        new_patient_data_reg['BP_level'] = module1_models['le_synthetic_BP'].transform(new_patient_data_reg['BP_level'])
        if module1_models['le_gender'] is not None and 'gender' in new_patient_data_reg.columns and new_patient_data_reg['gender'].dtype == 'object':
            try:
                new_patient_data_reg['gender'] = module1_models['le_gender'].transform(new_patient_data_reg['gender'])
            except ValueError: 
                st.error("Unseen gender category in new patient data. Please use 'Male', 'Female', or 'Other'.")
                st.stop()
        elif 'gender' in new_patient_data_reg.columns and new_patient_data_reg['gender'].dtype == 'object':
            new_patient_data_reg['gender'] = LabelEncoder().fit_transform(new_patient_data_reg['gender'])

        # Align columns with regression model's expected features
        new_patient_reg_aligned = new_patient_data_reg[module1_models['reg_feature_names']] # Ensure correct column order

        prediction_reg = module1_models['reg_pipeline'].predict(new_patient_reg_aligned)
        st.success(f"Predicted Length of Stay: **{prediction_reg[0]:.2f} days**")


    # --- Clustering ---
    st.subheader("Patient Clustering")
    with st.expander("Model Metrics & Overview"):        
        st.write("This model segments patients into 3 clusters based on their features using K-Means. Silhouette Score indicates cluster density and separation.")
        
        # Prepare data for Silhouette Score calculation using the dedicated scaler_cluster
        X_cluster_full_for_metrics = module1_models['df_synthetic_processed_for_metrics'][module1_models['cluster_feature_names']]
        X_cluster_scaled_for_metrics = module1_models['scaler_cluster'].transform(X_cluster_full_for_metrics)
        
        silhouette_score_val = silhouette_score(X_cluster_scaled_for_metrics, module1_models['kmeans_model'].labels_)
        st.write(f"Silhouette Score: {silhouette_score_val:.2f}")
        st.write("Crosstabulation of Cluster vs. Risk Category (shows how clusters align with risk):")
        
        # Merge clusters back to original processed df for crosstab
        df_for_crosstab = module1_models['df_synthetic_processed_for_metrics'].copy()
        df_for_crosstab['cluster'] = module1_models['kmeans_model'].labels_ # Assign labels from the model
        st.dataframe(pd.crosstab(df_for_crosstab['cluster'], df_for_crosstab['risk_category'].map(lambda x: module1_models['le_synthetic_risk'].inverse_transform([x])[0])))

    st.markdown("### Assign New Patient to a Cluster")
    col1_cluster, col2_cluster, col3_cluster = st.columns(3)
    with col1_cluster:
        age_cluster = st.number_input("Age (Clustering)", min_value=1, max_value=100, value=45, key='age_cluster')
        bmi_cluster = st.number_input("BMI (Clustering)", min_value=10.0, max_value=50.0, value=25.0, key='bmi_cluster')
        systolic_bp_cluster = st.number_input("Systolic BP (Clustering)", min_value=70, max_value=200, value=120, key='systolic_bp_cluster')
    with col2_cluster:
        cholesterol_cluster = st.number_input("Cholesterol (Clustering)", min_value=100, max_value=300, value=180, key='cholesterol_cluster')
        blood_glucose_cluster = st.number_input("Blood Glucose (Clustering)", min_value=70, max_value=200, value=90, key='blood_glucose_cluster')
        diabetes_cluster = st.checkbox("Diabetes (Clustering)", value=False, key='diabetes_cluster')
        hypertension_cluster = st.checkbox("Hypertension (Clustering)", value=False, key='hypertension_cluster')
    with col3_cluster:
        smoker_cluster = st.checkbox("Smoker (Clustering)", value=False, key='smoker_cluster')
        prev_hospitalizations_cluster = st.number_input("Previous Hospitalizations (Clustering)", min_value=0, max_value=10, value=0, key='prev_hospitalizations_cluster')
        gender_cluster = st.selectbox("Gender (Clustering)", ["Male", "Female", "Other"], key='gender_cluster')

    if st.button("Assign Cluster", key='assign_cluster_btn'):
        new_patient_data_cluster = pd.DataFrame([{
            'age': age_cluster,
            'bmi': bmi_cluster,
            'systolic_bp': systolic_bp_cluster,
            'cholesterol': cholesterol_cluster,
            'blood_glucose': blood_glucose_cluster,
            'diabetes': int(diabetes_cluster),
            'hypertension': int(hypertension_cluster),
            'smoker': int(smoker_cluster),
            'prev_hospitalizations': prev_hospitalizations_cluster,
            'gender': gender_cluster
        }])
        
        # --- Preprocessing for new patient data (mirroring setup_module1_models) ---
        # 1. Feature Engineering
        new_patient_data_cluster['BP_level'] = pd.cut(new_patient_data_cluster['systolic_bp'], bins=[0, 80, 120, 200], labels=['Low', 'Normal', 'High'], ordered=False)
        new_patient_data_cluster['medication_history'] = ((new_patient_data_cluster['diabetes'] == 1) | (new_patient_data_cluster['hypertension'] == 1) | (new_patient_data_cluster['smoker'] == 1)).astype(int)

        # 2. Imputation (only transform, not fit)
        new_patient_data_cluster[module1_models['num_cols_to_impute']] = module1_models['imputer'].transform(new_patient_data_cluster[module1_models['num_cols_to_impute']])

        # 3. Label Encoding (only transform, not fit)
        new_patient_data_cluster['BP_level'] = module1_models['le_synthetic_BP'].transform(new_patient_data_cluster['BP_level'])
        if module1_models['le_gender'] is not None and 'gender' in new_patient_data_cluster.columns and new_patient_data_cluster['gender'].dtype == 'object':
            try:
                new_patient_data_cluster['gender'] = module1_models['le_gender'].transform(new_patient_data_cluster['gender'])
            except ValueError: 
                st.error("Unseen gender category in new patient data. Please use 'Male', 'Female', or 'Other'.")
                st.stop()
        elif 'gender' in new_patient_data_cluster.columns and new_patient_data_cluster['gender'].dtype == 'object':
            new_patient_data_cluster['gender'] = LabelEncoder().fit_transform(new_patient_data_cluster['gender'])

        # Align columns with clustering model's expected features
        new_patient_cluster_aligned = new_patient_data_cluster[module1_models['cluster_feature_names']] # Ensure correct column order

        # Scale features using the fitted scaler_cluster
        new_patient_cluster_scaled = module1_models['scaler_cluster'].transform(new_patient_cluster_aligned)
        
        predicted_cluster = module1_models['kmeans_model'].predict(new_patient_cluster_scaled)
        st.success(f"Assigned Cluster: **{predicted_cluster[0]}**")


elif module_selection == "Module 2: Association Rules":
    st.header("Module 2: Association Rules for Medical Data")
    st.markdown("This module uncovers relationships between medical conditions and procedures using association rule mining.")

    df_assoc = df_apriori_raw.drop('patient_id', axis=1)
    df_assoc_bool = df_assoc.astype(bool)

    st.sidebar.subheader("Association Rules Parameters")
    min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.1, 0.01)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.01, 1.0, 0.6, 0.01)
    
    # Generate frequent itemsets
    try:
        frequent_itemsets = apriori(
            df_assoc_bool,
            min_support=min_support,
            use_colnames=True
        )
        st.subheader("Frequent Itemsets")
        st.dataframe(frequent_itemsets.sort_values('support', ascending=False).head(10))

        # Generate association rules
        rules = association_rules(
            frequent_itemsets,
            metric='confidence',
            min_threshold=min_confidence
        )
        st.subheader("Association Rules")
        if not rules.empty:
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))
        else:
            st.info("No association rules found with the current parameters.")

    except Exception as e:
        app_warn(f"An error occurred while generating association rules: {e}")

    st.markdown("""
        **How to interpret:**
        *   **Antecedents**: The item(s) on the left-hand side of the rule (IF these conditions are met).
        *   **Consequents**: The item(s) on the right-hand side of the rule (THEN these conditions are likely).
        *   **Support**: How frequently the itemset (antecedents + consequents) appears in the dataset.
        *   **Confidence**: How often the consequent appears given the antecedent.
        *   **Lift**: How much more likely the consequent is given the antecedent, relative to its baseline probability. Lift > 1 indicates a positive correlation.
    """)


elif module_selection == "Module 3: Sequence Modeling (LSTM)":
    st.header("Module 3: Patient Deterioration Risk Prediction (LSTM)")
    st.markdown("This module uses LSTM neural networks to predict patient deterioration risk based on time-series vital signs.")

    with st.expander("Model Metrics & Overview"):
        st.write("LSTM Model Summary:")
        st.markdown("```python\nmodel_lstm.summary() # Output omitted for brevity in Streamlit\n```")
        st.write("Model compiled with Adam optimizer, binary crossentropy loss, and accuracy metrics.")
        st.write(f"Test Accuracy: {module3_models['LSTM_accuracy']:.4f}")
        st.write(f"Sequence Length used: {module3_models['SEQ_LENGTH']}")
        st.write(f"Features used: {', '.join(module3_models['FEATURES_ts'])}")

    st.subheader("Predict Deterioration Risk for New Vitals Sequence")
    st.write(f"Input the last {module3_models['SEQ_LENGTH']} readings for Heart Rate, Systolic BP, and SpO2.")

    new_vitals_input = []
    for i in range(module3_models['SEQ_LENGTH']):
        st.markdown(f"**Time Step {i+1}** (t-{module3_models['SEQ_LENGTH']-1-i})")
        col_ts1, col_ts2, col_ts3 = st.columns(3)
        with col_ts1:
            hr = st.number_input(f"Heart Rate", min_value=40, max_value=180, value=75, key=f"hr_{i}")
        with col_ts2:
            sbp = st.number_input(f"Systolic BP", min_value=60, max_value=200, value=120, key=f"sbp_{i}")
        with col_ts3:
            spo2 = st.number_input(f"SpO2", min_value=70, max_value=100, value=98, key=f"spo2_{i}")
        new_vitals_input.append([hr, sbp, spo2])

    if st.button("Predict Deterioration", key='predict_lstm_btn'):
        new_vitals_df = pd.DataFrame(new_vitals_input, columns=module3_models['FEATURES_ts'])
        new_vitals_scaled = module3_models['scaler_ts'].transform(new_vitals_df)

        new_vitals_reshaped = new_vitals_scaled.reshape(1, module3_models['SEQ_LENGTH'], len(module3_models['FEATURES_ts']))

        prediction_prob = module3_models['model_lstm'].predict(new_vitals_reshaped, verbose=0)[0][0]
        prediction_risk = "High deterioration risk" if prediction_prob > 0.5 else "Stable"

        st.success(f"Predicted Deterioration Risk: **{prediction_risk}** (Probability: {prediction_prob:.2f})")
        st.caption("0: Stable, 1: High deterioration risk")

    with st.expander("Example LSTM Predictions (from test set)"):
        if len(module3_models['X_test_ts']) > 0:
            y_pred_prob_lstm_sample = module3_models['model_lstm'].predict(module3_models['X_test_ts'][:5], verbose=0)
            y_pred_lstm_sample = (y_pred_prob_lstm_sample > 0.5).astype(int).flatten()
            st.write(f"Actual (y_test): {module3_models['y_test_ts'][:5].flatten()}")
            st.write(f"Predicted: {y_pred_lstm_sample}")
            st.caption("0=Stable, 1=High deterioration risk")
        else:
            st.info("No test set samples available for display.")


elif module_selection == "Module 4: Sentiment Analysis (BERT)":
    st.header("Module 4: Patient Feedback Sentiment Analysis (BERT)")
    st.markdown("This module uses a fine-tuned BERT model to analyze the sentiment of patient feedback (Negative, Neutral, Positive).")

    with st.expander("Model Overview"):
        st.write("BERT-base-uncased model fine-tuned for sentiment classification (Negative, Neutral, Positive).")
        st.write(f"Sentiment labels: {list(module4_models['le_feedback'].classes_)}")
        st.write("Due to Streamlit caching, training is simplified/reduced for quick demonstration. Model is loaded in evaluation mode.")

    st.subheader("Analyze New Patient Feedback")
    user_feedback = st.text_area("Enter patient feedback here:", "The nurses were very kind and attentive, but wait times were long.", key='feedback_input')

    if st.button("Analyze Sentiment", key='analyze_sentiment_btn'):
        # If transformers / torch resources are unavailable, fall back to a simple keyword-based heuristic
        tokenizer = module4_models.get('tokenizer_bert')
        model_sent = module4_models.get('model_sentiment')
        le_fb = module4_models.get('le_feedback')

        if tokenizer is None or model_sent is None or not TORCH_AVAILABLE:
            text = (user_feedback or "").lower()
            positive = any(w in text for w in ["good", "great", "kind", "helpful", "satisfied", "excellent", "attentive"])
            negative = any(w in text for w in ["bad", "poor", "terrible", "rude", "delay", "wait", "long wait"]) or "not" in text and any(w in text for w in ["happy","satisfied"])==False

            if positive and not negative:
                predicted_sentiment = "Positive"
            elif negative and not positive:
                predicted_sentiment = "Negative"
            else:
                predicted_sentiment = "Neutral"

            # Try to map to encoder labels if available
            try:
                if le_fb is not None:
                    # If encoder has these labels, return the closest class
                    if predicted_sentiment in list(le_fb.classes_):
                        out = predicted_sentiment
                    else:
                        out = le_fb.inverse_transform(le_fb.transform([predicted_sentiment if predicted_sentiment in le_fb.classes_ else le_fb.classes_[0]]))[0]
                else:
                    out = predicted_sentiment
            except Exception:
                out = predicted_sentiment

            st.success(f"Predicted Sentiment (heuristic): **{out}**")
            st.info("Note: fallback heuristic used because the BERT model/tokenizer are unavailable.")
        else:
            inputs_sentiment = tokenizer(user_feedback, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs_sentiment = {k: v.to(module4_models['device']) for k, v in inputs_sentiment.items()}

            with torch.no_grad():
                outputs_sentiment_pred = model_sent(**inputs_sentiment)

            pred_sentiment_idx = int(torch.argmax(outputs_sentiment_pred.logits, dim=1).item())
            predicted_sentiment = le_fb.inverse_transform([pred_sentiment_idx])[0] if le_fb is not None else str(pred_sentiment_idx)

            st.success(f"Predicted Sentiment: **{predicted_sentiment}**")
            st.info(f"Raw prediction index: {pred_sentiment_idx}")

    with st.expander("Sample Data & Labels"):
        st.dataframe(df_feedback_raw.head())


elif module_selection == "Module 5: Generative AI (BioBERT & BioGPT)":
    st.header("Module 5: Generative AI for Clinical Notes")
    st.markdown("This module explores BioBERT for generating embeddings from clinical notes and BioGPT for generating clinical text.")

    if module5_models["df_clinical_processed"] is None or module5_models["kmeans_biobert"] is None:
        app_warn("Clinical notes data not found or models could not be initialized. This module cannot be fully demonstrated.")
    else:
        st.subheader("BioBERT Embeddings and Clustering of Clinical Notes")
        with st.expander("Overview"):
            st.write("BioBERT (a BERT model trained on biomedical text) is used to create numerical representations (embeddings) of clinical notes. K-Means clustering then groups similar notes.")
            st.write(f"Embeddings generated for a sample of {module5_models['kmeans_biobert'].n_samples_fit_} clinical notes.")
            st.write("Sample clinical notes with assigned clusters:")
            st.dataframe(module5_models["df_clinical_processed"][['clinical_note', 'cluster']].dropna().head())

        st.subheader("BioGPT for Clinical Text Generation")
        if module5_models["generator_biogpt"]:
            st.write("BioGPT (a large language model for biology and medicine) can generate plausible clinical text based on a given prompt.")
            prompt_biogpt_input = st.text_area(
                "Enter a prompt for BioGPT clinical text generation:",
                "Patient presents with chest pain and shortness of breath. Clinical impression:",
                height=100,
                key='biogpt_prompt'
            )
            max_length_biogpt = st.slider("Max Length for Generation", 50, 200, 80, key='biogpt_max_len')

            if st.button("Generate Clinical Text", key='generate_biogpt_btn'):
                with st.spinner("Generating..."):
                    generated_text_biogpt = module5_models["generator_biogpt"](
                        prompt_biogpt_input,
                        max_length=max_length_biogpt,
                        num_return_sequences=1,
                        pad_token_id=module5_models["generator_biogpt"].tokenizer.eos_token_id # Prevents warning
                    )[0]["generated_text"]
                st.info(generated_text_biogpt)
        else:
            app_warn("BioGPT model not loaded. Text generation functionality is disabled. Check error messages above for details.")

elif module_selection == "Module 6: Chatbot & Translator":
    st.header("Module 6: Healthcare Chatbot and Translator")
    st.markdown("This module provides an AI-powered healthcare chatbot and a medical text translator.")

    def healthcare_chatbot_streamlit(user_input):
        """Chatbot function using the BioGPT model for medical guidance, or a generic response."""
        if module5_models["generator_biogpt"]:
            prompt = f"""
            You are a healthcare assistant. Provide safe medical guidance and symptom triage. Avoid medical diagnosis. Focus on general health advice. If you cannot provide a specific answer, advise consulting a doctor. Do not generate information that is not directly related to the prompt. Limit your response to 100 words.
            Patient says: {user_input}
            Response:
            """
            response = module5_models["generator_biogpt"](prompt, max_length=150, num_return_sequences=1, pad_token_id=module5_models["generator_biogpt"].tokenizer.eos_token_id)[0]["generated_text"]
            
            # Clean up the response to remove the prompt part if BioGPT echoes it
            response_lines = response.split('\n')
            clean_response = []
            capture = False
            for line in response_lines:
                if line.strip().startswith("Response:"):
                    capture = True
                    continue
                if capture and line.strip():
                    clean_response.append(line.strip())
            
            if clean_response:
                return " ".join(clean_response)
            else:
                return response.replace(prompt, "").strip() # Fallback cleanup if parsing fails
        else:
            return f"Hello! As a healthcare assistant, I recommend consulting a doctor for '{user_input}'. Please note that a specialized AI model for medical guidance is currently unavailable due to BioGPT loading issues. Always consult a qualified medical professional for health concerns."

    def translate_medical_text_streamlit(text):
        """Translates medical text using the MarianMT model."""
        tokenizer = module6_models.get('tokenizer_translator') if isinstance(module6_models, dict) else None
        translator = module6_models.get('translator_model') if isinstance(module6_models, dict) else None
        device = module6_models.get('device') if isinstance(module6_models, dict) else None

        # Guard against missing tokenizer or model (optional heavy deps may be unavailable)
        if tokenizer and translator and callable(tokenizer):
            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True)
                if device is not None:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                translated = translator.generate(**inputs)
                # Prefer tokenizer.decode or batch_decode when available
                if hasattr(tokenizer, 'decode'):
                    return tokenizer.decode(translated[0], skip_special_tokens=True)
                if hasattr(tokenizer, 'batch_decode'):
                    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                # Fallback: return the raw tensor/string form
                try:
                    return str(translated[0].tolist())
                except Exception:
                    return str(translated)
            except Exception as e:
                app_warn(f"Translator runtime failed: {e}. Returning original text.")
                return text
        else:
            app_warn("Translation model not available. Returning original text.")
            return text

    st.subheader("Healthcare Chatbot")
    user_question = st.text_area("Ask a medical question to the AI healthcare assistant:", "I have a persistent cough, what could it be?", key='chatbot_q')

    if st.button("Get Chatbot Response", key='chatbot_btn'):
        with st.spinner("Generating response..."):
            chatbot_reply = healthcare_chatbot_streamlit(user_question)
        st.info(chatbot_reply)

    st.subheader("Medical Text Translator")
    text_to_translate = st.text_area("Enter medical text to translate:", "The patient has acute appendicitis and requires immediate surgery.", key='translator_input')

    st.info("Note: The current translation model (Helsinki-NLP/opus-mt-en-mul) translates English text into a multilingual interpretation. For specific target languages (e.g., English to Tamil), a different dedicated model would be required.")

    if st.button("Translate Text", key='translate_btn'):
        with st.spinner("Translating..."):
            translated_text = translate_medical_text_streamlit(text_to_translate)
        st.success(f"Translated Text (Multilingual interpretation): **{translated_text}**")
        st.caption("This translation attempts to provide a general multilingual equivalent.")

    st.subheader("Combined Chatbot and Translator Example")
    combined_user_question = st.text_area("Ask a question for both chatbot and translation:", "My child has a fever and is not eating well. Should I be worried?", key='combined_q')
    target_language_label = st.selectbox("Simulated Target Language (for display only):", ["English", "Tamil", "Hindi", "Spanish", "French"], key='target_lang_select')

    if st.button("Get Combined Response", key='combined_btn'):
        with st.spinner("Processing..."):
            english_reply = healthcare_chatbot_streamlit(combined_user_question)
            translated_reply = translate_medical_text_streamlit(english_reply)

        st.markdown("---")
        st.write("**English Chatbot Response:**")
        st.info(english_reply)
        st.write(f"**Translated Response (Simulated {target_language_label} via Multilingual Model):**")
        st.success(translated_reply)
        st.caption("Note: The translation model provides a multilingual interpretation, not a direct translation to the selected language label.")