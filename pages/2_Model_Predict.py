import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Model Predict", page_icon="🤖", layout="wide")

TARGET_COL = "satisfaction"
MODEL_PATH = "models/best_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model_and_encoder():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder


def prepare_input_for_model(input_df: pd.DataFrame, model) -> pd.DataFrame:
    prepared = input_df.copy()

    prepared.columns = prepared.columns.str.lower().str.replace(" ", "_")

    if "arrival_delay_in_minutes" in prepared.columns:
        prepared["arrival_delayed_flag"] = (
            prepared["arrival_delay_in_minutes"] > 0
        ).astype(int)
        prepared = prepared.drop(columns=["arrival_delay_in_minutes"])

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in prepared.columns:
                prepared[col] = 0
        prepared = prepared[expected]

    return prepared


train_df = load_data("data/train.csv")
feature_data = train_df.drop(
    columns=[
        c for c in [TARGET_COL, "id", "Unnamed: 0"] if c in train_df.columns
    ]
)
model, label_encoder = load_model_and_encoder()

st.title("🤖 Model Predict")
st.caption("Create a passenger profile and estimate satisfaction probability")
st.info(
    f"Loaded model from {MODEL_PATH} and label encoder from {LABEL_ENCODER_PATH}"
)

with st.form("prediction_form"):
    st.subheader("Passenger Input")

    input_data = {}

    left, right = st.columns(2)
    cols = feature_data.columns.tolist()

    for idx, col in enumerate(cols):
        target_col = left if idx % 2 == 0 else right
        with target_col:
            series = feature_data[col]
            if pd.api.types.is_numeric_dtype(series):
                min_value = float(series.min())
                max_value = float(series.max())
                median_value = float(series.median())

                if pd.api.types.is_integer_dtype(series):
                    input_data[col] = st.number_input(
                        col,
                        min_value=int(min_value),
                        max_value=int(max_value),
                        value=int(median_value),
                        step=1,
                    )
                else:
                    input_data[col] = st.number_input(
                        col,
                        min_value=min_value,
                        max_value=max_value,
                        value=median_value,
                    )
            else:
                options = sorted(series.dropna().unique().tolist())
                input_data[col] = st.selectbox(col, options=options)

    submitted = st.form_submit_button("Predict Satisfaction")

if submitted:
    input_df = pd.DataFrame([input_data])
    model_input_df = prepare_input_for_model(input_df, model)
    prediction = model.predict(model_input_df)[0]

    if isinstance(label_encoder, LabelEncoder):
        decoded_prediction = label_encoder.inverse_transform(
            [int(prediction)]
        )[0]
    else:
        decoded_prediction = str(prediction)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(model_input_df)[0]
        classes = (
            list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        )
        satisfied_numeric = int(label_encoder.transform(["satisfied"])[0])
        satisfied_idx = (
            classes.index(satisfied_numeric)
            if satisfied_numeric in classes
            else int(np.argmax(probs))
        )
        prob_satisfied = float(probs[satisfied_idx])
    else:
        prob_satisfied = 1.0 if decoded_prediction == "satisfied" else 0.0

    st.subheader("Prediction Result")
    st.metric("Predicted Class", decoded_prediction)
    st.progress(prob_satisfied)
    st.write(
        f"Estimated probability of satisfaction: **{prob_satisfied * 100:.1f}%**"
    )
