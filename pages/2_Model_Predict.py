import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Predict", page_icon="🤖", layout="wide")

MODEL_PATH = "models/best_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
CONFIG_PATH = "models/config.pkl"
ordinal_cols = [
    'inflight_wifi_service', 'departure/arrival_time_convenient',
    'ease_of_online_booking', 'gate_location', 'food_and_drink',
    'online_boarding', 'seat_comfort', 'inflight_entertainment',
    'on-board_service', 'leg_room_service', 'baggage_handling',
    'checkin_service', 'inflight_service', 'cleanliness'
]

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    config = joblib.load(CONFIG_PATH)
    return model, label_encoder, config


def prepare_input(input_df: pd.DataFrame, model):
    df = input_df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    if "arrival_delay_in_minutes" in df.columns:
        df["arrival_delayed_flag"] = (
            df["arrival_delay_in_minutes"] > 0
        ).astype(int)
        df.drop(columns=["arrival_delay_in_minutes"], inplace=True)

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        df = df[expected]
    df[ordinal_cols] = df[ordinal_cols].replace(0, np.nan)

    return df


model, label_encoder, config = load_artifacts()

st.title("🤖 Model Predict")
st.caption("Create a passenger profile and estimate satisfaction probability")

with st.form("prediction_form"):
    st.subheader("Passenger Input")

    input_data = {}
    cols = list(config.keys())

    left, right = st.columns(2)

    for idx, col in enumerate(cols):
        container = left if idx % 2 == 0 else right

        with container:
            col_config = config[col]

            if col_config["type"] == "numeric":
                if col_config["is_int"]:
                    input_data[col] = st.number_input(
                        col,
                        min_value=int(col_config["min"]),
                        max_value=int(col_config["max"]),
                        value=int(col_config["median"]),
                        step=1,
                    )
                else:
                    input_data[col] = st.number_input(
                        col,
                        min_value=col_config["min"],
                        max_value=col_config["max"],
                        value=col_config["median"],
                    )
            else:
                input_data[col] = st.selectbox(
                    col,
                    options=col_config["options"]
                )

    submitted = st.form_submit_button("Predict Satisfaction")


if submitted:
    input_df = pd.DataFrame([input_data])
    model_input = prepare_input(input_df, model)
    prediction = model.predict(model_input)[0]
    decoded = label_encoder.inverse_transform([int(prediction)])[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(model_input)[0]
        classes = list(model.classes_)

        satisfied_numeric = int(label_encoder.transform(["satisfied"])[0])

        if satisfied_numeric in classes:
            idx = classes.index(satisfied_numeric)
        else:
            idx = int(np.argmax(probs))

        prob_satisfied = float(probs[idx])
    else:
        prob_satisfied = 1.0 if decoded == "satisfied" else 0.0

    st.subheader("Prediction Result")
    st.metric("Predicted Class", decoded)
    st.progress(prob_satisfied)
    st.write(
        f"Estimated probability of satisfaction: **{prob_satisfied * 100:.1f}%**"
    )