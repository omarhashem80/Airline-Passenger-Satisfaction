import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Feature Importance", page_icon="📌", layout="wide")


# =========================
# 📥 LOAD DATA
# =========================
@st.cache_data
def load_feature_importance(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("📌 Feature Importance Dashboard")
st.caption("Model interpretability across RF, AdaBoost, and Logistic Regression")

df = load_feature_importance("tmp/feature_importance.csv")

required_cols = ["feature", "RF", "AdaBoost", "LR"]
if not all(col in df.columns for col in required_cols):
    st.error("Missing required columns")
    st.stop()

df = df.dropna()

# =========================
# ⚙️ NORMALIZATION (CRITICAL)
# =========================
norm_df = df.copy()
for col in ["RF", "AdaBoost", "LR"]:
    norm_df[col] = norm_df[col] / norm_df[col].sum()

# Average importance
norm_df["avg"] = norm_df[["RF", "AdaBoost", "LR"]].mean(axis=1)
norm_df["std"] = norm_df[["RF", "AdaBoost", "LR"]].std(axis=1)

norm_df = norm_df.sort_values("avg", ascending=False)


# =========================
# 🧩 TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Overview", "⚖️ Model Comparison", "🧠 Insights"])


# =========================
# 📊 TAB 1 — OVERVIEW
# =========================
with tab1:
    st.subheader("Top Driving Features")

    top_n = st.slider("Select Top Features", 5, 20, 10)

    top_df = norm_df.head(top_n)

    fig = px.bar(
        top_df,
        x="avg",
        y="feature",
        orientation="h",
        text="avg",
        title="Top Features (Average Importance Across Models)"
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


# =========================
# ⚖️ TAB 2 — MODEL COMPARISON
# =========================
with tab2:
    st.subheader("Model-wise Comparison")

    selected_features = st.multiselect(
        "Select Features",
        options=norm_df["feature"].tolist(),
        default=norm_df.head(8)["feature"].tolist()
    )

    comp_df = norm_df[norm_df["feature"].isin(selected_features)]

    melt_df = comp_df.melt(
        id_vars="feature",
        value_vars=["RF", "AdaBoost", "LR"],
        var_name="Model",
        value_name="Importance"
    )

    fig = px.bar(
        melt_df,
        x="feature",
        y="Importance",
        color="Model",
        barmode="group",
        text_auto=".2f",
        title="Feature Importance by Model"
    )

    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# 🧠 TAB 3 — INSIGHTS
# =========================
with tab3:
    st.subheader("Model Interpretation")

    # Top features
    top5 = norm_df.head(5)["feature"].tolist()

    # Leaders
    top_rf = norm_df.loc[norm_df["RF"].idxmax(), "feature"]
    top_ab = norm_df.loc[norm_df["AdaBoost"].idxmax(), "feature"]
    top_lr = norm_df.loc[norm_df["LR"].idxmax(), "feature"]

    # Agreement
    consistent = norm_df.nsmallest(5, "std")["feature"].tolist()
    disagreement = norm_df.nlargest(5, "std")["feature"].tolist()

    # Dominance
    dominant = norm_df[
        (norm_df[["RF", "AdaBoost", "LR"]].max(axis=1) > 0.25) &
        (norm_df["std"] > 0.05)
    ]["feature"].tolist()

    st.markdown("### 🔍 Key Findings")

    st.markdown(f"""
- 🔝 **Top drivers overall**: {', '.join(top5)}
- 🌳 **Random Forest focuses on**: **{top_rf}**
- ⚡ **AdaBoost focuses on**: **{top_ab}**
- 📈 **Logistic Regression focuses on**: **{top_lr}**
    """)

    st.markdown("### 🤝 Model Agreement")
    st.success(", ".join(consistent))

    st.markdown("### ⚠️ Model Disagreement")
    st.warning(", ".join(disagreement))

    if dominant:
        st.markdown("### 🔥 Model-Sensitive Features")
        st.error(", ".join(dominant))

    st.markdown("""
### 💡 Interpretation

- Tree-based models (RF, AdaBoost) capture **nonlinear interactions** and service-related features.
- Logistic Regression emphasizes **structural factors** (e.g., travel type, customer type).
- Features like **online boarding and wifi** are consistently strong → key business drivers.
""")