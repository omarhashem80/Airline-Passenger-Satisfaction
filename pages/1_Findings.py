import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Findings", page_icon="📊", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def satisfaction_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col)["satisfaction"]
        .apply(lambda x: (x == "satisfied").mean() * 100)
        .reset_index(name="satisfaction_rate")
    )
    return grouped.sort_values("satisfaction_rate", ascending=False)


def two_class_distribution(df: pd.DataFrame, by_col: str) -> pd.DataFrame:
    dff = (
        df.groupby(by_col)["satisfaction"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )
    dff["proportion_pct"] = dff["proportion"] * 100
    return dff


train_df = load_data("data/train.csv")


st.title("📊 Findings")
st.caption("Interactive EDA dashboard based on notebook analysis")

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Records", f"{len(train_df):,}")
with k2:
    st.metric("Average Age", f"{train_df['Age'].mean():.1f}")
with k3:
    st.metric(
        "Overall Satisfaction",
        f"{(train_df['satisfaction'].eq('satisfied').mean() * 100):.1f}%",
    )

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Univariate", "🔗 Bivariate", "🧠 Summary"])

with tab1:
    st.subheader("Univariate Analysis")

    # --- Travel Type ---
    travel_rate = satisfaction_rate(train_df, "Type of Travel")
    fig_travel = px.bar(
        travel_rate,
        x="Type of Travel",
        y="satisfaction_rate",
        color="Type of Travel",
        text="satisfaction_rate",
        title="Satisfaction by Travel Type",
    )
    fig_travel.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_travel.update_layout(showlegend=False, yaxis_title="Satisfaction Rate (%)")

    st.plotly_chart(fig_travel, width="stretch")
    with st.expander("📌 Insight"):
        st.write(
            "Business travelers are significantly more satisfied, while personal travelers show strong dissatisfaction trends."
        )

    # --- Class ---
    class_rate = satisfaction_rate(train_df, "Class")
    fig_class = px.bar(
        class_rate,
        x="Class",
        y="satisfaction_rate",
        color="Class",
        text="satisfaction_rate",
        title="Satisfaction by Class",
    )
    fig_class.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_class.update_layout(showlegend=False, yaxis_title="Satisfaction Rate (%)")

    st.plotly_chart(fig_class, width="stretch")
    with st.expander("📌 Insight"):
        st.write(
            "Business class dominates satisfaction, while economy classes consistently underperform."
        )

    # --- Age Distribution ---
    fig_age_dist = px.histogram(
        train_df,
        x="Age",
        nbins=30,
        title="Age Distribution",
    )
    st.plotly_chart(fig_age_dist, width="stretch")



with tab2:
    st.subheader("Bivariate Analysis Highlights")

    # --- Numerical vs Numerical ---
    st.markdown("### Numerical vs Numerical")

    num_cols = [
        "Age",
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
    ]

    corr = train_df[num_cols].corr(numeric_only=True, method="spearman")
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        title="Spearman Correlation Heatmap",
    )
    st.plotly_chart(fig_corr, width="stretch")

    fig_delay_scatter = px.scatter(
        train_df,
        x="Departure Delay in Minutes",
        y="Arrival Delay in Minutes",
        title="Departure Delay vs Arrival Delay",
        opacity=0.45,
    )
    st.plotly_chart(fig_delay_scatter, width="stretch")

    # --- Categorical vs Numerical ---
    st.markdown("### Categorical vs Numerical")

    age_by_sat = train_df.groupby("satisfaction", as_index=False)["Age"].mean()
    fig_age = px.bar(
        age_by_sat,
        x="satisfaction",
        y="Age",
        color="satisfaction",
        text_auto=".2f",
        title="Mean Age by Satisfaction",
    )
    fig_age.update_layout(showlegend=False)

    st.plotly_chart(fig_age, width="stretch")
    with st.expander("📌 Insight"):
        st.write(
            "Older passengers tend to report higher satisfaction, while younger passengers are more critical."
        )

    # --- Categorical vs Categorical ---
    st.markdown("### Categorical vs Categorical")

    commentary_map = {
        "Gender": "Low predictive power: satisfaction is nearly identical across genders.",
        "Customer Type": "Disloyal customers are far more dissatisfied than loyal ones.",
        "Type of Travel": "Personal travelers show extreme dissatisfaction (~90%).",
        "Class": "Business class drives satisfaction; economy segments fail expectations.",
        "Inflight wifi service": "Poor wifi is worse than no wifi.",
        "Online boarding": "One of the strongest drivers of satisfaction.",
        "Seat comfort": "Critical factor — low comfort leads to high dissatisfaction.",
        "Cleanliness": "Clear linear relationship with satisfaction.",
    }

    focus_features = list(commentary_map.keys())
    selected_feature = st.selectbox(
        "Choose a feature to inspect", focus_features
    )

    distribution_df = two_class_distribution(train_df, selected_feature)

    fig_feature = px.bar(
        distribution_df,
        x=selected_feature,
        y="proportion_pct",
        color="satisfaction",
        barmode="group",
        text="proportion_pct",
        title=f"Satisfaction Distribution by {selected_feature}",
    )
    fig_feature.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_feature.update_layout(yaxis_title="Percentage (%)")

    st.plotly_chart(fig_feature, width="stretch")

    with st.expander("📌 Insight"):
        st.write(commentary_map[selected_feature])


with tab3:
    st.subheader("Notebook-Aligned EDA Summary")

    st.markdown(
        """
### Key Findings

1. Digital touchpoints (online boarding, wifi) are the strongest drivers.
2. Business passengers are structurally more satisfied than personal travelers.
3. Comfort and cleanliness must reach rating 4+ to impact satisfaction.
4. Economy class shows consistent dissatisfaction across most features.
"""
    )