import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Airline Satisfaction Insights",
    page_icon="✈️",
    layout="wide",
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


train_df = load_data("data/train.csv")

st.title("✈️ Airline Passenger Satisfaction Dashboard")
st.caption("From raw passenger records to actionable service insights")

left, right = st.columns([2, 1])
with left:
    st.markdown(
        """
### Project Story
This app publishes key findings from the ML project and provides a prediction workspace
for testing passenger profiles. Use the pages in the left sidebar to explore:

- **Findings**: full EDA walkthrough with interpreted insights from the notebook
- **Model Predict**: input passenger details and estimate satisfaction probability
- **Feature Importance**: model-side ranking of drivers across RF, AdaBoost, and LR
- **Conclusion**: final takeaways and action plan
        """
    )

with right:
    total_rows = len(train_df)
    satisfied_share = (train_df["satisfaction"].eq("satisfied").mean()) * 100

    st.metric("Passenger Records", f"{total_rows:,}")
    st.metric("Satisfied Share", f"{satisfied_share:.1f}%")

st.divider()
st.subheader("Quick Dataset Snapshot")
st.dataframe(train_df.head(10), width="stretch")

st.info(
    "Use the sidebar to switch pages. The app is designed for both project demo and decision support."
)
