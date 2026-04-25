import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Findings", page_icon="📊", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


train_df = load_data("data/train.csv")


def create_histogram_with_boxplot(df: pd.DataFrame, x: str):
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Histogram(x=df[x], name='Histogram'), row=1, col=1)
    fig.add_trace(go.Box(x=df[x], name="Box"), row=2, col=1)
    fig.update_layout(
        title=f'Distribution of {x}',
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)


def create_pie_chart(df: pd.DataFrame, cat: str):
    fig = px.pie(names=df[cat], hole=0.3, title=f"Distribution of {cat}")
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)


def create_bar_chart(df: pd.DataFrame, cat: str, target: str):
    dff = df.groupby(cat)[target].value_counts(normalize=True).unstack().reset_index()
    dff = dff.melt(id_vars=cat, var_name=target, value_name='proportion')
    dff["proportion_pct"] = dff["proportion"] * 100
    fig = px.bar(
        dff,
        x=cat,
        y='proportion_pct',
        color=target,
        barmode='group',
        text_auto='.2f',
        title=f'Distribution of {target} by {cat}',
        labels={cat: cat.capitalize(), 'proportion_pct': 'Proportion (%)', target: target.capitalize()}
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def satisfaction_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)["satisfaction"]
        .apply(lambda x: (x == "satisfied").mean() * 100)
        .reset_index(name="satisfaction_rate")
        .sort_values("satisfaction_rate", ascending=False)
    )


def two_class_distribution(df: pd.DataFrame, by_col: str) -> pd.DataFrame:
    dff = (
        df.groupby(by_col)["satisfaction"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )
    dff["proportion_pct"] = dff["proportion"] * 100
    return dff


def categorical_distribution(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df[col]
        .value_counts(normalize=True)
        .mul(100)
        .reset_index(name="percentage")
        .rename(columns={"index": col})
    )


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

    # Define all columns
    num_cols = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class", "Inflight wifi service", 
                "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", 
                "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", 
                "On-board service", "Leg room service", "Baggage handling", "Checkin service", 
                "Inflight service", "Cleanliness", "satisfaction"]

    st.markdown("### Numerical Features")
    selected_num = st.selectbox(
        "Select a numerical feature",
        num_cols,
        key="num_select"
    )
    create_histogram_with_boxplot(train_df, selected_num)

    st.markdown("### Categorical Features")
    selected_cat = st.selectbox(
        "Select a categorical feature",
        cat_cols,
        key="cat_select"
    )
    create_pie_chart(train_df, selected_cat)


with tab2:
    st.subheader("Bivariate Analysis")

    st.markdown("### Categorical vs Satisfaction")

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

    st.markdown("### Numerical vs Numerical")

    num_cols = [
        "Age",
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
    ]

    corr = train_df[num_cols].corr(method="spearman")

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        title="Spearman Correlation Heatmap",
    )
    st.plotly_chart(fig_corr, width="stretch")

    fig_delay = px.scatter(
        train_df,
        x="Departure Delay in Minutes",
        y="Arrival Delay in Minutes",
        opacity=0.4,
        title="Departure vs Arrival Delay",
    )
    st.plotly_chart(fig_delay, width="stretch")

    st.markdown("### Feature vs Satisfaction Distribution")

    commentary_map = {
        "Gender": "Low impact on satisfaction.",
        "Customer Type": "Disloyal customers are much more dissatisfied.",
        "Type of Travel": "A massive 90% of personal travelers are dissatisfied, making this one of the strongest negative predictors.",
        "Class": "Business class is the only category where a majority of passengers are satisfied (69%).",
        "Inflight wifi service": "Satisfaction spikes to nearly 99% when wifi is rated 5, while ratings below 4 result in at least 67% dissatisfaction.",
        "Departure/Arrival time convenient": "Satisfaction remains relatively stable across all ratings, suggesting that schedule convenience is not a primary driver of overall passenger sentiment.",
        "Ease of Online booking": "Satisfaction jumps significantly once booking ease reaches a 4 or 5 rating, while low scores (1-3) correlate with roughly 70% dissatisfaction.",
        "Gate location": "Gate location shows a weak relationship with satisfaction, as dissatisfaction peaks at mid-range ratings.",
        "Food and drink": "Satisfaction only becomes the majority outcome when food and drink are rated 4 or 5. A rating of 1 leads to 80% dissatisfaction.",
        "Online boarding": "Satisfaction sky-rockets from roughly 13% to over 87% once the online boarding score hits 5, making this a critical tipping point feature.",
        "Seat comfort": "Satisfaction flips to a majority only at ratings of 4 or 5, proving that good seat comfort is a prerequisite for a positive experience.",
        "Inflight entertainment": "Satisfaction jumps from 27% to over 61% when entertainment moves from a 3 to a 4, marking it as a high-impact delighter feature.",
        "On-board service": "Satisfaction only breaks the 50% threshold when on-board service reaches a rating of 4 or 5, highlighting it as a crucial driver of positive sentiment.",
        "Leg room service": "Satisfaction becomes the majority only when leg room is rated 4 or 5, highlighting it as a significant physical comfort driver.",
        "Baggage handling": "Satisfaction only becomes the majority at a top rating of 5, indicating that passengers view good baggage handling as a basic expectation.",
        "Checkin service": "Satisfaction only hits a clear majority when check-in service is rated a perfect 5. Poor check-in (ratings 0-2) almost guarantees a negative outcome.",
        "Inflight service": "Much like baggage handling, inflight service only reaches a majority satisfaction level when rated a perfect 5.",
        "Cleanliness": "Satisfaction follows a clear linear trend—once cleanliness hits a rating of 4 or 5, the majority of passengers shift to being satisfied.",
    }

    selected_feature = st.selectbox(
        "Choose a feature",
        list(commentary_map.keys()),
    )

    create_bar_chart(train_df, selected_feature, "satisfaction")

    with st.expander("📌 Insight"):
        st.write(commentary_map[selected_feature])


with tab3:
    st.subheader("EDA Summary")

    st.markdown(
        """
### Key Findings

1. Digital services (online boarding, wifi) are the strongest drivers.
2. Business travelers are significantly more satisfied.
3. Comfort and cleanliness must be high to impact satisfaction.
4. Economy class consistently underperforms.
"""
    )