import streamlit as st

st.set_page_config(page_title="Conclusion", layout="wide")

st.title("Conclusion")
st.caption("Airline Passenger Satisfaction: Model Evaluation & Findings")

st.divider()

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Best Model", "Random Forest", "n_estimators=100 · max_depth=10")
col2.metric("Test Accuracy", "94%", "Train 95% · gap 1%")
col3.metric("Recall (dissatisfied)", "95.92%", "Primary target")
col4.metric("F1 Score", "0.94", "Balanced")

st.markdown(
    """
| Model                | Test Accuracy | Recall (dissatisfied) | F1 (macro) | Train–Test Gap | Overfitting |
|---------------------|--------------|------------------------|------------|----------------|-------------|
| **Random Forest**   | **94%**      | **0.96**               | **0.94**   | 1%             | None        |
| AdaBoost            | 91%          | 0.95                   | 0.91       | 0%             | None        |
| Logistic Regression | 87%          | 0.91                   | 0.87       | 1%             | None        |
| Baseline (ZeroR)    | 57%          | —                      | —          | —              | —           |
"""
)

st.info(
    "Recall on the dissatisfied class was the main objective, "
    "ensuring that unhappy passengers are not misclassified. "
)


st.header("Key Insights")

st.markdown(
    """
- **Random Forest** captures non-linear relationships between features such as
travel class, wifi quality, and online boarding.
- **AdaBoost** improves over linear models but is limited by shallow learners.
- **Logistic Regression** is constrained by its linear structure, limiting its
ceiling.
- A **1% train–test gap** indicates strong generalization with no overfitting.
"""
)

st.header("Feature Signal")

st.markdown(
    """
**Top contributing features:**
- Online Boarding (~19%)
- Inflight Wifi (~15%)
- Travel Type (Personal)
- Class (Economy)
- Entertainment, Seat Comfort

**Low-impact features:**
- Delay metrics (~0.3%)
- Gender (~0.15%)
"""
)

st.markdown(
    """
**Segment behavior:**
- Business passengers: ~69% satisfied
- Personal travelers:  ~90% dissatisfied
- Economy passengers:  >75% dissatisfied
"""
)

st.header("In a Nutshell")

best_model = "Random Forest"
acc = 0.94
recall = 0.9592
f1 = 0.94
ada_acc = 0.91
lr_acc = 0.87
baseline = 0.57

st.markdown(
    f"""
The best results were achieved by **{best_model}**, with accuracy=
{acc:.2f}, recall={recall:.4f}, and F1={f1:.2f}.
AdaBoost performed slightly lower (accuracy={ada_acc:.2f}), while Logistic
Regression went behind (accuracy={lr_acc:.2f}).
All models greatly outperform the baseline model (accuracy={baseline:.2f}),
confirming that the data has useful patterns beyond class imbalance.

This points to the ensemble model capturing most of the useful structure in
the data, likely because it can model non-linear relationships between service
quality, passenger profile, and travel context.
Linear models fail to represent these relationships, while boosting
improves performance but remains limited.

The results indicate that **digital service touchpoints and
passenger segmentation carry the majority of feature importance **, while
operational factors such as delays contribute very little.
Improving satisfaction therefore depends more on optimizing user
experiences (boarding, wifi) than on traditional operational metrics.

The reported metrics provide an overall picture, but a deep look
at misclassified cases are still needed to evaluate edge scenarios such
as mixed passenger profiles or medium service ratings which evaluation
scores like F1 do not capture.
"""
)
