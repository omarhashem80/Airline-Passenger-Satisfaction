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
| Baseline (ZeroR)    | 56%          |  1.00                      | 0.72         | Not Applicable             | Not Applicable           |
"""
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

st.header("Features Signal")

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
1. The best results were achieved by **{best_model}**, with accuracy=
{acc:.2f}, recall={recall:.4f}, and F1={f1:.2f}.
AdaBoost performed slightly lower (accuracy={ada_acc:.2f}), while Logistic
Regression went behind (accuracy={lr_acc:.2f}).
All models greatly outperform the baseline model (accuracy={baseline:.2f}),
confirming that the data has useful patterns beyond class imbalance.

2. This points to the ensemble model capturing most of the useful structure in
the data, likely because it can model non-linear relationships between service
quality, passenger profile, and travel context.
Linear models fail to represent these relationships, while boosting
improves performance but remains limited.

3. The business should prioritize online boarding and inflight Wi-Fi, since
these are among the strongest drivers of satisfaction.
4. Travel class and type of travel should be treated as major customer
segments, because business travelers and business-class passengers are much
more satisfied than economy and personal travelers.
5. Seat comfort, leg room, inflight entertainment, and cleanliness should be
improved together, as they strongly shape the passenger experience.
6. Poor service ratings on these experience-related features lead to much
higher dissatisfaction, so fixing them will have more impact on satisfaction
level.

7. The airline should pay special attention to economy and personal-travel
passengers, since they show the lowest satisfaction and represent the biggest
improvement opportunity.

8. The reported metrics provide an overall picture, but a deep look
at misclassified cases are still needed to evaluate edge scenarios such
as mixed passenger profiles or medium service ratings which evaluation
scores like F1 do not capture.
"""
)
