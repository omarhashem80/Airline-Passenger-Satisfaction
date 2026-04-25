import streamlit as st

st.set_page_config(page_title="Conclusion", page_icon="", layout="wide")

st.title("✅ Final Conclusion")
st.caption("Business-driven insights derived from EDA and model interpretation")


st.subheader("🎯 Core Insight")

st.markdown(
    """
Passenger satisfaction is **not random** — it is driven by a small number of high-impact factors:

### 1. Digital Journey is the Deciding Moment
Online boarding and inflight wifi consistently appear as **top features across all models**.  
Failure here creates dissatisfaction **before the flight experience even begins**.

### 2. Satisfaction is Segment-Driven
- **Business travelers & Business class** → majority satisfied  
- **Personal travelers & Economy classes** → majority dissatisfied  

This indicates a **structural imbalance**, not just service inconsistency.

### 3. Comfort is a Threshold, Not a Differentiator
Seat comfort, leg room, and cleanliness must reach **rating ≥ 4**:
- Below this → strong dissatisfaction  
- Above this → only neutralizes complaints (does not guarantee delight)

👉 These are **minimum expectations**, not competitive advantages.
"""
)

st.subheader("🚀 Strategic Action Plan")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("📱 Digital Experience (Highest ROI)")
    st.write(
        """
- Optimize online boarding flow (reduce friction & errors)  
- Improve app performance and UX  
- Stabilize inflight wifi (consistency > speed)  

👉 Fastest way to increase satisfaction globally
"""
    )

with col2:
    st.success("🧭 Segment Strategy (Structural Fix)")
    st.write(
        """
- Maintain premium quality for business class  
- Redesign economy experience (target biggest dissatisfaction segment)  
- Personal travel requires **value-focused improvements**

👉 Biggest opportunity for impact
"""
    )

with col3:
    st.success("🛫 Cabin Fundamentals (Risk Control)")
    st.write(
        """
- Enforce minimum standards for comfort & cleanliness  
- Prevent low ratings (<4) at all costs  
- Treat these as **quality constraints**, not features

👉 Reduces consistent dissatisfaction baseline
"""
    )


st.subheader("📊 Model Interpretation Takeaway")

st.markdown(
    """
- **Tree-based models (RF, AdaBoost)** highlight service quality and nonlinear effects  
- **Logistic Regression** emphasizes structural factors (travel type, customer type)  

👉 Combined insight:
Satisfaction is driven by both:
- **Operational quality (service, experience)**
- **Customer segmentation (who the passenger is)**

Ignoring either leads to incomplete strategy.
"""
)


st.subheader("🧠 Final Statement")

st.warning(
    """
Improving passenger satisfaction is not about optimizing every feature.

It is about:

✔ Fixing digital entry points  
✔ Addressing economy-class dissatisfaction  
✔ Enforcing minimum service standards  

👉 A focused strategy on these areas will yield the highest return.
"""
)
