# Airline Passenger Satisfaction ✈️

End-to-end data science project to analyze airline customer experience and predict passenger satisfaction using machine learning and interactive analytics.

---

## 🎯 Project Objective

Passenger satisfaction is influenced by multiple factors across the travel journey.  
This project focuses on:

- Identifying **key drivers of satisfaction and dissatisfaction**
- Performing **deep exploratory data analysis (EDA)**
- Training and comparing **three complementary models**
- Delivering **actionable business insights**, not just predictions

---

## 📊 Key Insights (From EDA + Models)

- **Digital touchpoints dominate**  
  → Online boarding and inflight wifi are the strongest drivers across all models  

- **Satisfaction is segment-driven**  
  → Business travelers are mostly satisfied  
  → Personal travelers (especially economy) are mostly dissatisfied  

- **Comfort is a threshold factor**  
  → Ratings below 4 → strong dissatisfaction  
  → Ratings ≥ 4 → only prevent complaints  

- **Different models capture different realities**  
  → Tree models focus on service quality (nonlinear effects)  
  → Logistic Regression highlights customer structure (linear trends)  

---

## 🧠 Business Conclusion

Improving satisfaction does **not require optimizing everything**.  
It requires focusing on:

1. **Digital Experience (highest impact)**  
   Fix onboarding flow and stabilize wifi  

2. **Segment Strategy (biggest opportunity)**  
   Address dissatisfaction in economy & personal travel  

3. **Service Baselines (risk control)**  
   Enforce minimum standards for comfort and cleanliness  

---

## 📘 Notebook Workflow

The notebook (`ml-airline-passenger-satisfaction.ipynb`) follows a full analytical pipeline:

1. Data loading and inspection  
2. Feature understanding  
3. Data cleaning  
4. Exploratory Data Analysis (EDA)  
5. Feature engineering & preprocessing  
6. Model training and evaluation  
7. Feature importance comparison  
8. Business interpretation  

---

## 🤖 Modeling Approach (Focused)

Instead of using many models superficially, this project focuses on **three models deeply**:

- **Logistic Regression**
  - Captures linear relationships
  - Highlights structural drivers (customer type, travel type)

- **Random Forest**
  - Captures nonlinear interactions
  - Identifies complex service-related patterns

- **AdaBoost**
  - Focuses on difficult cases
  - Amplifies high-impact features


---

## 📌 Interactive Dashboard (Streamlit)

Run the app:

```bash
poetry run streamlit run app.py
```

### Pages:

* **📊 Findings**
  - Univariate & Bivariate EDA  
  - Insight-driven visualizations  

* **📌 Feature Importance**
  - Comparison across RF, AdaBoost, LR  
  - Normalized importance analysis  
  - Model agreement vs disagreement  

* **🔮 Prediction**
  - Passenger-level satisfaction prediction  

* **✅ Conclusion**
  - Business strategy recommendations  

---

## 🛠 Tech Stack

* Python 3.14  
* Poetry  
* JupyterLab  

### Data & ML

* Pandas, NumPy  
* Scikit-learn  

### Visualization

* Plotly  

### App

* Streamlit  

---

## ⚙️ Setup & Run

```bash
poetry install --sync
poetry run jupyter lab
poetry run streamlit run app.py
```

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── train.csv
│   └── test.csv
├── app.py
├── ml-airline-passenger-satisfaction.ipynb
├── tmp/
│   └── feature_importance.csv
├── models/
│   └── best_model.pkl
│   └── label_encoder.pkl
├── pyproject.toml
├── poetry.lock
└── README.md
```

---

## 🚀 Future Improvements

- Adding SHAP explainability  

---


## 🧠 Final Note

This project emphasizes **depth over breadth**:

- Strong EDA  
- Focused model comparison  
- Clear business translation  