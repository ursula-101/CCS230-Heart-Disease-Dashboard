# Heart Disease Risk Prediction: Data Mining for Clinical Decision Support

A strategic data mining implementation designed to predict heart disease risk using machine learning classification models. This project combines rigorous statistical analysis with interpretable clinical decision-making to optimize patient screening workflows in hospital settings.

---

## Project Overview

**Objective:** Deploy an automated screening tool that identifies high-risk heart disease patients early, minimizing false negatives and enabling timely clinical interventions.

**Key Achievement:** Random Forest model achieves **91.2% Recall** compared to 76.5% for baseline Decision Tree, ensuring critical cases are not missed during initial triage.

**Use Case:** Integration with hospital Electronic Health Record (EHR) systems for automated patient risk stratification and cardiology resource allocation.

---

## Dataset & Methodology

### Data Profile
- **Patient Records:** 918 historical cases
- **Clinical Features:** 11 key metrics
  - Demographics: Age, Sex
  - Vital Signs: Resting Blood Pressure, Maximum Heart Rate
  - Diagnostic Markers: Cholesterol, Chest Pain Type, ECG Readings, ST-Slope, Exercise-Induced Angina
- **Data Quality:** Handles missing values (zero-imputed blood pressure/cholesterol) through median substitution

### Analysis Phases
1. **Association Rule Mining** – Identifies symptom co-occurrence patterns predicting heart disease
2. **Clustering Analysis** – K-Means segmentation into 3 distinct patient phenotypes
3. **Classification Modeling** – Decision Tree vs. Random Forest comparison
4. **Feature Importance Analysis** – Validates clinical relevance of predictive drivers

---

## Model Comparison

| Metric | Decision Tree | Random Forest |
|--------|---------------|---------------|
| **Accuracy** | 83.0% | 87.7% |
| **Precision** | 78.3% | 88.0% |
| **Recall** | 76.5% | 91.2% |
| **False Negatives** | 24 | 9 |

### Why Random Forest Was Selected
- **Superior Recall:** Minimizes missed diagnoses (false negatives)
- **Ensemble Robustness:** 300-tree aggregation eliminates individual tree overfitting
- **Clinical Priority:** Aligns with non-maleficence principle—safety net must catch true positives even if slightly higher false alarms
- **Balanced Performance:** Maintains 88% precision while maximizing sensitivity

---

## Key Clinical Insights

### High-Risk Patient Patterns

**"Silent" Risk Profile**
- Asymptomatic chest pain + Flat ST-slope during stress testing = Exceptionally high disease probability
- Emphasizes: Absence of symptoms should NOT lower clinical suspicion when electrical anomalies present

**Exercise-Induced Ischemia**
- Exercise-induced angina + Flat ST-slope = Immediate advanced diagnostic routing
- Clear indicator requiring secondary screening

**Combined Asymptomatic Risk**
- Initially asymptomatic → Develops exercise-induced angina = High-risk group
- Baseline symptoms misleading; stress testing is critical

### Patient Clustering

| Cluster | Profile | Preventive Strategy | Disease Incidence |
|---------|---------|-------------------|-------------------|
| **1: Silent** | Older males, asymptomatic, abnormal ECG | Automated EHR alerts for stress echo/cardiology | Highest |
| **2: Metabolic** | Middle-aged, elevated BP/cholesterol | Aggressive lifestyle interventions, lipid monitoring | Moderate |
| **3: Baseline** | Younger, normal ECG, high exercise tolerance | Standard primary care tracking | Lowest |

---

## Getting Started

### Requirements
```
Python 3.8+
streamlit >= 1.28.0
scikit-learn >= 1.3.0
pandas >= 1.5.0
plotly >= 5.15.0
mlxtend >= 0.23.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Ensure heart.csv is in the same directory as streamlitdashboard.py
```

### Running the Dashboard
```bash
streamlit run streamlitdashboard.py
```
Dashboard launches at `http://localhost:8501`

---

## Dashboard Features

### 1. **Overview Page**
- Summary statistics of cleaned patient dataset
- Disease distribution (donut chart)
- Demographic breakdowns

### 2. **Exploratory Data Analysis (EDA)**
- Numeric distributions (Age, BP, Cholesterol, Heart Rate)
- Correlation heatmap
- Categorical pattern analysis (Sex, ECG, Chest Pain Type)
- Multivariate scatter plots
- Disease prevalence by clinical feature

### 3. **Association Rules**
- High-confidence symptom combinations
- Lift and support metrics
- Actionable clinical insights from frequent patterns

### 4. **Risk Predictor**
- Interactive form for single-patient risk assessment
- Side-by-side model predictions (Decision Tree vs. Random Forest)
- Combined risk score
- Screening priority classification

### 5. **Model Comparison**
- Performance metrics visualization
- Confusion matrices (false negatives/positives highlighted)
- Feature importance ranking
- Interpretable decision tree logic diagrams

---

## Feature Importance Ranking

**Top 5 Predictive Drivers (Random Forest):**
1. **ST_Slope (Upsloping vs. Flat/Down)** – Strongest indicator
2. **ST_Slope (Flat patterns)** – Electrical abnormality
3. **Chest Pain Type (Asymptomatic)** – Silent ischemia signal
4. **Oldpeak (ST depression)** – Exercise-induced changes
5. **MaxHR (Maximum heart rate)** – Exercise capacity

*Note:* Algorithm correctly prioritizes clinical markers over demographics, validating medical principles.

---

## Expected Clinical Impact

### Patient Safety
- **Reduced False Negatives:** 24 → 9 cases (63% improvement)
- **Early Detection:** Enable timely interventions before acute events
- **Risk Awareness:** Automated screening reduces physician cognitive load

### Operational Efficiency
- **Resource Optimization:** Prioritize cardiology appointments for high-risk patients
- **Cost Reduction:** Shift care from emergency to preventative settings
- **Data-Driven Scheduling:** Evidence-based cardiologist capacity planning

### Business Value
- Estimated 15-20% reduction in preventable readmissions
- Improved patient outcomes → better quality metrics
- Scalable framework for other chronic conditions

---

## Project Structure

```
heart-disease-prediction/
├── streamlitdashboard.py       # Main Streamlit application
├── styles.py                    # Custom CSS styling
├── heart.csv                    # Patient dataset (918 records)
├── Icons/                       # Navigation icons
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── Aquino_Bunag_Carbonell_Corpes_CCS230_Finals.pdf
                                # Full technical report
```

---

## Ethical Considerations

### Clinical Governance
- Model serves as **decision-support, not replacement** for physician judgment
- All algorithmic recommendations require physician review
- Explainability frameworks (SHAP/Feature Importance) ensure transparency

### Bias Mitigation
- Balanced class weighting in model training
- Validation across demographic subgroups
- Regular audits for performance disparities

### Data Privacy
- De-identified patient records
- HIPAA-compliant EHR integration
- Access controls and audit logging

---

## Team

**Authors (West Visayas State University, CCS 230 - Data Mining)**
- Dallas A. Aquino
- Frederick Jibril L. Bunag
- Ethan Jed V. Carbonell
- Vincent L. Corpes, Jr.

**Advisor:** Ralph Voltaire Dayot, Professor

**Submission Date:** May 2026
