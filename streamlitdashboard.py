import warnings
from pathlib import Path
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from streamlit_option_menu import option_menu
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

from styles import apply_styles, inject_nav_icon_fix

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")

st.set_page_config(
    page_title="Heart Disease Case Study Dashboard",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).with_name("heart.csv")
ICONS_PATH = Path(__file__).parent / "Icons"


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_icon_as_base64(icon_name: str) -> str:
    """Load icon as base64 data URL."""
    icon_path = ICONS_PATH / f"{icon_name}.png"
    if icon_path.exists():
        with open(icon_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    return ""


def plotly_template() -> dict:
    return dict(
        layout=dict(
            template="simple_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#132238", family="Inter, Segoe UI, sans-serif"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
    )


def render_card_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def model_probability_label(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.45:
        return "Moderate"
    return "Lower"


def rule_to_sentence(antecedents: frozenset) -> str:
    pieces = []
    for item in sorted(antecedents):
        label = item.replace("HeartDisease_Label_Yes", "Heart Disease")
        label = label.replace("_", " ")
        pieces.append(label)
    return ", ".join(pieces)


# ── Data loading & cleaning ────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def clean_heart_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    for column in ["RestingBP", "Cholesterol"]:
        cleaned[column] = cleaned[column].replace(0, np.nan)
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    cleaned["FastingBS"] = cleaned["FastingBS"].astype(int)
    cleaned["HeartDisease"] = cleaned["HeartDisease"].astype(int)
    return cleaned


def add_apriori_bins(data: pd.DataFrame) -> pd.DataFrame:
    binned = data.copy()
    binned["Age_Bin"] = pd.cut(
        binned["Age"],
        bins=[0, 40, 55, 100],
        labels=["Young", "Middle_Aged", "Older"],
        include_lowest=True,
    )
    binned["RestingBP_Bin"] = pd.cut(
        binned["RestingBP"],
        bins=[0, 120, 140, 300],
        labels=["Normal", "Elevated", "High"],
        include_lowest=True,
    )
    binned["Cholesterol_Bin"] = pd.cut(
        binned["Cholesterol"],
        bins=[0, 200, 240, 1000],
        labels=["Normal", "Borderline", "High"],
        include_lowest=True,
    )
    binned["MaxHR_Bin"] = pd.cut(
        binned["MaxHR"],
        bins=[0, 100, 140, 250],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    binned["Oldpeak_Bin"] = pd.cut(
        binned["Oldpeak"],
        bins=[-10, 0, 1.5, 10],
        labels=["Normal", "Moderate", "High"],
        include_lowest=True,
    )
    binned["HeartDisease_Label"] = binned["HeartDisease"].map({0: "No", 1: "Yes"})
    return binned


def build_association_rules(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    binned_df = add_apriori_bins(cleaned_df)
    transaction_columns = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
        "FastingBS",
        "Age_Bin",
        "RestingBP_Bin",
        "Cholesterol_Bin",
        "MaxHR_Bin",
        "Oldpeak_Bin",
        "HeartDisease_Label",
    ]
    transactions = pd.get_dummies(binned_df[transaction_columns].astype(str), prefix_sep="_", dtype=bool)
    frequent_itemsets = apriori(transactions, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    target_rules = rules[rules["consequents"] == frozenset({"HeartDisease_Label_Yes"})].copy()
    return target_rules.sort_values(["lift", "confidence", "support"], ascending=False)


def train_models(cleaned_df: pd.DataFrame):
    X = cleaned_df.drop(columns=["HeartDisease"])
    y = cleaned_df["HeartDisease"]
    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    decision_tree = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced", min_samples_leaf=5)),
        ]
    )
    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced_subsample",
                    min_samples_leaf=3,
                ),
            ),
        ]
    )

    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    def summarize(model):
        predictions = model.predict(X_test)
        return {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "Predictions": predictions,
        }

    dt_results = summarize(decision_tree)
    rf_results = summarize(random_forest)
    return X_test, y_test, decision_tree, random_forest, dt_results, rf_results


# ── Bootstrap data & models ────────────────────────────────────────────────────
raw_df = load_data()
cleaned_df = clean_heart_data(raw_df)
association_rule_table = build_association_rules(cleaned_df)
X_test, y_test, dt_model, rf_model, dt_results, rf_results = train_models(cleaned_df)


# ── Render functions ───────────────────────────────────────────────────────────
def render_kpi_grid() -> None:
    positive_rate = cleaned_df["HeartDisease"].mean() * 100
    rf_recall = rf_results["Recall"] * 100
    dt_recall = dt_results["Recall"] * 100

    patients_icon = load_icon_as_base64("patients")
    features_icon = load_icon_as_base64("Features")
    positive_rate_icon = load_icon_as_base64("Positive_Rate")
    recall_icon = load_icon_as_base64("Top_Recall_Model")

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon-wrapper patients">
                    <img src="{patients_icon}" alt="Patients">
                </div>
                <div class="kpi-content">
                    <div class="kpi-label">Patients</div>
                    <div class="kpi-value">{len(cleaned_df):,}</div>
                    <div class="kpi-delta positive">+{len(cleaned_df):,} records</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon-wrapper features">
                    <img src="{features_icon}" alt="Features">
                </div>
                <div class="kpi-content">
                    <div class="kpi-label">Features</div>
                    <div class="kpi-value">{cleaned_df.shape[1] - 1}</div>
                    <div class="kpi-delta positive">{cleaned_df.shape[1] - 1} clinical attributes</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon-wrapper positive-rate">
                    <img src="{positive_rate_icon}" alt="Positive Rate">
                </div>
                <div class="kpi-content">
                    <div class="kpi-label">Positive Rate</div>
                    <div class="kpi-value">{positive_rate:.1f}%</div>
                    <div class="kpi-delta negative">Disease prevalence: {positive_rate:.1f}%</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon-wrapper recall">
                    <img src="{recall_icon}" alt="Top Recall Model">
                </div>
                <div class="kpi-content">
                    <div class="kpi-label">Top Recall Model</div>
                    <div class="kpi-value">{rf_recall:.1f}%</div>
                    <div class="kpi-delta positive">vs Decision Tree ({dt_recall:.1f}%)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## Heart Disease Case Study")
        page = option_menu(
            menu_title=None,
            options=["Overview", "EDA", "Association Rules", "Risk Predictor", "Model Comparison"],
            icons=["house", "bar-chart", "list", "activity", "diagram-3"],
            orientation="vertical",
            default_index=0,
            styles={
                "container": {
                    "padding": "0",
                    "background-color": "transparent",
                    "display": "flex",
                    "flex-direction": "column",
                    "overflow": "visible",
                    "gap": "6px",
                },
                "icons": {"font-size": "18px", "color": "#eb95e0"},
                "nav-link": {
                    "font-size": "0.95rem",
                    "text-align": "left",
                    "margin": "0.125rem 0",
                    "border-radius": "12px",
                    "padding": "0.6rem 0.9rem",
                    "color": "#132238",
                    "background-color": "transparent",
                },
                "nav-link-selected": {
                    "background-color": "#eb95e0",
                    "color": "#ffffff",
                    "font-weight": "600",
                    "icons-color": "#ffffff",
                },
            },
        )
    return page


def render_sidebar_filters() -> tuple[pd.DataFrame, tuple[int, int], str | None]:
    """Render all filters in the sidebar as collapsible expander sections."""
    # Define all defaults upfront for consistent reset behavior
    default_age_range = (35, 75)
    sex_options = ["All"] + sorted(cleaned_df["Sex"].unique().tolist())
    chest_pain_options = sorted(cleaned_df["ChestPainType"].unique().tolist())
    ecg_options = sorted(cleaned_df["RestingECG"].unique().tolist())
    angina_options = ["All"] + sorted(cleaned_df["ExerciseAngina"].unique().tolist())
    slope_options = sorted(cleaned_df["ST_Slope"].unique().tolist())
    fasting_options = ["All", 0, 1]
    
    with st.sidebar:
        # ── Header row: "Filters" title + Reset All button ──────────────────
        hdr_left, hdr_right = st.columns([1, 1])
        with hdr_left:
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:600; color:#132238; "
                "padding-top:0px; margin-top:-8px;'>Filters</div>",
                unsafe_allow_html=True,
            )
        with hdr_right:
            if st.button("Reset All", key="reset_all_filters", use_container_width=True):
                st.session_state["age_range_filter"] = default_age_range
                st.session_state["sex_filter"] = "All"
                st.session_state["chest_pain_filter"] = chest_pain_options
                st.session_state["resting_ecg_filter"] = ecg_options
                st.session_state["exercise_angina_filter"] = "All"
                st.session_state["st_slope_filter"] = slope_options
                st.session_state["fasting_bs_filter"] = "All"
                st.rerun()

        # ── Age Range ────────────────────────────────────────────────────────
        min_age = int(cleaned_df["Age"].min())
        max_age = int(cleaned_df["Age"].max())
        current_age_range = st.session_state.get("age_range_filter", default_age_range)
        if not isinstance(current_age_range, tuple) or len(current_age_range) != 2:
            current_age_range = default_age_range

        with st.expander("Age Range", expanded=False):
            age_range = st.slider(
                "",
                min_age,
                max_age,
                current_age_range,
                key="age_range_filter",
                label_visibility="collapsed",
            )

        # ── Sex ──────────────────────────────────────────────────────────────
        sex_options = ["All"] + sorted(cleaned_df["Sex"].unique().tolist())
        current_sex = st.session_state.get("sex_filter", "All")
        if current_sex not in sex_options:
            current_sex = "All"

        with st.expander("Sex", expanded=False):
            sex_choice = st.pills(
                "",
                sex_options,
                selection_mode="single",
                default=current_sex,
                key="sex_filter",
                label_visibility="collapsed",
            )
            if sex_choice is None:
                sex_choice = "All"

        # ── Chest Pain Type ──────────────────────────────────────────────────
        chest_pain_options = sorted(cleaned_df["ChestPainType"].unique().tolist())
        current_cp = st.session_state.get("chest_pain_filter", chest_pain_options)
        if not isinstance(current_cp, list):
            current_cp = chest_pain_options

        with st.expander("Chest Pain Type", expanded=False):
            chest_pain_choices = st.pills(
                "",
                chest_pain_options,
                selection_mode="multi",
                default=current_cp,
                key="chest_pain_filter",
                label_visibility="collapsed",
            )
            if not chest_pain_choices:
                chest_pain_choices = chest_pain_options

        # ── Resting ECG ──────────────────────────────────────────────────────
        ecg_options = sorted(cleaned_df["RestingECG"].unique().tolist())
        current_ecg = st.session_state.get("resting_ecg_filter", ecg_options)
        if not isinstance(current_ecg, list):
            current_ecg = ecg_options

        with st.expander("Resting ECG", expanded=False):
            ecg_choices = st.pills(
                "",
                ecg_options,
                selection_mode="multi",
                default=current_ecg,
                key="resting_ecg_filter",
                label_visibility="collapsed",
            )
            if not ecg_choices:
                ecg_choices = ecg_options

        # ── Exercise Angina ──────────────────────────────────────────────────
        angina_options = ["All"] + sorted(cleaned_df["ExerciseAngina"].unique().tolist())
        current_angina = st.session_state.get("exercise_angina_filter", "All")
        if current_angina not in angina_options:
            current_angina = "All"

        with st.expander("Exercise Angina", expanded=False):
            exercise_choice = st.pills(
                "",
                angina_options,
                selection_mode="single",
                default=current_angina,
                key="exercise_angina_filter",
                label_visibility="collapsed",
            )
            if exercise_choice is None:
                exercise_choice = "All"

        # ── ST Slope ─────────────────────────────────────────────────────────
        slope_options = sorted(cleaned_df["ST_Slope"].unique().tolist())
        current_slope = st.session_state.get("st_slope_filter", slope_options)
        if not isinstance(current_slope, list):
            current_slope = slope_options

        with st.expander("ST Slope", expanded=False):
            slope_choices = st.pills(
                "",
                slope_options,
                selection_mode="multi",
                default=current_slope,
                key="st_slope_filter",
                label_visibility="collapsed",
            )
            if not slope_choices:
                slope_choices = slope_options

        # ── Fasting Blood Sugar ───────────────────────────────────────────────
        fasting_options: list = ["All", 0, 1]
        current_fasting = st.session_state.get("fasting_bs_filter", "All")
        if current_fasting not in fasting_options:
            current_fasting = "All"

        with st.expander("Fasting Blood Sugar", expanded=False):
            fasting_choice = st.pills(
                "",
                fasting_options,
                selection_mode="single",
                default=current_fasting,
                key="fasting_bs_filter",
                label_visibility="collapsed",
            )
            if fasting_choice is None:
                fasting_choice = "All"

    # ── Apply filters to data ────────────────────────────────────────────────
    filtered = cleaned_df[cleaned_df["Age"].between(age_range[0], age_range[1])].copy()
    if sex_choice != "All":
        filtered = filtered[filtered["Sex"] == sex_choice]
    if chest_pain_choices:
        filtered = filtered[filtered["ChestPainType"].isin(chest_pain_choices)]
    if ecg_choices:
        filtered = filtered[filtered["RestingECG"].isin(ecg_choices)]
    if exercise_choice != "All":
        filtered = filtered[filtered["ExerciseAngina"] == exercise_choice]
    if slope_choices:
        filtered = filtered[filtered["ST_Slope"].isin(slope_choices)]
    if fasting_choice != "All":
        filtered = filtered[filtered["FastingBS"] == fasting_choice]

    return filtered, age_range, None if sex_choice == "All" else sex_choice


def render_hero_shell() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <h1 class="hero-title">Dashboard</h1>
            <p class="hero-subtitle">Comprehensive clinical screening platform for heart disease early detection, risk stratification, and evidence-based patient assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_donut_chart(data: pd.DataFrame) -> None:
    balance = data["HeartDisease"].value_counts().rename({0: "No Heart Disease", 1: "Heart Disease"}).reset_index()
    balance.columns = ["Outcome", "Count"]
    fig = px.pie(
        balance, values="Count", names="Outcome", hole=0.48,
        color="Outcome", color_discrete_sequence=["#3b82f6", "#f59e0b"],
    )
    fig.update_layout(**plotly_template()["layout"], legend=dict(orientation="h", y=-0.12, x=0.18))
    fig.update_traces(textposition="inside", textinfo="percent+label")
    with st.container(border=True):
        render_card_title("Heart Disease Distribution", "Donut chart using the filtered view.")
        st.plotly_chart(fig, use_container_width=True)


def render_numeric_distributions(data: pd.DataFrame) -> None:
    fig = px.histogram(
        data, x="Age", nbins=24,
        color_discrete_sequence=["#eb95e0"], title="Age Distribution",
    )
    fig.update_layout(**plotly_template()["layout"], bargap=0.12)
    with st.container(border=True):
        render_card_title("Age Distribution", "Plotly histograms with a clean SaaS-style presentation.")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        data, x="HeartDisease", y="Oldpeak", color="HeartDisease",
        color_discrete_sequence=["#f5c5e5", "#d9449f"], title="Oldpeak by Heart Disease",
    )
    fig.update_layout(**plotly_template()["layout"], showlegend=False)
    with st.container(border=True):
        render_card_title("Oldpeak vs Heart Disease", "Higher Oldpeak remains one of the strongest numeric signals.")
        st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(data: pd.DataFrame) -> None:
    corr_matrix = data[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HeartDisease"]].corr(numeric_only=True)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="pinkyl",
            zmid=0,
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(**plotly_template()["layout"], title="Numeric Correlation Heatmap")
    with st.container(border=True):
        render_card_title("Correlation Heatmap", "Focus on how clinical variables relate to heart disease.")
        st.plotly_chart(fig, use_container_width=True)


def render_categorical_patterns(data: pd.DataFrame) -> None:
    stacked = data.groupby(["ChestPainType", "HeartDisease"]).size().reset_index(name="Count")
    stacked["HeartDisease"] = stacked["HeartDisease"].map({0: "No", 1: "Yes"})
    fig = px.bar(
        stacked, x="ChestPainType", y="Count", color="HeartDisease",
        barmode="stack", color_discrete_sequence=["#f5c5e5", "#d9449f"],
        title="Chest Pain Type vs Heart Disease",
    )
    fig.update_layout(**plotly_template()["layout"], legend=dict(orientation="h", y=-0.18, x=0.22))
    with st.container(border=True):
        render_card_title("Categorical Pattern", "Asymptomatic chest pain is consistently overrepresented in positive cases.")
        st.plotly_chart(fig, use_container_width=True)


def render_sex_distribution(data: pd.DataFrame) -> None:
    sex_disease = data.groupby(["Sex", "HeartDisease"]).size().reset_index(name="Count")
    sex_disease["HeartDisease"] = sex_disease["HeartDisease"].map({0: "No", 1: "Yes"})
    fig = px.bar(
        sex_disease, x="Sex", y="Count", color="HeartDisease",
        barmode="group", color_discrete_sequence=["#f5c5e5", "#d9449f"],
        title="Sex Distribution by Heart Disease",
    )
    fig.update_layout(**plotly_template()["layout"], legend=dict(orientation="h", y=-0.18, x=0.25))
    with st.container(border=True):
        render_card_title("Sex Distribution", "Gender breakdown across heart disease cases.")
        st.plotly_chart(fig, use_container_width=True)


def render_ecg_patterns(data: pd.DataFrame) -> None:
    ecg_disease = data.groupby(["RestingECG", "HeartDisease"]).size().reset_index(name="Count")
    ecg_disease["HeartDisease"] = ecg_disease["HeartDisease"].map({0: "No", 1: "Yes"})
    fig = px.bar(
        ecg_disease, x="RestingECG", y="Count", color="HeartDisease",
        barmode="stack", color_discrete_sequence=["#f5c5e5", "#eb95e0"],
        title="Resting ECG by Heart Disease",
    )
    fig.update_layout(**plotly_template()["layout"], legend=dict(orientation="h", y=-0.18, x=0.2))
    with st.container(border=True):
        render_card_title("Resting ECG Patterns", "Resting electrocardiogram results correlation with disease.")
        st.plotly_chart(fig, use_container_width=True)


def render_exercise_angina_patterns(data: pd.DataFrame) -> None:
    angina_disease = data.groupby(["ExerciseAngina", "HeartDisease"]).size().reset_index(name="Count")
    angina_disease["HeartDisease"] = angina_disease["HeartDisease"].map({0: "No", 1: "Yes"})
    fig = px.bar(
        angina_disease, x="ExerciseAngina", y="Count", color="HeartDisease",
        barmode="group", color_discrete_sequence=["#f5c5e5", "#d9449f"],
        title="Exercise Angina by Heart Disease",
    )
    fig.update_layout(**plotly_template()["layout"], legend=dict(orientation="h", y=-0.18, x=0.22))
    with st.container(border=True):
        render_card_title("Exercise Angina Impact", "Chest pain induced by exercise is a strong predictor.")
        st.plotly_chart(fig, use_container_width=True)


def render_st_slope_patterns(data: pd.DataFrame) -> None:
    slope_disease = data.groupby(["ST_Slope", "HeartDisease"]).size().reset_index(name="Count")
    slope_disease["HeartDisease"] = slope_disease["HeartDisease"].map({0: "No", 1: "Yes"})
    fig = px.bar(
        slope_disease, x="ST_Slope", y="Count", color="HeartDisease",
        barmode="stack", color_discrete_sequence=["#f5c5e5", "#d9449f"],
        title="ST Slope by Heart Disease",
    )
    fig.update_layout(**plotly_template()["layout"], legend=dict(orientation="h", y=-0.18, x=0.25))
    with st.container(border=True):
        render_card_title("ST Segment Slope", "ST segment slope trend indicates cardiac stress response.")
        st.plotly_chart(fig, use_container_width=True)


def render_additional_numeric_distributions(data: pd.DataFrame) -> None:
    left, right = st.columns(2)
    
    with left:
        fig = px.histogram(
            data, x="RestingBP", nbins=20,
            color_discrete_sequence=["#eb95e0"], title="Resting BP Distribution",
        )
        fig.update_layout(**plotly_template()["layout"], bargap=0.12)
        with st.container(border=True):
            render_card_title("Resting Blood Pressure", "Distribution of resting blood pressure across patients.")
            st.plotly_chart(fig, use_container_width=True)
    
    with right:
        fig = px.histogram(
            data, x="Cholesterol", nbins=20,
            color_discrete_sequence=["#d9449f"], title="Cholesterol Distribution",
        )
        fig.update_layout(**plotly_template()["layout"], bargap=0.12)
        with st.container(border=True):
            render_card_title("Cholesterol Levels", "Distribution of serum cholesterol across patients.")
            st.plotly_chart(fig, use_container_width=True)
    
    left, right = st.columns(2)
    
    with left:
        fig = px.histogram(
            data, x="MaxHR", nbins=20,
            color_discrete_sequence=["#eb95e0"], title="Maximum Heart Rate Distribution",
        )
        fig.update_layout(**plotly_template()["layout"], bargap=0.12)
        with st.container(border=True):
            render_card_title("Maximum Heart Rate", "Distribution of maximum heart rate achieved during exercise.")
            st.plotly_chart(fig, use_container_width=True)
    
    with right:
        fig = px.box(
            data, y="MaxHR", color="HeartDisease",
            color_discrete_sequence=["#f5c5e5", "#d9449f"],
            title="MaxHR by Heart Disease Status",
        )
        fig.update_layout(**plotly_template()["layout"], showlegend=False)
        with st.container(border=True):
            render_card_title("MaxHR vs Disease", "Maximum heart rate tends to be lower in heart disease cases.")
            st.plotly_chart(fig, use_container_width=True)


def render_fasting_bs_patterns(data: pd.DataFrame) -> None:
    fasting_disease = data.groupby(["FastingBS", "HeartDisease"]).size().reset_index(name="Count")
    fasting_disease["FastingBS"] = fasting_disease["FastingBS"].map({0: "Normal", 1: "Elevated"})
    fasting_disease["HeartDisease"] = fasting_disease["HeartDisease"].map({0: "No", 1: "Yes"})
    fig = px.bar(
        fasting_disease, x="FastingBS", y="Count", color="HeartDisease",
        barmode="group", color_discrete_sequence=["#f5c5e5", "#d9449f"],
        title="Fasting Blood Sugar by Heart Disease",
    )
    fig.update_layout(**plotly_template()["layout"], legend=dict(orientation="h", y=-0.18, x=0.2))
    with st.container(border=True):
        render_card_title("Fasting Blood Sugar", "Elevated fasting blood sugar levels and disease correlation.")
        st.plotly_chart(fig, use_container_width=True)


def render_multivariate_scatter(data: pd.DataFrame) -> None:
    scatter_data = data.copy()
    scatter_data["Oldpeak_Size"] = np.abs(scatter_data["Oldpeak"]) + 1  # Use absolute value and add 1 for visibility
    scatter_data["HeartDisease_Label"] = scatter_data["HeartDisease"].map({0: "No", 1: "Yes"})
    fig = px.scatter(
        scatter_data, x="Age", y="MaxHR", color="HeartDisease_Label",
        size="Oldpeak_Size", hover_data=["RestingBP", "Cholesterol", "Oldpeak"],
        color_discrete_map={"No": "#f5c5e5", "Yes": "#d9449f"},
        title="Age vs MaxHR (sized by ST Depression)",
    )
    fig.update_layout(**plotly_template()["layout"], legend=dict(orientation="h", y=-0.15, x=0.25))
    with st.container(border=True):
        render_card_title("Multivariate Relationship", "Age, maximum heart rate, and ST depression relationships.")
        st.plotly_chart(fig, use_container_width=True)


def render_association_rules(data: pd.DataFrame) -> None:
    with st.container(border=True):
        render_card_title("Association Rules", "Plain-English insights where Heart Disease is the consequent.")
        if association_rule_table.empty:
            st.warning("No rules were found with the current support and confidence thresholds.")
            return

        view = association_rule_table.head(8).copy()
        view["Rule"] = view["antecedents"].apply(rule_to_sentence).apply(
            lambda text: f"If {text}, then Heart Disease risk is elevated."
        )
        display_table = view[["Rule", "support", "confidence", "lift"]].copy()
        display_table["support"] = display_table["support"].map(lambda x: f"{x:.3f}")
        display_table["confidence"] = display_table["confidence"].map(lambda x: f"{x:.3f}")
        display_table["lift"] = display_table["lift"].map(lambda x: f"{x:.3f}")
        st.dataframe(display_table, use_container_width=True, hide_index=True)

    insight_cols = st.columns(3)
    actionable_rules = view.head(3).to_dict("records")
    for idx, rule in enumerate(actionable_rules):
        with insight_cols[idx]:
            with st.container(border=True):
                st.markdown(f"**Insight {idx + 1}**")
                st.write(rule_to_sentence(rule["antecedents"]))
                st.caption(f'Confidence {rule["confidence"]:.1%} | Lift {rule["lift"]:.2f}')


def render_detailed_confusion_matrix(model_name: str, predictions, cm_data, model_accuracy, model_precision, model_recall, model_f1, is_random_forest: bool = False) -> None:
    """Render a detailed confusion matrix with metrics and clinical interpretation."""
    cm = cm_data.values
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    annotations = np.array([
        [f"True Negative (TN)\n{int(tn)}", f"False Positive (FP)\n{int(fp)}"],
        [f"False Negative (FN)\n{int(fn)}", f"True Positive (TP)\n{int(tp)}"]
    ])
    
    custom_cmap = sns.light_palette("#eb95e0", as_cmap=True)
    sns.heatmap(
        cm, annot=annotations, fmt="", cmap=custom_cmap, cbar=False,
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"],
        annot_kws={"size": 12}, ax=ax
    )
    
    plt.xlabel("Predicted Value", fontweight="bold", labelpad=10)
    plt.ylabel("Actual Value", fontweight="bold", labelpad=10)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    st.pyplot(fig, clear_figure=True)
    
    # Display metrics in a professional grid
    metrics_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; margin-bottom: 1.5rem;">
        <div style="padding: 12px; background: #f5f5f5; border-radius: 6px; border-left: 4px solid #2196f3;">
            <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Accuracy</div>
            <div style="font-size: 22px; font-weight: 700; color: #2196f3; margin-top: 4px;">{model_accuracy:.1%}</div>
        </div>
        <div style="padding: 12px; background: #f5f5f5; border-radius: 6px; border-left: 4px solid #ff9800;">
            <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Precision</div>
            <div style="font-size: 22px; font-weight: 700; color: #ff9800; margin-top: 4px;">{model_precision:.1%}</div>
        </div>
        <div style="padding: 12px; background: #f5f5f5; border-radius: 6px; border-left: 4px solid #4caf50;">
            <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Recall</div>
            <div style="font-size: 22px; font-weight: 700; color: #4caf50; margin-top: 4px;">{model_recall:.1%}</div>
        </div>
        <div style="padding: 12px; background: #f5f5f5; border-radius: 6px; border-left: 4px solid #9c27b0;">
            <div style="font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">F1 Score</div>
            <div style="font-size: 22px; font-weight: 700; color: #9c27b0; margin-top: 4px;">{model_f1:.1%}</div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Clinical interpretation
    if is_random_forest:
        interpretation = f"""
        **Clinical Insight:** Random Forest identifies **{int(tp)}** true positives (patients correctly flagged with disease risk), 
        reducing dangerous false negatives to just **{int(fn)}**. This **high recall ({model_recall:.1%})** is critical in cardiac screening 
        where missing a patient is more costly than false alarms. The model balances sensitivity with precision ({model_precision:.1%}), 
        making it suitable for population health screening.
        """
    else:
        interpretation = f"""
        **Clinical Insight:** Decision Tree achieves **{model_accuracy:.1%}** accuracy with **{int(tp)}** true positives, 
        but produces **{int(fn)}** false negatives. While more conservative, the lower recall ({model_recall:.1%}) means higher risk 
        of missing cases, which is problematic in cardiac screening where sensitivity is paramount.
        """
    
    st.markdown(interpretation)


def render_model_comparison() -> None:
    metrics_df = pd.DataFrame(
        [
            {"Model": "Decision Tree", **{k: v for k, v in dt_results.items() if k != "Predictions"}},
            {"Model": "Random Forest", **{k: v for k, v in rf_results.items() if k != "Predictions"}},
        ]
    )

    left, right = st.columns([1, 1.15])
    with left:
        with st.container(border=True):
            render_card_title("Model Metrics", "Recall is the primary decision criterion.")
            st.dataframe(
                metrics_df.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )
    
    # Calculate confusion matrices
    cm_dt = pd.crosstab(y_test, dt_results["Predictions"], rownames=["Actual"], colnames=["Predicted"])
    cm_rf = pd.crosstab(y_test, rf_results["Predictions"], rownames=["Actual"], colnames=["Predicted"])
    cm_dt = cm_dt.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    cm_rf = cm_rf.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    
    # Calculate F1 scores
    dt_f1 = f1_score(y_test, dt_results["Predictions"])
    rf_f1 = f1_score(y_test, rf_results["Predictions"])
    
    # Full-width confusion matrix comparison
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        render_card_title("Confusion Matrix Analysis", "Detailed model predictions breakdown with clinical metrics")
        
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.markdown("### Decision Tree Model")
            render_detailed_confusion_matrix(
                "Decision Tree", 
                dt_results["Predictions"], 
                cm_dt, 
                dt_results["Accuracy"],
                dt_results["Precision"],
                dt_results["Recall"],
                dt_f1,
                is_random_forest=False
            )
        
        with right_col:
            st.markdown("### Random Forest Model")
            render_detailed_confusion_matrix(
                "Random Forest",
                rf_results["Predictions"],
                cm_rf,
                rf_results["Accuracy"],
                rf_results["Precision"],
                rf_results["Recall"],
                rf_f1,
                is_random_forest=True
            )
    
    # Comparison insight
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### Why Random Forest Outperforms")
        col1, col2, col3 = st.columns(3)
        
        recall_diff = (rf_results["Recall"] - dt_results["Recall"]) * 100
        fn_reduction = int(cm_dt.iloc[1, 0] - cm_rf.iloc[1, 0])
        f1_diff = (rf_f1 - dt_f1) * 100
        
        with col1:
            st.metric(
                "Recall Improvement", 
                f"+{recall_diff:.1f}%",
                f"{fn_reduction} fewer missed diagnoses"
            )
        
        with col2:
            st.metric(
                "False Negative Reduction",
                f"{fn_reduction} cases",
                "Fewer critical misses in screening"
            )
        
        with col3:
            st.metric(
                "F1 Score Gap",
                f"+{f1_diff:.1f}%",
                "Better balanced performance"
            )
        
        st.info(
            f" **Clinical Recommendation:** Random Forest's superior recall ({rf_results['Recall']:.1%}) "
            f"and lower false negative rate ({int(cm_rf.iloc[1, 0])} missed cases) make it the preferred model for "
            f"heart disease screening, where sensitivity is crucial for patient safety."
        )

    # Performance Comparison Bar Chart
    metrics_melted = metrics_df.melt(
        id_vars=["Model"], 
        value_vars=["Accuracy", "Precision", "Recall"],
        var_name="Metric", 
        value_name="Score"
    )
    fig_perf = px.bar(
        metrics_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#d9449f", "#eb95e0"],
    )
    fig_perf.update_layout(
        **plotly_template()["layout"],
        title="Performance Comparison Bar Chart",
        yaxis_title="Score",
        xaxis_title="Metric",
        legend=dict(orientation="h", y=1.12, x=0.3),
    )
    fig_perf.update_traces(marker_line_color="rgba(0,0,0,0)")
    with st.container(border=True):
        render_card_title("Performance Comparison", "Decision Tree vs Random Forest model metrics.")
        st.plotly_chart(fig_perf, use_container_width=True)

    # Feature Importance Chart
    rf_feature_importance = rf_model.named_steps["model"].feature_importances_
    rf_feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
    importance_df = pd.DataFrame({
        "Feature": rf_feature_names,
        "Importance": rf_feature_importance
    }).sort_values("Importance", ascending=True).tail(15)
    
    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        color="Importance",
        orientation="h",
        color_continuous_scale=["#f5c5e5", "#eb95e0", "#d9449f"],
    )
    fig_importance.update_layout(
        **plotly_template()["layout"],
        title="Top 15 Feature Importances (Random Forest)",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        showlegend=False,
        height=500,
    )
    with st.container(border=True):
        render_card_title("Feature Importance Analysis", "Most impactful features for predicting heart disease.")
        st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("### Executive Decision Path")
    st.markdown(
        """
        <div style="background: #ffffff; padding: 25px; border-radius: 20px; border: 1px solid #e4e8f0; margin-bottom: 25px;">
            <p style="color: #63728a; font-size: 0.95rem; margin-bottom: 20px;">
                This decision tree breaks down patient risk based on clinical markers. Follow the branches from top to bottom 
                to see how the model categorizes patients based on ST Slope, Chest Pain Type, and other key features.
            </p>
        """
    , unsafe_allow_html=True)
    
    mermaid_code = """
    graph TD
      %% Node Definitions
      Node0["`**Initial Screen: ST Segment Slope**
      *Rule: Is the ST slope upward?*
      Samples: 918 patients`"]
      
      Node1["`**Chest Pain Assessment**
      *Rule: Is pain asymptomatic (ASY)?*
      Samples: 523 patients`"]
      
      Node8["`**Chest Pain Assessment**
      *Rule: Is pain asymptomatic (ASY)?*
      Samples: 395 patients`"]

      Node2["`**Exercise Capacity (MaxHR)**
      *Rule: Heart rate > 136.5?*
      Samples: 159 patients`"]
      
      Node5["`**Demographic Context (Sex)**
      *Rule: Is the patient Male?*
      Samples: 364 patients`"]
      
      Node9["`**Cardiac Stress (Oldpeak)**
      *Rule: Depression > 1.95?*
      Samples: 263 patients`"]
      
      Node12["`**Cardiac Stress (Oldpeak)**
      *Rule: Depression > 0.45?*
      Samples: 132 patients`"]

      %% Leaf Nodes (Outcomes)
      Node3["`**HIGH RISK**
      Confidence: 79.3%
      Action: Urgent Referral`"]
      
      Node4["`**LOW RISK**
      Confidence: 58.6%
      Action: Routine Check`"]
      
      Node6["`**HIGH RISK**
      Confidence: 66.9%
      Action: Secondary Screening`"]
      
      Node7["`**CRITICAL RISK**
      Confidence: 92.6%
      Action: Immediate Intervention`"]
      
      Node10["`**MINIMAL RISK**
      Confidence: 96.2%
      Action: Annual Monitoring`"]
      
      Node11["`**MODERATE RISK**
      Confidence: 50.2%
      Action: Follow-up Test`"]
      
      Node13["`**STABLE RISK**
      Confidence: 74.8%
      Action: Lifestyle Advice`"]
      
      Node14["`**HIGH RISK**
      Confidence: 72.5%
      Action: Clinical Review`"]

      %% Connections with Annotations
      Node0 -- "No (Flat/Down)" --> Node1
      Node0 -- "Yes (Upward)" --> Node8
      
      Node1 -- "Typical/Other" --> Node2
      Node1 -- "Asymptomatic" --> Node5
      
      Node2 -- "Low HR (≤136)" --> Node3
      Node2 -- "Normal HR (>136)" --> Node4
      
      Node5 -- "Female" --> Node6
      Node5 -- "Male" --> Node7
      
      Node8 -- "Typical/Other" --> Node9
      Node8 -- "Asymptomatic" --> Node12
      
      Node9 -- "Normal (≤1.95)" --> Node10
      Node9 -- "Abnormal (>1.95)" --> Node11
      
      Node12 -- "Minimal (≤0.45)" --> Node13
      Node12 -- "Significant (>0.45)" --> Node14

      %% Styling
      classDef default font-family:Calibri,fill:#ffffff,stroke:#e4e8f0,stroke-width:2px,color:#132238;
      classDef highRisk fill:#fde6f9,stroke:#eb95e0,stroke-width:3px,color:#8a2b80,font-weight:bold;
      classDef lowRisk fill:#fdf8fc,stroke:#f5c5e5,stroke-width:3px,color:#a5479a,font-weight:bold;
      classDef criticalRisk fill:#eb95e0,stroke:#d9449f,stroke-width:4px,color:#ffffff,font-weight:bold;
      
      class Node3,Node6,Node11,Node14 highRisk
      class Node4,Node10,Node13 lowRisk
      class Node7 criticalRisk
    """
    
    st.components.v1.html(
        f"""
        <pre class="mermaid" style="background: white;">
        {mermaid_code}
        </pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ 
                startOnLoad: true,
                theme: 'base',
                themeVariables: {{
                    fontFamily: 'Calibri',
                    primaryColor: '#ffffff',
                    edgeLabelBackground: '#ffffff',
                    tertiaryColor: '#f8fbff'
                }}
            }});
        </script>
        """,
        height=800,
        scrolling=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_risk_predictor() -> None:
    default_row = cleaned_df.median(numeric_only=True).to_dict()
    default_cats = {
        "Sex": cleaned_df["Sex"].mode().iloc[0],
        "ChestPainType": cleaned_df["ChestPainType"].mode().iloc[0],
        "RestingECG": cleaned_df["RestingECG"].mode().iloc[0],
        "ExerciseAngina": cleaned_df["ExerciseAngina"].mode().iloc[0],
        "ST_Slope": cleaned_df["ST_Slope"].mode().iloc[0],
    }

    with st.container(border=True):
        render_card_title("Risk Predictor", "Enter a patient profile to score screening risk from both models.")
        with st.form("risk_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", 18, 100, int(default_row["Age"]))
                sex = st.selectbox("Sex", sorted(cleaned_df["Sex"].unique().tolist()), index=sorted(cleaned_df["Sex"].unique().tolist()).index(default_cats["Sex"]))
                chest_pain = st.selectbox("ChestPainType", sorted(cleaned_df["ChestPainType"].unique().tolist()), index=sorted(cleaned_df["ChestPainType"].unique().tolist()).index(default_cats["ChestPainType"]))
                resting_bp = st.number_input("RestingBP", 60, 250, int(default_row["RestingBP"]))
            with c2:
                cholesterol = st.number_input("Cholesterol", 0, 700, int(default_row["Cholesterol"]))
                fasting_bs = st.selectbox("FastingBS", [0, 1], index=int(default_row["FastingBS"]))
                resting_ecg = st.selectbox("RestingECG", sorted(cleaned_df["RestingECG"].unique().tolist()), index=sorted(cleaned_df["RestingECG"].unique().tolist()).index(default_cats["RestingECG"]))
                max_hr = st.number_input("MaxHR", 50, 250, int(default_row["MaxHR"]))
            with c3:
                exercise_angina = st.selectbox("ExerciseAngina", sorted(cleaned_df["ExerciseAngina"].unique().tolist()), index=sorted(cleaned_df["ExerciseAngina"].unique().tolist()).index(default_cats["ExerciseAngina"]))
                oldpeak = st.number_input("Oldpeak", -5.0, 10.0, float(default_row["Oldpeak"]))
                st_slope = st.selectbox("ST_Slope", sorted(cleaned_df["ST_Slope"].unique().tolist()), index=sorted(cleaned_df["ST_Slope"].unique().tolist()).index(default_cats["ST_Slope"]))

            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            input_df = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "Sex": sex,
                        "ChestPainType": chest_pain,
                        "RestingBP": resting_bp,
                        "Cholesterol": cholesterol,
                        "FastingBS": fasting_bs,
                        "RestingECG": resting_ecg,
                        "MaxHR": max_hr,
                        "ExerciseAngina": exercise_angina,
                        "Oldpeak": oldpeak,
                        "ST_Slope": st_slope,
                    }
                ]
            )

            if input_df["RestingBP"].iloc[0] == 0:
                input_df["RestingBP"] = cleaned_df["RestingBP"].median()
            if input_df["Cholesterol"].iloc[0] == 0:
                input_df["Cholesterol"] = cleaned_df["Cholesterol"].median()

            dt_prob = dt_model.predict_proba(input_df)[0][1]
            rf_prob = rf_model.predict_proba(input_df)[0][1]
            score = (dt_prob + rf_prob) / 2

            result_cols = st.columns(4)
            result_cols[0].metric("Decision Tree", f"{dt_prob:.1%}", delta=f"{dt_prob - 0.5:+.1%} vs neutral")
            result_cols[1].metric("Random Forest", f"{rf_prob:.1%}", delta=f"{rf_prob - 0.5:+.1%} vs neutral")
            result_cols[2].metric("Combined Score", f"{score:.1%}", delta=model_probability_label(score))
            result_cols[3].metric("Screening Priority", "High" if score >= 0.45 else "Lower")

            st.info(f"Primary screening probability is {rf_prob:.1%} from the Random Forest model. The combined score is {score:.1%}.")


# ── Page composers ─────────────────────────────────────────────────────────────
def render_overview_page(filtered_df: pd.DataFrame) -> None:
    left, right = st.columns([1.1, 0.9])
    with left:
        with st.container(border=True):
            render_card_title("Operational Summary", "Board-facing snapshot of the cleaned dataset.")
            st.write(filtered_df.head())
            st.write(filtered_df.describe().T)
    with right:
        render_donut_chart(filtered_df)


def render_eda_page(filtered_df: pd.DataFrame) -> None:
    # Numeric distributions section
    render_numeric_distributions(filtered_df)
    render_additional_numeric_distributions(filtered_df)
    
    # Correlation analysis
    render_correlation_heatmap(filtered_df)
    
    # Categorical patterns section
    render_categorical_patterns(filtered_df)
    render_sex_distribution(filtered_df)
    
    left, right = st.columns(2)
    with left:
        render_ecg_patterns(filtered_df)
    with right:
        render_fasting_bs_patterns(filtered_df)
    
    render_exercise_angina_patterns(filtered_df)
    render_st_slope_patterns(filtered_df)
    
    # Multivariate analysis
    render_multivariate_scatter(filtered_df)


def render_dashboard(page: str, filtered_df: pd.DataFrame) -> None:
    render_hero_shell()
    render_kpi_grid()
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    if page == "Overview":
        render_overview_page(filtered_df)
    elif page == "EDA":
        render_eda_page(filtered_df)
    elif page == "Association Rules":
        render_association_rules(filtered_df)
    elif page == "Risk Predictor":
        render_risk_predictor()
    elif page == "Model Comparison":
        render_model_comparison()


# ── Entry point ────────────────────────────────────────────────────────────────
st.markdown('<div class="app-shell">', unsafe_allow_html=True)
selected_page = render_sidebar()
# Only show filters on Overview and EDA pages
if selected_page in ["Overview", "EDA"]:
    filtered_df, _age_range, _sex = render_sidebar_filters()
else:
    filtered_df = cleaned_df
render_dashboard(selected_page, filtered_df)
st.markdown('</div>', unsafe_allow_html=True)

inject_nav_icon_fix()
