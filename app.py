
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    matthews_corrcoef
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ML Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)




# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ASSET_DIR = BASE_DIR / "asset"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

STYLE_FILE = ASSET_DIR / "style.css"

DEFAULT_TEST_PATH = DATA_DIR / "pendigits.tes"
DEFAULT_TRAIN_PATH = DATA_DIR / "pendigits.tra"

# --------------------------------------------------
# LOAD EXTERNAL CSS
# --------------------------------------------------

if STYLE_FILE.exists():
    with open(STYLE_FILE, "r", encoding="utf-8") as css:
        st.markdown(
            f"<style>{css.read()}</style>",
            unsafe_allow_html=True
        )
else:
    st.warning("CSS file not found: asset/style.css")

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "K-Nearest Neighbors": MODEL_DIR / "knn.pkl",
    "Gaussian Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Support Vector Machine": MODEL_DIR / "svm.pkl"
}

TARGET_COLUMN_DEFAULT = "digit"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

with st.container(border=True, key="assignment_header"):

    st.markdown(
        """
        <div style="
            text-align: center;
            width: 100%;
            font-size: 15px;
            font-weight: 500;
            color: #808080;
            padding: 8px 0;
            margin-left: -40px;
        ">
            BITS PILANI | M.Tech AI & ML | AIMLZG565 - Machine Learning
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div style='text-align:right; font-size: 18px;font-weight: 700;color: #172554;'><b>BITS ID:</b> 2026AK05722</div>",
                    unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='text-align:left; font-size: 18px;font-weight: 700;'><b>Assignment:</b> 2</div>",
                    unsafe_allow_html=True)

    st.markdown(
        '<div class="main-title">🔢 ✍️ Handwritten Digit Recognition</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Multiclass classification of handwritten digits from 0 to 9 "
        "using the PenDigits dataset."
    )
 
    
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------


with st.sidebar:

    st.markdown("## 🧪 Experiment Setup")

    st.markdown("### STEP 01 | Test Dataset")

    # Download sample CSV - first
    if DEFAULT_TEST_PATH.exists():

        feature_names = [f"feature_{i}" for i in range(1, 17)]
        column_names = feature_names + ["digit"]

        sample_df = pd.read_csv(
            DEFAULT_TEST_PATH,
            header=None,
            names=column_names
        )

        st.download_button(
            "⬇️ Download Sample Test CSV",
            data=sample_df.to_csv(index=False),
            file_name="pendigits_test.csv",
            mime="text/csv",
            use_container_width=False
        )

    else:
        st.warning("Default dataset pendigits.tes not found.")

    # Upload test CSV
    uploaded_file = st.file_uploader(
        "Upload test CSV file",
        type=["csv", "tes"],
        help="Upload test data containing 16 features and digit target."
    )

    # Default dataset button - below upload information
    if DEFAULT_TEST_PATH.exists():

        if st.button(
            "📂 Use Default Test Dataset",
            use_container_width=False
        ):
            st.session_state["use_default_test"] = True

    # Uploaded file takes priority
    if uploaded_file is not None:
        st.session_state["use_default_test"] = False

    use_default_test = st.session_state.get(
        "use_default_test", False
    )

    # Target column - Dynamic selection
    st.markdown("### STEP 02 | Target Column")

    if uploaded_file is not None and uploaded_file.name.lower().endswith(".csv"):

        uploaded_file.seek(0)
        target_options = list(pd.read_csv(uploaded_file, nrows=0).columns)
        uploaded_file.seek(0)

    else:
        target_options = ["digit"]

    default_index = (
        target_options.index("digit")
        if "digit" in target_options
        else len(target_options) - 1
    )

    target_column = st.selectbox(
        "Select target column",
        options=target_options,
        index=default_index
    )

    # Model selection
    st.markdown("### STEP 03 | Model Selection")

    available_models = {
        name: path
        for name, path in MODEL_FILES.items()
        if path.exists()
    }

    if available_models:
        def sync_sidebar_model():
            st.session_state["dashboard_model_select"] = (
                st.session_state["sidebar_model_select"]
            )

        selected_model = st.selectbox(
            "Choose a model",
            list(available_models.keys()),
            key="sidebar_model_select",
            on_change=sync_sidebar_model
        )
    else:
        selected_model = None
        st.warning("No model files found in the model folder.")

    compare_all = st.checkbox(
        "Compare all available models",
        value=False
    )

    st.markdown(
        "<hr style='margin: 8px 0; padding: 0; border: none; "
        "border-top: 1px solid #d1d5db;'>",
        unsafe_allow_html=True
    )

    st.caption("M.Tech AI & ML | 📊 Model Evaluation Project")

# --------------------------------------------------
# DATASET REFERENCE LINK
# --------------------------------------------------

st.markdown(
    """
    <div style="margin-bottom:20px;">
        <a href="https://archive.ics.uci.edu/dataset/81/pen-based+recognition+of+handwritten+digits"
           target="_blank"
           style="
               color: #172554;
               font-weight: 700;
               font-size: 14px;
               text-decoration: none;
           ">
            📁 <span style="color: #138B80;font-weight: 700;font-size: 16px;">Dataset: </span> 
                            <span style="text-decoration: underline;"> Pen-Based Recognition of Handwritten Digits
                            (PenDigits) | UCI Machine Learning Repository</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# DASHBOARD CONTENT
# --------------------------------------------------


if uploaded_file is None and not use_default_test:

    st.markdown("#### 🚀 Getting Started")

    st.info(
        "Upload your test dataset from the sidebar to start evaluating "
        "your trained machine learning models."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">STEP 01</div>
            <div class="metric-value">📁 Upload</div>
            <p>Upload your test CSV dataset.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">STEP 02</div>
            <div class="metric-value">🤖 Select</div>
            <p>Choose a trained ML model.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">STEP 03</div>
            <div class="metric-value">📊 Evaluate</div>
            <p>View performance and predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# --------------------------------------------------
# READ DATASET
# --------------------------------------------------

feature_names = [f"feature_{i}" for i in range(1, 17)]
column_names = feature_names + ["digit"]

try:

    if uploaded_file is not None:

        uploaded_file.seek(0)

        if uploaded_file.name.lower().endswith(".tes"):

            df = pd.read_csv(
                uploaded_file,
                header=None,
                names=column_names
            )

        else:
            df = pd.read_csv(uploaded_file)

    elif use_default_test:

        st.info("📁 Dataset Source: Default PenDigits Test Dataset")

        if not DEFAULT_TEST_PATH.exists():
            st.error("Default dataset pendigits.tes not found.")
            st.stop()

        df = pd.read_csv(
            DEFAULT_TEST_PATH,
            header=None,
            names=column_names
        )

    else:
        st.warning("Please select a test dataset.")
        st.stop()

except Exception as e:

    st.error(f"Unable to read dataset: {e}")
    st.stop()




# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.markdown('<div class="section-title">📁 Dataset Overview</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Records", f"{df.shape[0]:,}")

with c2:
    st.metric("Total Columns", df.shape[1])

with c3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

with c4:
    st.metric("Duplicate Rows", int(df.duplicated().sum()))


with st.expander("Preview Test Dataset", expanded=True):

    preview_df = df.head(10)

    styled_dataset = preview_df.style.set_table_attributes(
        'class="dataset-table"'
    )

    st.markdown(
        '<div class="dataset-table-wrap">'
        + styled_dataset.to_html()
        + '</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# VALIDATE DATASET
# --------------------------------------------------

if target_column not in df.columns:
    st.error(
        f"Target column '{target_column}' not found. "
        "Enter the correct target column name in the sidebar."
    )
    st.stop()

if not available_models:
    st.error("Please add trained model files to the model folder.")
    st.stop()

feature_names = [f"feature_{i}" for i in range(1, 17)]

missing_features = [
    col for col in feature_names if col not in df.columns
]

if missing_features:
    st.error(f"Missing input features: {missing_features}")
    st.stop()

if df[feature_names + [target_column]].isnull().values.any():
    st.error("Dataset contains missing values. Please clean the dataset.")
    st.stop()

if not all(
    pd.api.types.is_numeric_dtype(df[col])
    for col in feature_names
):
    st.error("All 16 input features must be numeric.")
    st.stop()

X_test = df[feature_names]
y_test = df[target_column]


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    # Multiclass AUC (One-vs-Rest)
    auc = np.nan

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test)

            auc = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="weighted",
                labels=model.classes_
            )

        except ValueError:
            auc = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),

        "AUC": auc,

        "Precision": precision_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),

        "MCC": matthews_corrcoef(y_test, predictions)
    }

    return predictions, metrics

# --------------------------------------------------
# MAIN DASHBOARD MODEL SELECTION
# --------------------------------------------------

def sync_dashboard_model():
    st.session_state["sidebar_model_select"] = (
        st.session_state["dashboard_model_select"]
    )

dashboard_model = st.selectbox(
    "Choose a model",
    list(available_models.keys()),
    key="dashboard_model_select",
    on_change=sync_dashboard_model
)

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown(
        f"**Selected Model:** {dashboard_model}"
    )

with col2:
    success_placeholder = st.empty()

# Decide which models to evaluate
if compare_all:
    models_to_evaluate = list(available_models.keys())
else:
    models_to_evaluate = [dashboard_model]

if not models_to_evaluate:
    st.warning("Select at least one available model.")
    st.stop()


results = {}
predictions_by_model = {}

try:

    for model_name in models_to_evaluate:

        model = load_model(available_models[model_name])

        predictions, metrics = evaluate_model(
            model, X_test, y_test
        )

        results[model_name] = metrics
        predictions_by_model[model_name] = predictions

except Exception as e:

    st.error(f"Model evaluation failed: {e}")
    st.info(
        "Check that your model uses the same input features and "
        "preprocessing as the uploaded test dataset."
    )
    st.stop()

success_placeholder.success(
    "Model evaluation completed successfully.",
    icon="✅"
)

# --------------------------------------------------
# MODEL PERFORMANCE CARDS
# --------------------------------------------------

st.markdown(
    "<hr style='margin: 1px 0; padding: 0; border: none; "
    "border-top: 1px solid #d1d5db;'>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODEL PERFORMANCE CARDS
# --------------------------------------------------

card_model = (
    dashboard_model
    if dashboard_model in results
    else models_to_evaluate[0]
)

metrics = results[card_model]

metric_items = [
    ("ACCURACY", "Accuracy"),
    ("AUC", "AUC"),
    ("PRECISION", "Precision"),
    ("RECALL", "Recall"),
    ("F1", "F1 Score"),
    ("MCC", "MCC")
]

rankings = {}

for label, key in metric_items:
    valid = {
        name: result[key]
        for name, result in results.items()
        if pd.notna(result.get(key, np.nan))
    }

    rankings[key] = {
        name: rank + 1
        for rank, name in enumerate(
            sorted(valid, key=valid.get, reverse=True)
        )
    }

columns = st.columns(6)

for col, (label, key) in zip(columns, metric_items):

    value = metrics.get(key, np.nan)
    rank = rankings[key].get(card_model)

    if pd.isna(value):
        rank_text = f"{label} unavailable"
    elif not compare_all:
        rank_text = ""
    elif rank == 1:
        rank_text = f"Best of {len(results)}"
    else:
        suffix = "nd" if rank == 2 else "rd" if rank == 3 else "th"
        rank_text = f"{rank}{suffix} of {len(results)}"

    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">
                {"N/A" if pd.isna(value) else f"{value:.4f}"}
            </div>
            {f'<div style="font-size:11px;color:#64748b;margin-top:10px;">{rank_text}</div>' if rank_text else ''}
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# CONFUSION MATRIX & CLASSIFICATION REPORT
# --------------------------------------------------

st.markdown(
    "<hr style='margin: 8px 0; padding: 0; border: none; "
    "border-top: 1px solid #d1d5db;'>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🎯 Model Analysis</div>',
    unsafe_allow_html=True
)

if compare_all:
    analysis_model = st.selectbox(
        "Select model for detailed analysis",
        models_to_evaluate,
        key="analysis_model_select"
    )
else:
    analysis_model = models_to_evaluate[0]

analysis_predictions = predictions_by_model[analysis_model]

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Confusion Matrix")

    labels = sorted(
        set(y_test.tolist()) |
        set(analysis_predictions.tolist()),
        key=str
    )

    cm = confusion_matrix(
        y_test,
        analysis_predictions,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title("Actual vs Predicted Values")
    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


with col2:
    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        analysis_predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    # Correct accuracy row
    report_df.loc["accuracy", "precision"] = np.nan
    report_df.loc["accuracy", "recall"] = np.nan
    report_df.loc["accuracy", "f1-score"] = report["accuracy"]
    report_df.loc["accuracy", "support"] = len(y_test)

    # Show complete class labels
    report_df.index.name = "Class"
    report_df = report_df.reset_index()

    styled_report = report_df.style.format({
        "precision": "{:.3f}",
        "recall": "{:.3f}",
        "f1-score": "{:.3f}",
        "support": "{:.0f}"
    }, na_rep="-").set_table_attributes(
        'class="report-table"'
    )

    st.markdown(
        styled_report.to_html(),
        unsafe_allow_html=True
    )    


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

if compare_all:

    st.markdown(
        "<hr style='margin: 8px 0; padding: 0; border: none; "
        "border-top: 1px solid #d1d5db;'>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🏆 Model Comparison</div>',
        unsafe_allow_html=True
    )

    comparison_df = pd.DataFrame(results).T

    st.dataframe(
        comparison_df.style.format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("#### Performance Comparison")

    chart_df = comparison_df.reset_index().rename(
        columns={"index": "Model"}
    )

    chart_df = chart_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    chart_data = chart_df.pivot(
        index="Model",
        columns="Metric",
        values="Score"
    )

    st.bar_chart(chart_data, use_container_width=True)


# --------------------------------------------------
# DOWNLOAD PREDICTIONS
# --------------------------------------------------

st.markdown(
    "<hr style='margin: 8px 0; padding: 0; border: none; "
    "border-top: 1px solid #d1d5db;'>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">⬇️ Download Predictions</div>',
    unsafe_allow_html=True
)

download_df = df.copy()

for model_name, predictions in predictions_by_model.items():
    download_df[f"{model_name}_Prediction"] = predictions

csv_data = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Evaluation Results (CSV)",
    data=csv_data,
    file_name="model_evaluation_results.csv",
    mime="text/csv"
)
