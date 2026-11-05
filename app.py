
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
    ConfusionMatrixDisplay
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
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    .main-title {
        font-size: 30px;
        font-weight: 750;
        color: #172554;
        margin-bottom: 4px;
    }

    .sub-title {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #172554;
        font-size: 28px;
        font-weight: 750;
        margin-top: 7px;
    }

    .section-title {
        color: #172554;
        font-size: 20px;
        font-weight: 700;
        margin-top: 15px;
    }

    div[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    .st-key-assignment_header {
        text-align: center;
    }

    .st-key-assignment_header [data-testid="stMarkdownContainer"] {
        text-align: center;
    }
    .block-container {
        padding-top: 4rem !important;
        padding-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

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

    st.caption(
        "BITS PILANI | M.Tech AI & ML | AIMLZG565 - Machine Learning"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div style='text-align:right'><b>BITS ID:</b> 2026AK05722</div>",
                    unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='text-align:left'><b>Assignment:</b> II</div>",
                    unsafe_allow_html=True)

    st.markdown(
        '<div class="main-title">🔢 ✍️ Handwritten Digit Recognition</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Multiclass classification of handwritten digits from 0 to 9 "
        "using the PenDigits dataset."
    )

    st.caption(
        "📁 Dataset: Pen-Based Recognition of Handwritten Digits "
        "(PenDigits) | UCI Machine Learning Repository"
    )

st.markdown(
    "<hr style='margin: 1px 0; padding: 0; border: none; "
    "border-top: 1px solid #d1d5db;'>",
    unsafe_allow_html=True
)



# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🧪 Experiment Setup")

    st.markdown("### STEP 01 | Test Dataset")

    DEFAULT_TEST_PATH = BASE_DIR / "pendigits.tes"

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

    # Target column
    st.markdown("### STEP 02 | Target Column")

    target_column = "digit"

    st.text_input(
        "Target column name",
        value=target_column,
        disabled=True
    )

    # Model selection
    st.markdown("### STEP 03 | Model Selection")

    available_models = {
        name: path
        for name, path in MODEL_FILES.items()
        if path.exists()
    }

    if available_models:
        selected_model = st.selectbox(
            "Choose a model",
            list(available_models.keys())
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
    st.dataframe(df.head(10), use_container_width=True)


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

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="weighted",
            zero_division=0
        )
    }

    return predictions, metrics


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

if compare_all:
    models_to_evaluate = list(available_models.keys())
else:
    models_to_evaluate = [selected_model] if selected_model else []


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

if not compare_all:

    metrics = results[models_to_evaluate[0]]

    c1, c2, c3, c4 = st.columns(4)

    metric_items = [
        ("Accuracy", metrics["Accuracy"]),
        ("Precision", metrics["Precision"]),
        ("Recall", metrics["Recall"]),
        ("F1 Score", metrics["F1 Score"])
    ]

    for col, (label, value) in zip(
        [c1, c2, c3, c4], metric_items
    ):

        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.2%}</div>
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
        models_to_evaluate
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

    st.dataframe(
        report_df.style.format("{:.3f}"),
        use_container_width=True,
        height=450
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
        comparison_df.style.format("{:.2%}"),
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

st.success("Model evaluation completed successfully.")

st.caption("Note: Models are evaluated on the uploaded test data. "
           "No model training is performed in this application.")
