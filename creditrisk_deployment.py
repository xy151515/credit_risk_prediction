import streamlit as st
import pandas as pd
import joblib
import json

# Load files
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scale.pkl')
selected_features = joblib.load('selected_features.pkl')

# Load evaluation metrics
with open('evaluation_metrics.json', 'r') as f:
    evaluation_metrics = json.load(f)

# App title
st.title("Credit Risk Prediction App")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose an option:",
    ["Home", "Upload Data", "Performance Metrics"]
)

# Home section
if app_mode == "Home":
    st.header("Welcome to the Credit Risk Prediction App")
    st.write("""
        This application predicts the likelihood of loan default using a pre-trained machine learning model.
        Features include:
        - Uploading new loan application data for prediction
        - Viewing model performance metrics
        - Understanding feature importance
    """)

# Upload Data section
elif app_mode == "Upload Data":
    st.header("Upload Loan Data for Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)

        # Validate and preprocess data
        if all(feature in new_data.columns for feature in selected_features):
            new_data = new_data[selected_features]
            new_data_scaled = scaler.transform(new_data)

            # Predictions
            predictions = model.predict(new_data_scaled)
            probabilities = model.predict_proba(new_data_scaled)[:, 1]

            # Display results
            new_data['Prediction'] = predictions
            new_data['Probability'] = probabilities

            st.subheader("Prediction Results")
            st.write(new_data)

            st.download_button(
                label="Download Predictions",
                data=new_data.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("The uploaded file does not contain all required features.")

# Performance Metrics section
elif app_mode == "Performance Metrics":
    st.header("Model Performance Metrics")
    st.metric("Accuracy", f"{evaluation_metrics['accuracy']:.4f}")
    st.metric("Precision", f"{evaluation_metrics['precision']:.4f}")
    st.metric("Recall", f"{evaluation_metrics['recall']:.4f}")
    st.metric("AUC-ROC", f"{evaluation_metrics['auc_roc']:.4f}")

    st.write("For detailed insights, refer to the model evaluation JSON file.")

