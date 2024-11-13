import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Set the page configuration
st.set_page_config(page_title="Anomaly Detection", page_icon="🌐")

# Function to load model
@st.cache_resource
def load_model():
    return joblib.load('/workspaces/intrusion_detection/XG_Boost_model.pkl')
# Encoding dictionary and StandardScaler
encoding_dict = {
    "protocol_type": LabelEncoder(),
    "service": LabelEncoder(),
    "flag": LabelEncoder()
}

scaler = StandardScaler()

# Preprocessing function to mirror training steps
def preprocess_data(df):
    for col, encoder in encoding_dict.items():
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col])
    
    if 'num_outbound_cmds' in df.columns:
        df = df.drop(columns=['num_outbound_cmds'])

    selected_features = [
        'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'hot',
        'logged_in', 'count', 'srv_count', 'same_srv_rate', 'diff_srv_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate'
    ]
    df = df[selected_features]

    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=selected_features)

    return df

# Toggle Theme Functionality
def set_theme():
    theme = st.selectbox("Choose Theme", ("Light", "Dark"))
    if theme == "Dark":
        st.markdown("""
            <style>
                body {
                    background-color: #1e1e1e;
                    color: #fff;
                }
                .sidebar {
                    background-color: #333;
                    color: #fff;
                }
                .sidebar .sidebar-content {
                    background-color: #444;
                }
                .nav-box {
                    background-color: #555;
                    color: #fff;
                }
                .nav-box:hover {
                    background-color: #777;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                body {
                    background-color: #f0f2f6;
                    color: #333;
                }
                .sidebar {
                    background-color: #f4f7f9;
                    color: #333;
                }
                .sidebar .sidebar-content {
                    background-color: #f4f7f9;
                }
                .nav-box {
                    background-color: #ffffff;
                    color: #333;
                }
                .nav-box:hover {
                    background-color: #e8f0fe;
                }
            </style>
        """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
home_button = st.sidebar.markdown('<div class="nav-box">Home</div>', unsafe_allow_html=True)
prediction_button = st.sidebar.markdown('<div class="nav-box">Model Prediction</div>', unsafe_allow_html=True)
evaluation_button = st.sidebar.markdown('<div class="nav-box">Evaluation Metrics</div>', unsafe_allow_html=True)

# Create a session state to keep track of the active page
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Detect click events to change the page content
if "Home" in st.session_state and st.sidebar.button("Home"):
    st.session_state.page = "Home"
elif "Model Prediction" in st.session_state and st.sidebar.button("Model Prediction"):
    st.session_state.page = "Model Prediction"
elif "Evaluation Metrics" in st.session_state and st.sidebar.button("Evaluation Metrics"):
    st.session_state.page = "Evaluation Metrics"

# Set the theme
set_theme()

# Show content based on the selected page
if st.session_state.page == "Home":
    st.title("Welcome to the Anomaly Detection App")
    st.write("""
    This app allows you to upload a CSV file and get predictions on network anomaly detection. 
    It uses a trained machine learning model to predict network traffic anomalies and provide evaluation metrics. 
    Choose the options from the sidebar to interact with the app.
    """)

elif st.session_state.page == "Model Prediction":
    st.title("Intrusion Detection Model Prediction")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        model = load_model()

        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV Data:")
            st.write(df.head())

            df_processed = preprocess_data(df)
            st.write("Preprocessed Data (selected and scaled features):")
            st.write(df_processed.head())

            predictions = model.predict(df_processed)
            df['Predictions'] = predictions

            st.write("Prediction Results:")
            st.write(df[['Predictions']].head())

            st.write("Prediction Distribution:")
            fig, ax = plt.subplots()
            sns.histplot(df['Predictions'], kde=True, ax=ax)
            st.pyplot(fig)

            st.write("Download Predictions:")
            df.to_csv("predictions_with_data.csv", index=False)
            st.download_button(
                label="Download CSV with Predictions",
                data=df.to_csv(index=False),
                file_name="predictions_with_data.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")

elif st.session_state.page == "Evaluation Metrics":
    st.title("Evaluation Metrics")
    
    uploaded_file = st.file_uploader("Upload a CSV file with ground truth labels", type=["csv"])
    
    if uploaded_file is not None:
        model = load_model()

        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV Data:")
            st.write(df.head())

            df_processed = preprocess_data(df)
            st.write("Preprocessed Data (selected and scaled features):")
            st.write(df_processed.head())

            predictions = model.predict(df_processed)
            df['Predictions'] = predictions

            if 'class' in df.columns:
                st.write("### Analysis and Evaluation Metrics")
                y_true = df['class']
                y_pred = predictions

                st.write("Confusion Matrix:")
                conf_matrix = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                st.write("Classification Report:")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.write(pd.DataFrame(report).transpose())

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
