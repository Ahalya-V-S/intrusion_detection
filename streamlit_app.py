import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set the page configuration
st.set_page_config(page_title="Anomaly Detection", page_icon="🌐")

# Custom CSS to improve UI appearance
st.markdown("""
    <style>
        /* Page Layout */
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            padding: 2rem;
        }

        /* Styling for sidebar */
        .sidebar .sidebar-content {
            background-color: #f4f7f9;
            border-radius: 10px;
            padding: 20px;
        }
        
        /* Card Style for Sidebar */
        .card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .card:hover {
            background-color: #e8f0fe;
            cursor: pointer;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 12px 24px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('/content/drive/MyDrive/ML_project/XG_Boost_model.pkl')  # Replace with your model path

# Encoding dictionary
encoding_dict = {
    "protocol_type": LabelEncoder(),
    "service": LabelEncoder(),
    "flag": LabelEncoder()
}

# StandardScaler instance for scaling
scaler = StandardScaler()

# Preprocessing function to mirror training steps
def preprocess_data(df):
    # Encode categorical features using predefined LabelEncoders
    for col, encoder in encoding_dict.items():
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col])
    
    # Drop 'num_outbound_cmds' if it exists in the dataframe
    if 'num_outbound_cmds' in df.columns:
        df = df.drop(columns=['num_outbound_cmds'])

    # Select only the relevant features
    selected_features = [
        'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'hot',
        'logged_in', 'count', 'srv_count', 'same_srv_rate', 'diff_srv_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate'
    ]
    df = df[selected_features]

    # Scale the features
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=selected_features)

    return df

# Sidebar for navigation with a card layout
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose an option", ["Home", "Model Prediction", "Evaluation Metrics"])

# Home page content
if app_mode == "Home":
    st.title("Welcome to the Anomaly Detection App")
    st.write("""
    This app allows you to upload a CSV file and get predictions on network anomaly detection. 
    It uses a trained machine learning model to predict network traffic anomalies and provide evaluation metrics. 
    Choose the options from the sidebar to interact with the app.
    """)

# Model Prediction page content
elif app_mode == "Model Prediction":
    st.title("Intrusion Detection Model Prediction")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load the model
        model = load_model()

        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV Data:")
            st.write(df.head())

            # Preprocess the data
            df_processed = preprocess_data(df)
            st.write("Preprocessed Data (selected and scaled features):")
            st.write(df_processed.head())

            # Make predictions
            predictions = model.predict(df_processed)
            df['Predictions'] = predictions

            st.write("Prediction Results:")
            st.write(df[['Predictions']].head())

            # Plot prediction distribution
            st.write("Prediction Distribution:")
            fig, ax = plt.subplots()
            sns.histplot(df['Predictions'], kde=True, ax=ax)
            st.pyplot(fig)

            # Option to download predictions
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

# Evaluation Metrics page content
elif app_mode == "Evaluation Metrics":
    st.title("Evaluation Metrics")
    
    # Upload CSV file for evaluation
    uploaded_file = st.file_uploader("Upload a CSV file with ground truth labels", type=["csv"])
    
    if uploaded_file is not None:
        # Load the model
        model = load_model()

        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV Data:")
            st.write(df.head())

            # Preprocess the data
            df_processed = preprocess_data(df)
            st.write("Preprocessed Data (selected and scaled features):")
            st.write(df_processed.head())

            # Make predictions
            predictions = model.predict(df_processed)
            df['Predictions'] = predictions

            # Display Confusion Matrix and Classification Report if ground truth is available
            if 'class' in df.columns:
                st.write("### Analysis and Evaluation Metrics")
                y_true = df['class']
                y_pred = predictions

                # Confusion Matrix
                st.write("Confusion Matrix:")
                conf_matrix = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                # Classification Report
                st.write("Classification Report:")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.write(pd.DataFrame(report).transpose())

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
