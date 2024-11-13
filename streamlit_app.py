import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

# Streamlit app
# Set the page configuration
st.set_page_config(page_title="Anomaly Detection", page_icon="🔍")

st.title("Anomaly Detection Model Prediction & Analysis")
# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the model
    model = load_model()
    
    # Read the uploaded file
    try:
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
