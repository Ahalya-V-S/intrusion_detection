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

#Styling configuration
st.markdown("""
    <style>
        /* Page Layout */
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
        }
        
        /* Styling for sidebar */
        .sidebar .sidebar-content {
            background-color: #f4f7f9;
            border-radius: 10px;
            padding: 20px;
            width: 250px; 
        }
        
        .css-1d391kg {  
            /* Adjusts sidebar width container */
            width: 300px !important;
        }
        
        /* Box Style for Sidebar Navigation */
        .nav-box {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            width: 300px;
            margin-bottom: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        
        .nav-box:hover {
            background-color: #e8f0fe;
        }

        .stButton>button {
            background-color: #ffffff;
            color: black;
            border-radius: 5px;
            padding: 12px 24px;
            font-size: 16px;
            border: 2px solid black;
            width: 100%; 
            text-align: center; 
            display: block; 
        }
        .stButton>button:hover {
            background-color: #000000;
            color: white;
        }

        .button-container {
            width: 200px;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('intrusionDetection_model.pkl')   

encoding_dict = {
    "protocol_type": LabelEncoder(),
    "service": LabelEncoder(),
    "flag": LabelEncoder()
}
scaler = StandardScaler()

# Preprocessing function 
def preprocess_data(df):
    # Encode categorical features using predefined LabelEncoders
    for col, encoder in encoding_dict.items():
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col])

    # Select only the relevant features found using RFE (refer the notebook)
    selected_features = ['protocol_type',  'service', 'flag', 'src_bytes', 'dst_bytes', 'hot','logged_in', 'count', 'srv_count', 'srv_serror_rate', 'same_srv_rate', 'diff_srv_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate']
            
    df = df[selected_features]

    # Scale the features
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=selected_features)

    return df

# Sidebar for navigation using buttons
st.markdown('<div class="button-container">', unsafe_allow_html=True)
home_button = st.sidebar.button("Home")
prediction_button = st.sidebar.button("Model Prediction")
evaluation_button = st.sidebar.button("Evaluation Metrics")
st.markdown('</div>', unsafe_allow_html=True)

# Create a session state to keep track of the active page
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Update the session state based on the button clicks
if home_button:
    st.session_state.page = "Home"
elif prediction_button:
    st.session_state.page = "Model Prediction"
elif evaluation_button:
    st.session_state.page = "Evaluation Metrics"

# Show content based on the selected page

if st.session_state.page == "Home":
    st.title("Welcome to the Anomaly Detection App")
    # Markdown about the info of the model
    st.write("""
An **Intrusion Detection System (IDS)** is a security application that continuously monitors network traffic, 
searching for known threats and suspicious or malicious activities. When any security risks or threats are 
detected, the IDS sends alerts to IT and security teams, helping them respond proactively to potential issues 
before they escalate.

This app allows you to leverage a trained machine learning model to analyze network traffic for possible anomalies, 
helping to secure your network by detecting patterns indicative of malicious activity. With a simple interface 
and powerful features, you can easily upload data, receive predictions, and analyze model performance metrics.

### Key Features

- **Model Prediction**: The core functionality of this IDS app is to provide **anomaly predictions** based on 
network traffic data. By uploading a CSV file containing network traffic details, you can leverage our trained 
machine learning model to detect anomalies. This feature makes it simple for users to analyze bulk traffic data 
and identify potential security threats.

- **Evaluation Metrics**: This feature allows users to evaluate the performance of the IDS model based on 
ground-truth labels provided in the uploaded dataset. It calculates metrics such as **accuracy, precision, 
recall, and F1-score**, which are essential to understanding the effectiveness of the IDS model. Visualizations 
like the confusion matrix are also included for a clear view of the model’s performance on different classes 
(anomaly or normal).

### How to Use the App

1. **Home Page**: This section gives an overview of the application, including its purpose and features. 
   Start here to understand what the app offers.

2. **Model Prediction**:
    - Go to the sidebar and click "Model Prediction".
    - Upload your network traffic CSV file (in the required format).
    - The model will process your data and provide anomaly predictions, indicating potential security threats.
    - Visualizations, including distribution of predictions, give a quick look at possible risks in your data.

3. **Evaluation Metrics**:
    - Click on "Evaluation Metrics" in the sidebar.
    - Upload a CSV file containing both network traffic data and true labels (ground-truth values).
    - This section provides a confusion matrix and classification report, helping you assess the IDS model’s 
      accuracy in detecting malicious traffic. 

By using this web application, you can get real-time insights into network security, identify suspicious patterns in 
network traffic, and evaluate the reliability of the IDS model for further improvements.
""")


elif st.session_state.page == "Model Prediction":
    st.title("Intrusion Detection Model Prediction")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load the model
        model = load_model()

        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            df1 = df.copy()
            st.write("Uploaded CSV Data:")
            st.write(df.head())

            # Check if required columns exist in the data
            required_columns = ['protocol_type',  'service', 'flag', 'src_bytes', 'dst_bytes', 'hot','logged_in', 'count', 'srv_count', 'srv_serror_rate', 'same_srv_rate', 'diff_srv_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate']
            
            if not all(col in df.columns for col in required_columns):
                st.error("Uploaded CSV is missing required columns.")
            else:
                # Preprocess the data
                df_processed = preprocess_data(df)
                st.write("Preprocessed Data (selected and scaled features):")
                st.write(df_processed.head())

                # Make predictions
                predictions = model.predict(df_processed)
                df1['Predictions'] = np.where(predictions == 1, 'Normal', 'Anomaly')
                st.write("Prediction Results:")
                st.write(df1[['Predictions']].head())

                 # Plot prediction distribution-The histogram shows discrete counts, while the KDE provides a continuous representation of the distribution.
                st.write("Prediction Distribution:")
                fig, ax = plt.subplots()
                sns.histplot(df1['Predictions'], ax=ax)
                st.pyplot(fig)

                # Pie chart to show proportion of Normal vs. Anomaly
                st.write("Prediction Proportion:")
                labels = df1['Predictions'].value_counts().index
                sizes = df1['Predictions'].value_counts().values

                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
                ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
                st.pyplot(fig)

                # Download predictions
                st.write("Download Predictions:")
                st.download_button(
                    label="Download CSV with Predictions",
                    data=df1.to_csv(index=False),
                    file_name="anomaly_prediction.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")

elif st.session_state.page == "Evaluation Metrics":
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
            st.write(df)

            # Check if required columns exist
            if 'class' not in df.columns:
                st.error("Uploaded CSV is missing the 'class' column for ground truth.")
            else:
                # Preprocess the data
                df_processed = preprocess_data(df)
                st.write("Preprocessed Data (selected and scaled features):")
                st.write(df_processed.head())

                # Make predictions
                predictions = model.predict(df_processed)
                df['Predictions'] = predictions

                # Encode 'class' column to match 'y_pred' (numeric) format
                label_encoder = LabelEncoder()
                y_true = label_encoder.fit_transform(df['class'])  # Encode true labels to numeric
                y_pred = predictions  # Predictions are already numeric

                # Display Confusion Matrix and Classification Report
                st.write("### Analysis and Evaluation Metrics")
                st.write("Confusion Matrix:")

                # Confusion Matrix
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
