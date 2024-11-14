# Intrusion Detection System (IDS) Using Streamlit and XGBoost

## Project Overview
This project is a **Intrusion Detection System (IDS)** web application built using Streamlit. The application leverages an XGBoost model to classify network traffic data as either "Normal" or "Anomalous." It provides an interactive interface for uploading network traffic data, performing anomaly predictions, and evaluating model performance through metrics and visualizations.

## Features

- **Model Prediction**: Users can upload a CSV file of network traffic data, and the IDS will classify each entry as either "Normal" or "Anomalous."
- **Visualization**: Visualizes the distribution of predictions through histograms and pie charts.
- **Evaluation Metrics**: Evaluates model performance on ground truth labels, displaying a confusion matrix and detailed classification report.

## Technology Stack

- **Front-End**: Streamlit for the interactive UI
- **Back-End**: Python, with Scikit-Learn and XGBoost for machine learning
- **Data Visualization**: Matplotlib and Seaborn for visualizations
- **Packet Preprocessing**: Scikit-Learn for encoding and scaling

## Installation

### Prerequisites

- Python 3.7 or later
- [Streamlit](https://streamlit.io/) for building the web app
- [XGBoost](https://xgboost.readthedocs.io/) for the anomaly detection model
- Additional Python packages listed in `requirements.txt`

### Steps

1.  **Install required libraries**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the application**:

    ```bash
    streamlit run app.py
    ```

4. **Open the app**: The Streamlit app will open in your default browser at `http://localhost:8501`.

## Usage

### Step 1: Home Page

The Home page provides an overview of the application and instructions on how to use each feature.

### Step 2: Model Prediction

1. Go to the **Model Prediction** section via the sidebar.
2. Upload a CSV file containing network traffic data with the required features.
3. The model will classify each record as either "Normal" or "Anomaly."
4. Visualizations including a histogram and a pie chart display the distribution of normal vs. anomalous traffic.
5. Download the prediction results as a CSV file if needed.

### Step 3: Evaluation Metrics

1. Go to the **Evaluation Metrics** section in the sidebar.
2. Upload a CSV file containing network traffic data with ground truth labels in the `class` column.
3. The app displays a confusion matrix and classification report to evaluate the model’s performance.

## Data Requirements

The uploaded CSV files for prediction and evaluation should contain the following columns:

- `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, `hot`, `logged_in`, `count`, `srv_count`
- `same_srv_rate`, `diff_srv_rate`, `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`
- `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`, `dst_host_serror_rate`
- `dst_host_srv_serror_rate`, `dst_host_srv_rerror_rate`
  
For the **Evaluation Metrics** section, a `class` column containing ground-truth labels is also required.

## Configuration

### Sidebar Navigation

The sidebar contains buttons for each of the three sections of the app:
- **Home**: Provides an overview and instructions.
- **Model Prediction**: Allows for data upload and real-time anomaly detection.
- **Evaluation Metrics**: Allows for performance evaluation based on ground-truth data.

### Preprocessing

- **Encoding**: `protocol_type`, `service`, and `flag` features are encoded using LabelEncoder.
- **Scaling**: Selected features are scaled using StandardScaler for model compatibility.
