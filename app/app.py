import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import load_model, predict
from src.data_processing import load_preprocessor

# Set page config
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

# Title and description
st.title("🍄 Mushroom Classification")
st.markdown("""
This application uses machine learning to classify mushrooms based on their characteristics.
Upload your mushroom data or input the features manually to get predictions.
""")

# Load model and preprocessor
@st.cache_resource
def load_resources():
    model = load_model('models/model.pkl')
    preprocessor = load_preprocessor('models/preprocessor.pkl')
    return model, preprocessor

try:
    model, preprocessor = load_resources()
except:
    st.error("Model files not found. Please train the model first.")
    st.stop()

# Sidebar for input method selection
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Input", "File Upload"]
)

if input_method == "Manual Input":
    st.header("Input Mushroom Characteristics")
    
    # Create input fields for each feature
    # Note: These should match your actual feature names
    col1, col2 = st.columns(2)
    
    with col1:
        cap_shape = st.selectbox("Cap Shape", ["bell", "conical", "convex", "flat", "knobbed", "sunken"])
        cap_surface = st.selectbox("Cap Surface", ["fibrous", "grooves", "scaly", "smooth"])
        cap_color = st.selectbox("Cap Color", ["brown", "buff", "cinnamon", "gray", "green", "pink", "purple", "red", "white", "yellow"])
        
    with col2:
        bruises = st.selectbox("Bruises", ["bruises", "no"])
        odor = st.selectbox("Odor", ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"])
        gill_attachment = st.selectbox("Gill Attachment", ["attached", "descending", "free", "notched"])
    
    # Create a dictionary of features
    features = {
        'cap-shape': cap_shape,
        'cap-surface': cap_surface,
        'cap-color': cap_color,
        'bruises': bruises,
        'odor': odor,
        'gill-attachment': gill_attachment
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([features])
    
else:
    st.header("Upload Mushroom Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file with mushroom characteristics.")
        st.stop()

# Make predictions
if st.button("Predict"):
    try:
        # Ensure input_df has all required columns
        required_columns = [
            'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]
        input_df = input_df.reindex(columns=required_columns, fill_value=None)

        # Preprocess the input data
        X_processed = preprocessor.transform(input_df)
        
        # Make predictions
        predictions, probabilities = predict(model, X_processed)
        
        # Display results
        st.header("Prediction Results")
        
        # Show prediction
        st.subheader("Predicted Class")
        st.write(predictions[0])
        
        # Show probabilities
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame(
            probabilities[0],
            columns=model.classes_,
            index=['Probability']
        ).T
        st.bar_chart(prob_df)
        
        # Show feature importance if using Random Forest
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': model.feature_importances_
            })
            st.bar_chart(feature_importance.set_index('Feature'))
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit • Data from UCI Machine Learning Repository</p>
</div>
""", unsafe_allow_html=True) 