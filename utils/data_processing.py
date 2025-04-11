import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st


@st.cache_data
def load_dataset(uploaded_file):
    """Load and cache dataset from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
    
def clear():
    """Clear the cache"""
    st.cache_data.clear()
    st.session_state.clear()
    

def process_features(df, target_column):
    """Process features with efficient handling of categorical and numerical variables"""
    # Make a copy to avoid modifying original data
    X = df.drop(target_column, axis=1)
    
    # Get categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Process numerical columns
    X_processed = X[numerical_columns].copy()
    
    # Process categorical columns using label encoding
    for col in categorical_columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X[col].astype(str))
    
    return X_processed