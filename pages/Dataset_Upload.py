import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils.data_processing import load_dataset
from utils.data_processing import clear
from models.logistic_regression import train_logistic_regression
from models.random_forest import train_random_forest
from model_export.export import export_model
from models.svm import train_svm
from models.KNN import knn_train
from data_cleaning.data_cleaning import data_cleaning_single
from visualization.plots import (
    create_feature_importance_plot,
    create_confusion_matrix_plot,
    create_roc_curve_plot,
    create_decision_boundary_plot,
    create_kernel_comparison_plot,
    create_support_vectors_plot
)
from reports.pdf_generator import create_pdf_report

st.set_page_config(page_icon="📟",
                   page_title="ML Studio",
                   layout="wide",
                   initial_sidebar_state="collapsed")

def display_dataset_stats(df):
    """Display detailed dataset statistics"""
    # Basic statistics
    stats = df.describe().round(2)
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df) * 100).round(2)
    
    # Create null values DataFrame
    null_df = pd.DataFrame({
        'Null Count': null_counts,
        'Null Percentage': null_percentages
    })
    
    return stats, null_df

def main():
    st.title("User Dashboard")
    
    # File upload
    uploaded = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    clr = st.button("Referesh")
    st.info("Before Uploading new dataset, please click on the refresh button to clear the previous dataset from cache.")
    if clr:
        clear()    

    if uploaded is not None:
        # Load dataset once and cache it

        df = load_dataset(uploaded)
        
        if df is not None:
            # Store the dataframe in session state
            if 'data' not in st.session_state:
                st.session_state.data = df
            
            # Use the cached dataframe for all operations
            df = st.session_state.data
            
            # Model selection and parameters
            model = st.selectbox(
                "Select a model", 
                ["Logistic Regression", "Random Forest", "Support Vector Machine","KNN"]
            )
            
            if model == "Logistic Regression":
                st.write("### Logistic Regression Parameters")
                # Add data type validation
                if model in ["Logistic Regression", "Random Forest", "Support Vector Machine"]:
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    target_column = st.selectbox(
                        "Select target column:", 
                        numeric_cols,  # Only show numeric columns
                        help="Select a numeric column for the target variable"
                    )
                st.write("Adjust model parameters as needed.")
                random_state = st.number_input("Random State", min_value=0, value=42, step=1)
                test_size = st.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)
                choice = st.radio("Data Cleaning", ("Yes", "No"))
                if choice == "Yes":
                    
                    type = st.selectbox(
                        "Select a method", 
                        ["mean", "median", "mode", "drop","custom","zero"],
                        help="Select a method to fill NaN values"
                    )
                    if type == "custom":
                        inp_int = st.number_input("Enter a value to fill NaN", min_value=0, value=0, step=1)
                    else:
                        inp_int = 0

                    df = data_cleaning_single(df, type, inp_int, target_column)

            elif model == "Random Forest":
                st.write("### Random Forest Parameters")
                # Add data type validation
                if model in ["Logistic Regression", "Random Forest", "Support Vector Machine"]:
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    target_column = st.selectbox(
                        "Select target column:", 
                        numeric_cols,  # Only show numeric columns
                        help="Select a numeric column for the target variable"
                    )
                st.write("Adjust model parameters as needed.")
                random_state = st.number_input("Random State", min_value=0, value=42, step=1)
                test_size = st.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)
                n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
                max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=10)
                choice = st.radio("Data Cleaning", ("Yes", "No"))
                if choice == "Yes":
                    
                    type = st.selectbox(
                        "Select a method", 
                        ["mean", "median", "mode", "drop","custom","zero"],
                        help="Select a method to fill NaN values"
                    )
                    if type == "custom":
                        inp_int = st.number_input("Enter a value to fill NaN", min_value=0, value=0, step=1)
                    else:
                        inp_int = 0

                    df = data_cleaning_single(df, type, inp_int, target_column)

                
            elif model == "Support Vector Machine":
                st.write("### Support Vector Machine Parameters")
                # Add data type validation
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                target_column_x = st.multiselect(
                    "Select columns:", 
                    numeric_cols,
                    help="Select a numeric column for the target variable"
                )
                target_column = st.selectbox(
                    "Select target column:", 
                    numeric_cols,
                    help="Select a numeric column for the target variable"
                )
                
                # SVM specific parameters
                kernel = st.selectbox(
                    "Kernel Function",
                    ["rbf", "linear", "poly", "sigmoid"],
                    help="The kernel function to be used in the algorithm"
                )
                
                C = st.slider(
                    "Regularization (C)",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.01,
                    help="Regularization parameter. The strength of the regularization is inversely proportional to C"
                )
                
                gamma = st.selectbox(
                    "Gamma",
                    ["scale", "auto"],
                    help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'"
                )
                
                random_state = st.number_input("Random State", min_value=0, value=42, step=1)
                test_size = st.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)
                choice = st.radio("Data Cleaning", ("Yes", "No"))
                if choice == "Yes":
                    
                    type = st.selectbox(
                        "Select a method", 
                        ["mean", "median", "mode", "drop","custom","zero"],
                        help="Select a method to fill NaN values in the target column"
                    )
                    if type == "custom":
                        inp_int = st.number_input("Enter a value to fill NaN", min_value=0, value=0, step=1)
                    else:
                        inp_int = 0

                    df = data_cleaning_single(df, type, inp_int, target_column)
            
            elif model == "KNN":
                st.write("### KNN Parameters")
                # Add data type validation
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                target_column_x = st.multiselect(
                    "Select columns:", 
                    numeric_cols,
                    help="Select a numeric column for the target variable"
                )
                target_column = st.selectbox(
                    "Select target column:", 
                    numeric_cols,
                    help="Select a numeric column for the target variable"
                )
                n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5, step=1)
                random_state = st.number_input("Random State", min_value=0, value=42, step=1)
                test_size = st.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.1)
                choice = st.radio("Data Cleaning", ("Yes", "No"))
                if choice == "Yes":
                    
                    type = st.selectbox(
                        "Select a method", 
                        ["mean", "median", "mode", "drop","custom","zero"],
                        help="Select a method to fill NaN values in the target column"
                    )
                    if type == "custom":
                        inp_int = st.number_input("Enter a value to fill NaN", min_value=0, value=0, step=1)
                    else:
                        inp_int = 0

                    df = data_cleaning_single(df, type, inp_int, target_column)

            col1, col2 = st.columns(2)
            with col1:
                info = st.button("View Dataset Info")
            with col2:
                train = st.button("Train Model")
            
            if info:
                try:
                    # Load and display dataset
                    st.write("### Dataset Preview")
                    st.dataframe(df.head())
                    
                    # Dataset Statistics
                    st.write("### Dataset Statistics")
                    stats, null_info = display_dataset_stats(df)
                    
                    # Display statistics in tabs
                    tab1, tab2, tab3 = st.tabs(["Basic Statistics", "Null Values", "Data Types"])
                    
                    with tab1:
                        st.write("#### Numerical Features Statistics")
                        st.dataframe(stats, use_container_width=True)
                        
                        # Display correlation heatmap
                        st.write("#### Correlation Matrix")
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        if len(numeric_cols) > 0:
                            corr = df[numeric_cols].corr()
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                            st.pyplot(fig)
                            plt.close()
                    
                    with tab2:
                        st.write("#### Missing Values Analysis")
                        # Display null values information
                        st.dataframe(null_info, use_container_width=True)
                        
                        # Bar chart for null values
                        if null_info['Null Count'].sum() > 0:
                            fig = px.bar(
                                null_info[null_info['Null Count'] > 0],
                                y=null_info[null_info['Null Count'] > 0].index,
                                x='Null Count',
                                title='Number of Missing Values by Feature',
                                orientation='h'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.success("No missing values found in the dataset!")
                    
                    with tab3:
                        st.write("#### Data Types Information")
                        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                        st.dataframe(dtype_df, use_container_width=True)
                        
                        # Display count of each data type
                        dtype_counts = df.dtypes.value_counts()
                        fig = px.pie(
                            values=dtype_counts.values,
                            names=dtype_counts.index.astype(str),
                            title='Distribution of Data Types'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Original dataset info
                    st.write("### Dataset Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Number of rows: {df.shape[0]}")
                        st.write(f"Number of columns: {df.shape[1]}")
                    with col2:
                        st.write("Data types:")
                        st.write(df.dtypes)
                    
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
            
            if train:
                with st.spinner(f"Training {model}..."):
                    try:
                        if model == "Logistic Regression":
                            print(df[target_column])
                            results = train_logistic_regression(df, target_column, test_size, random_state)
                            
                            # Check if it's a binary classification problem
                            is_binary = len(np.unique(df[target_column])) == 2
                            
                            # Display results
                            st.success("Model trained successfully!")
                            
                            # Model metrics
                            st.write("### Model Performance")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Accuracy", f"{results['accuracy']:.2%}")
                            with col2:
                                report = classification_report(
                                    results['y_test'], 
                                    results['y_pred'], 
                                    output_dict=True
                                )
                                f1_weighted = report['weighted avg']['f1-score']
                                st.metric("F1 Score (Weighted)", f"{f1_weighted:.2%}")
                            
                            # Classification Report
                            st.write("### Classification Report")
                            st.text(results['classification_report'])
                            
                            # Create visualizations
                            st.write("### Visualizations")
                            
                            # Feature Importance
                            fig_importance = create_feature_importance_plot(
                                pd.DataFrame(
                                    results['feature_importance'].items(),
                                    columns=['Feature', 'Importance']
                                ).sort_values('Importance', ascending=False)
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Confusion Matrix
                            fig_conf_matrix = create_confusion_matrix_plot(results['confusion_matrix'])
                            st.plotly_chart(fig_conf_matrix, use_container_width=True)
                            
                            # ROC Curve (only for binary classification)
                            if is_binary:
                                st.write("### ROC Curve")
                                fig_roc = create_roc_curve_plot(results['y_test'], results['y_pred'])
                                st.plotly_chart(fig_roc, use_container_width=True)
                                # Add ROC curve to figures dictionary for PDF
                                figures = {
                                    "Feature Importance": fig_importance,
                                    "Confusion Matrix": fig_conf_matrix,
                                    "ROC Curve": fig_roc
                                }
                            else:
                                st.info("ROC Curve is only available for binary classification problems.")
                                figures = {
                                    "Feature Importance": fig_importance,
                                    "Confusion Matrix": fig_conf_matrix
                                }
                            
                        elif model == "Random Forest":
                            results = train_random_forest(
                                df, 
                                target_column,
                                test_size=test_size,
                                random_state=random_state,
                                n_estimators=n_estimators,
                                max_depth=max_depth
                            )
                            
                            # Check if it's a binary classification problem
                            is_binary = len(np.unique(df[target_column])) == 2
                            
                            # Display results
                            st.success("Model trained successfully!")
                            
                            # Model metrics
                            st.write("### Model Performance")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Accuracy", f"{results['accuracy']:.2%}")
                            with col2:
                                report = classification_report(
                                    results['y_test'], 
                                    results['y_pred'], 
                                    output_dict=True
                                )
                                f1_weighted = report['weighted avg']['f1-score']
                                st.metric("F1 Score (Weighted)", f"{f1_weighted:.2%}")
                            
                            # Classification Report
                            st.write("### Classification Report")
                            st.text(results['classification_report'])
                            
                            # Create visualizations
                            st.write("### Visualizations")
                            
                            # Feature Importance
                            fig_importance = create_feature_importance_plot(
                                pd.DataFrame(
                                    results['feature_importance'].items(),
                                    columns=['Feature', 'Importance']
                                ).sort_values('Importance', ascending=False)
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Confusion Matrix
                            fig_conf_matrix = create_confusion_matrix_plot(results['confusion_matrix'])
                            st.plotly_chart(fig_conf_matrix, use_container_width=True)
                            
                            # ROC Curve (only for binary classification)
                            if is_binary:
                                st.write("### ROC Curve")
                                fig_roc = create_roc_curve_plot(results['y_test'], results['y_pred'])
                                st.plotly_chart(fig_roc, use_container_width=True)
                                # Add ROC curve to figures dictionary for PDF
                                figures = {
                                    "Feature Importance": fig_importance,
                                    "Confusion Matrix": fig_conf_matrix,
                                    "ROC Curve": fig_roc
                                }
                            else:
                                st.info("ROC Curve is only available for binary classification problems.")
                                figures = {
                                    "Feature Importance": fig_importance,
                                    "Confusion Matrix": fig_conf_matrix
                                }
                        
                        elif model == "Support Vector Machine":
                            results = train_svm(
                                df,
                                target_column_x,
                                target_column,
                                test_size=test_size,
                                random_state=random_state,
                                kernel=kernel,
                                C=C,
                                gamma=gamma
                            )
                            
                            # Display results
                            st.success("Model trained successfully!")
                            
                            # Model metrics
                            st.write("### Model Performance")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Accuracy", f"{results['accuracy']:.2%}")
                            with col2:
                                report = classification_report(
                                    results['y_test'],
                                    results['y_pred'],
                                    output_dict=True
                                )
                                f1_weighted = report['weighted avg']['f1-score']
                                st.metric("F1 Score (Weighted)", f"{f1_weighted:.2%}")
                            
                            # Classification Report
                            st.write("### Classification Report")
                            st.text(results['classification_report'])
                            
                            # Create visualizations
                            st.write("### Visualizations")
                            
                            # Feature Importance
                            fig_importance = create_feature_importance_plot(
                                pd.DataFrame(
                                    results['feature_importance'].items(),
                                    columns=['Feature', 'Importance']
                                ).sort_values('Importance', ascending=False)
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Confusion Matrix
                            fig_conf_matrix = create_confusion_matrix_plot(results['confusion_matrix'])
                            st.plotly_chart(fig_conf_matrix, use_container_width=True)
                            
                            # ROC Curve (only for binary classification)
                            if len(np.unique(df[target_column])) == 2:
                                st.write("### ROC Curve")
                                fig_roc = create_roc_curve_plot(results['y_test'], results['y_pred'])
                                st.plotly_chart(fig_roc, use_container_width=True)
                                # Add ROC curve to figures dictionary for PDF
                                figures = {
                                    "Feature Importance": fig_importance,
                                    "Confusion Matrix": fig_conf_matrix,
                                    "ROC Curve": fig_roc
                                }
                            else:
                                st.info("ROC Curve is only available for binary classification problems.")
                                figures = {
                                    "Feature Importance": fig_importance,
                                    "Confusion Matrix": fig_conf_matrix
                                }
                            
                            # Add SVM-specific visualizations
                            st.write("### SVM Visualizations")
                            results['X_test'] = pd.DataFrame(results['X_test'])
                            # 1. Decision Boundary Plot
                            if results['X_test'].shape[1] <= 2 or st.checkbox("Show 2D Decision Boundary (using PCA)"):
                                fig_boundary = create_decision_boundary_plot(
                                    results['X_test'], results['y_test'], results['model'], results['scaler']
                                )
                                st.plotly_chart(fig_boundary, use_container_width=True)
                            
                            # 2. Kernel Performance Comparison
                            fig_kernel = create_kernel_comparison_plot(
                                results['X_test'], results['y_test'], results['y_pred']
                            )
                            st.plotly_chart(fig_kernel, use_container_width=True)
                            
                            # 3. Support Vectors Visualization
                            fig_sv = create_support_vectors_plot(results['model'], results['X_test'])
                            st.plotly_chart(fig_sv, use_container_width=True)
                        elif model == "KNN":
                            results = knn_train(
                                df,
                                target_column_x, 
                                target_column, 
                                test_size=test_size, 
                                random_state=random_state, 
                                n_neighbors=n_neighbors
                            )
                            
                            # Display results
                            st.success("Model trained successfully!")
                            
                            # Model metrics
                            st.write("### Model Performance")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Accuracy", f"{results['accuracy']:.2%}")
                            with col2:
                                report = classification_report(
                                    results['y_test'], 
                                    results['y_pred'], 
                                    output_dict=True
                                )
                                f1_weighted = report['weighted avg']['f1-score']
                                st.metric("F1 Score (Weighted)", f"{f1_weighted:.2%}")
                            
                            # Classification Report
                            st.write("### Classification Report")
                            st.text(results['classification_report'])
                            
                            # Create visualizations
                            st.write("### Visualizations")

                            # Confusion Matrix
                            fig_conf_matrix = create_confusion_matrix_plot(results['confusion_matrix'])
                            st.plotly_chart(fig_conf_matrix, use_container_width=True)
                            
                        # Generate PDF report
                        pdf_buffer = create_pdf_report(results, df.shape, model)
                        export = export_model(results['trained'],model,"export_path")
                        

                        # Add download button
                        st.download_button(
                            label="Download Report as PDF",
                            data=pdf_buffer,
                            file_name="ml_model_report.pdf",
                            mime="application/pdf"
                        )
                        with open(export, "rb") as f:
                            st.download_button(
                                label="📥 Download Trained Model",
                                data=f,
                                file_name="trained_model.pkl",
                                mime="application/octet-stream"
                            )
                            
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")

if __name__ == "__main__":
    main()







