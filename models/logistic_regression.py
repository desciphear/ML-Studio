from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from utils.data_processing import process_features

def train_logistic_regression(df, target_column, test_size=0.2, random_state=42):
    """Train logistic regression model"""
    try:
        # Validate target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
       
        # Process features
        X = process_features(df, target_column)
        
          # Fill NaN values in features
        # Convert target to numeric if needed
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        scaler = MaxAbsScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=random_state, solver='saga', n_jobs=-1)
        trained = model.fit(X_train_scaled, y_train)
        
        # Predictions and metrics
        y_pred = model.predict(X_test_scaled)
        
        # Create feature importance dict only if model has coef_ attribute
        feature_importance = {}
        if hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, np.abs(model.coef_[0])))
        
        return {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance,
            'scaler': scaler,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'trained': trained
        }
    
    except Exception as e:
        raise Exception(f"Error in model training: {str(e)}")