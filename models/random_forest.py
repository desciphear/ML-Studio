from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from utils.data_processing import process_features

def train_random_forest(df, target_column, test_size=0.2, random_state=42, n_estimators=100, max_depth=10):
    """Train random forest model"""
    # Process features
    print(test_size)
    X = process_features(df, target_column)
    y = df[target_column]
    
    # Convert target to numeric if it's not
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    trained = model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Get feature importance
    feature_importance = dict(zip(
        X.columns,  # Now we have proper column names
        model.feature_importances_
    ))
    
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