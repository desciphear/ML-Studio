from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from utils.data_processing import process_features

def train_svm(df, target_column_x, target_column, test_size=0.2, random_state=42, kernel='rbf', C=1.0, gamma='scale'):
    """Train SVM model"""
    try:
        # Validate target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
            
        # Separate features and target
        X = df[target_column_x]
        y = df[target_column]
        
        # Process features
        
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True
        )
        trained = model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Get feature importance (for linear kernel only)
        feature_importance = {}
        if kernel == 'linear':
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