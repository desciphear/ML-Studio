from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st


@st.cache_data
def knn_train(df, target_column_x ,target_column, test_size=0.2, random_state=42, n_neighbors=5):
    """Train KNN model"""
    try:
        # Validate target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Separate features and target
        X = df[target_column_x]
        y = df[target_column]
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        trained = model.fit(X_train_scaled, y_train)
        
        # Predictions and metrics
        y_pred = model.predict(X_test_scaled)

        
        
        return {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'scaler': scaler,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'trained': trained
        }
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None